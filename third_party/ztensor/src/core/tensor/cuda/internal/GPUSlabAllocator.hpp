// ztensor/core/tensor/internal/GPUSlabAllocator.hpp
//
// GPU slab allocator for small allocations (≤ 256 KiB).  Slabs are committed
// on first use per size class and divided into fixed-size blocks.
//
// Free lists are kept per stream: a block freed on stream S is immediately
// reusable on S (safe by stream ordering).  Fresh slab blocks live in a
// "virgin" list and are stream-free.  Cross-stream reuse bridges with a
// GPU-side event edge via bridgeStreams, so reuse never needs a host sync.
//
// Design (§8.5.E E-3b):
//   - 11 power-of-2 size classes (256 B … 256 KiB).
//   - Per-class mutex + independent slabs_mutex_.
//   - Lazy commit: slab allocated on first touch (double-check).
//   - FreeLists: {per_stream, virgin, mutex, atomic count}.
//   - Cross-stream stealing deferred (TODO(E-enhance)).
//   - owns_pointer(p): linear scan of slab address ranges.
//
// Modeled on LichtFeld-Studio's GPUSlabAllocator.
//
// This header is visible only when BUILD_CUDA_MODULE is enabled
// (same guard discipline as CUDAUtils.h / CUDAEventPool.h).

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda_runtime.h>

#include "core/cuda/CUDAEventPool.h"  // bridgeStreams

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ztensor/zt/utility/Log.h"

namespace zt {

// ── GPUSlabAllocator ───────────────────────────────────────────────────────────

class GPUSlabAllocator {
public:
    static constexpr size_t MIN_BLOCK_SIZE = 256;
    static constexpr size_t MAX_BLOCK_SIZE = 256 * 1024;         // 256 KiB
    static constexpr size_t NUM_SIZE_CLASSES = 11;               // 256B … 256KiB
    static constexpr size_t MIN_SLAB_SIZE = 256 * 1024;          // 256 KiB
    static constexpr size_t MAX_SLAB_SIZE = 8 * 1024 * 1024;     // 8 MiB
    static constexpr size_t TARGET_BLOCKS_PER_SLAB = 1024;
    static constexpr size_t MAX_BLOCKS_PER_CLASS = 512 * 1024;   // tracked cap

    struct Stats {
        std::atomic<uint64_t> alloc_count{0};
        std::atomic<uint64_t> free_count{0};
        std::atomic<uint64_t> miss_count{0};
        std::atomic<uint64_t> steal_count{0};  // always 0 until TODO(E-enhance)
        size_t total_slab_memory{0};
        size_t blocks_per_class[NUM_SIZE_CLASSES]{0};
    };

    static GPUSlabAllocator& instance();

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    void shutdown() {
        bool expected = false;
        if (!shutdown_.compare_exchange_strong(expected, true)) return;
        enabled_.store(false, std::memory_order_release);
        cleanup();
    }

    // ── Size-class helpers ─────────────────────────────────────────────────────

    static size_t get_size_class(size_t bytes) {
        if (bytes <= MIN_BLOCK_SIZE) return 0;
        size_t size = MIN_BLOCK_SIZE;
        size_t class_idx = 0;
        while (size < bytes && class_idx < NUM_SIZE_CLASSES - 1) {
            size *= 2;
            class_idx++;
        }
        return class_idx;
    }

    static size_t get_block_size(size_t size_class) {
        return MIN_BLOCK_SIZE << size_class;
    }

    static size_t slab_size_for_class(size_t size_class) {
        const size_t block_size = get_block_size(size_class);
        const size_t target_bytes = block_size * TARGET_BLOCKS_PER_SLAB;
        const size_t slab_size =
            std::min(std::max(target_bytes, MIN_SLAB_SIZE), MAX_SLAB_SIZE);
        return (slab_size / block_size) * block_size;  // round down to block multiple
    }

    // ── Allocation / deallocation ──────────────────────────────────────────────

    // Returns nullptr on miss (caller falls through to next tier).
    void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
        if (!enabled_.load(std::memory_order_acquire) ||
            bytes == 0 ||
            bytes > MAX_BLOCK_SIZE) {
            return nullptr;
        }

        const size_t size_class = get_size_class(bytes);
        if (size_class >= NUM_SIZE_CLASSES) return nullptr;

        void* ptr = pop_block(size_class, stream);
        if (ptr) {
            stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
            return ptr;
        }

        // Try to expand the slab for this size class.
        if (expand_slab(size_class)) {
            ptr = pop_block(size_class, stream);
            if (ptr) {
                stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
                return ptr;
            }
        }

        stats_.miss_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    // `stream` must be the stream the block's last use is ordered on.
    void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
        if (!ptr || bytes == 0 || bytes > MAX_BLOCK_SIZE) return;

        const size_t size_class = get_size_class(bytes);
        if (size_class >= NUM_SIZE_CLASSES) return;

        push_block(size_class, ptr, stream);
        stats_.free_count.fetch_add(1, std::memory_order_relaxed);
    }

    // ── Pointer ownership ─────────────────────────────────────────────────────

    bool owns_pointer(void* ptr) const {
        if (!ptr) return false;
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

        std::lock_guard<std::mutex> lock(slabs_mutex_);
        for (const auto& slab : slabs_) {
            uintptr_t slab_start = reinterpret_cast<uintptr_t>(slab.base);
            uintptr_t slab_end = slab_start + slab.size;
            if (addr >= slab_start && addr < slab_end) return true;
        }
        return false;
    }

    // ── Stream re-tagging ─────────────────────────────────────────────────────

    // Move `stream`'s free-list entries to the virgin list.  Caller must have
    // synchronised the stream (or device) first.
    void merge_stream_into_virgin(cudaStream_t stream) {
        for (auto& lists : free_lists_) {
            std::lock_guard<std::mutex> lock(lists.mutex);
            auto it = lists.per_stream.find(stream);
            if (it == lists.per_stream.end()) continue;
            lists.virgin.insert(
                lists.virgin.end(), it->second.begin(), it->second.end());
            lists.per_stream.erase(it);
        }
    }

    // Same for every stream.  Caller must have synchronised the device.
    void merge_all_streams_into_virgin() {
        for (auto& lists : free_lists_) {
            std::lock_guard<std::mutex> lock(lists.mutex);
            for (auto& [stream, blocks] : lists.per_stream) {
                lists.virgin.insert(
                    lists.virgin.end(), blocks.begin(), blocks.end());
            }
            lists.per_stream.clear();
        }
    }

    // ── State ─────────────────────────────────────────────────────────────────

    bool is_enabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    const Stats& stats() const { return stats_; }

    void print_stats() const {
        ZT_LOG_INFO(
            "GPUSlabAllocator: slab_mem={:.2f} MiB allocs={} frees={} "
            "misses={} steals={}",
            stats_.total_slab_memory / (1024.0 * 1024.0),
            stats_.alloc_count.load(),
            stats_.free_count.load(),
            stats_.miss_count.load(),
            stats_.steal_count.load());
        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            if (stats_.blocks_per_class[i] > 0) {
                ZT_LOG_INFO("  class {} ({} B): {} blocks",
                            i,
                            get_block_size(i),
                            stats_.blocks_per_class[i]);
            }
        }
    }

    GPUSlabAllocator(const GPUSlabAllocator&) = delete;
    GPUSlabAllocator& operator=(const GPUSlabAllocator&) = delete;

private:
    // ── Internal structures ───────────────────────────────────────────────────

    struct Slab {
        void* base = nullptr;
        size_t size = 0;
        size_t size_class = 0;
    };

    struct FreeLists {
        std::unordered_map<cudaStream_t, std::vector<void*>> per_stream;
        std::vector<void*> virgin;
        std::mutex mutex;
        std::atomic<size_t> count{0};
    };

    // ── Construction ──────────────────────────────────────────────────────────

    GPUSlabAllocator() {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            enabled_.store(false, std::memory_order_release);
            return;
        }

        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            const size_t initial_blocks =
                slab_size_for_class(i) / get_block_size(i);
            free_lists_[i].virgin.reserve(
                std::min(initial_blocks, MAX_BLOCKS_PER_CLASS));
        }

        enabled_.store(true, std::memory_order_release);
    }

    ~GPUSlabAllocator() { shutdown(); }

    // ── Slab management ──────────────────────────────────────────────────────

    bool allocate_slab(size_t size_class) {
        const size_t block_size = get_block_size(size_class);
        const size_t slab_size = slab_size_for_class(size_class);

        void* slab_base = nullptr;
        if (cudaMalloc(&slab_base, slab_size) != cudaSuccess) return false;

        const size_t num_blocks = slab_size / block_size;
        {
            std::lock_guard<std::mutex> lock(free_lists_[size_class].mutex);
            for (size_t i = 0; i < num_blocks; ++i) {
                void* block =
                    static_cast<char*>(slab_base) + i * block_size;
                free_lists_[size_class].virgin.push_back(block);
            }
            free_lists_[size_class].count.fetch_add(num_blocks,
                                                    std::memory_order_release);
        }

        {
            std::lock_guard<std::mutex> lock(slabs_mutex_);
            slabs_.push_back({slab_base, slab_size, size_class});
        }

        stats_.total_slab_memory += slab_size;
        stats_.blocks_per_class[size_class] += num_blocks;
        return true;
    }

    bool expand_slab(size_t size_class) {
        static std::mutex expand_mutex;
        std::lock_guard<std::mutex> lock(expand_mutex);
        // Double-check: another thread may have expanded while we waited.
        if (free_lists_[size_class].count.load(std::memory_order_acquire) > 0) {
            return true;
        }
        return allocate_slab(size_class);
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(slabs_mutex_);
        for (const auto& slab : slabs_) {
            cudaFree(slab.base);
        }
        slabs_.clear();
        stats_.total_slab_memory = 0;
    }

    // ── Block-level operations ────────────────────────────────────────────────

    void* pop_block(size_t size_class, cudaStream_t stream) {
        FreeLists& lists = free_lists_[size_class];
        if (lists.count.load(std::memory_order_acquire) == 0) return nullptr;

        std::lock_guard<std::mutex> lock(lists.mutex);

        // 1. Try same-stream reuse (fast path, no sync needed).
        if (auto it = lists.per_stream.find(stream);
            it != lists.per_stream.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            lists.count.fetch_sub(1, std::memory_order_release);
            return ptr;
        }

        // 2. Try virgin blocks (never used, stream-free).
        if (!lists.virgin.empty()) {
            void* ptr = lists.virgin.back();
            lists.virgin.pop_back();
            lists.count.fetch_sub(1, std::memory_order_release);
            return ptr;
        }

        // 3. Cross-stream stealing: deferred to TODO(E-enhance).
        //    For now, return nullptr so the caller falls through to the
        //    bucketed pool.  Once stealing is implemented the logic below
        //    this point will search the richest other stream and bridge.
        return nullptr;
    }

    void push_block(size_t size_class, void* ptr, cudaStream_t stream) {
        FreeLists& lists = free_lists_[size_class];
        std::lock_guard<std::mutex> lock(lists.mutex);
        lists.per_stream[stream].push_back(ptr);
        lists.count.fetch_add(1, std::memory_order_release);
    }

    // ── Data members ──────────────────────────────────────────────────────────

    std::array<FreeLists, NUM_SIZE_CLASSES> free_lists_;
    std::vector<Slab> slabs_;
    mutable std::mutex slabs_mutex_;
    Stats stats_;
    std::atomic<bool> enabled_{false};
    std::atomic<bool> shutdown_{false};
};

// ── Singleton accessor (out-of-line inline) ────────────────────────────────────

inline GPUSlabAllocator& GPUSlabAllocator::instance() {
    static GPUSlabAllocator allocator;
    return allocator;
}

}  // namespace zt

#endif  // BUILD_CUDA_MODULE
