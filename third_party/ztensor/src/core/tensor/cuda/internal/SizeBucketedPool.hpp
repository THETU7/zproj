// ztensor/core/tensor/internal/SizeBucketedPool.hpp
//
// Size-bucketed CUDA memory pool for medium-sized allocations (256 KB –
// 16 GB).  Rounds allocations to bucket boundaries and caches freed memory
// per bucket to maximise reuse and reduce fragmentation.
//
// Cache entries are tagged with the stream their last use is ordered on.
// Same-stream reuse is free (stream ordering guarantees safety); cross-stream
// reuse bridges with a GPU-side event edge via bridgeStreams and retags the
// entry.  Evictions and cache-budget enforcement free on the entry's tagged
// stream — freeing on any other stream would be unordered with the last use.
//
// Design (§8.5.E E-3a):
//   - 128 non-uniform logarithmic buckets (256 KB – 16 GB).
//   - Per-bucket mutex (128 independent locks → low contention).
//   - Cache budget = total VRAM / 96, clamped to [64 MiB, 256 MiB].
//   - Per-bucket entry caps: ≤16 MB → 4, ≤64 MB → 3, ≤256 MB → 2, >256 MB → 1.
//   - Large probationary heuristic: one-shot allocations (hits==0 && misses<2)
//     whose bucket exceeds half the budget are freed immediately rather than
//     cached, to avoid blowing out the cache with a transient giant tensor.
//
// Modeled on LichtFeld-Studio's SizeBucketedPool.
//
// This header is visible only when BUILD_CUDA_MODULE is enabled
// (same guard discipline as the public zt/cuda/{Exception,Guard,Stream}.h).

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include "ztensor/zt/cuda/Exception.h"  // ZT_CUDA_CHECK_SOFT
#include "ztensor/zt/cuda/Vendor.h"

#include "core/cuda/CUDAEventPool.h"  // bridgeStreams

namespace zt {

// ── SizeBucketedPool
// ───────────────────────────────────────────────────────────

class SizeBucketedPool {
public:
    static constexpr size_t MIN_BUCKET_SIZE = 256 * 1024;  // 256 KiB
    static constexpr size_t MAX_TRACKED_SIZE =
        16ULL * 1024 * 1024 * 1024;  // 16 GiB
    static constexpr size_t CACHE_SIZE_PER_BUCKET = 4;
    static constexpr size_t MIN_CACHE_BUDGET = 64ULL * 1024 * 1024;   // 64 MiB
    static constexpr size_t MAX_CACHE_BUDGET = 256ULL * 1024 * 1024;  // 256 MiB
    static constexpr size_t NUM_BUCKETS = 128;

    struct Stats {
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<uint64_t> alloc_count{0};
        std::atomic<uint64_t> free_count{0};
        std::atomic<uint64_t> bytes_cached{0};
        std::atomic<uint64_t> bytes_wasted{0};
        std::atomic<uint64_t> cross_stream_reuse{0};
    };

    static SizeBucketedPool& instance() {
        static SizeBucketedPool pool;
        return pool;
    }

    void shutdown() {
        bool expected = false;
        if (!shutdown_.compare_exchange_strong(expected, true)) return;
        trim_cache();
    }

    // ── Bucket sizing
    // ──────────────────────────────────────────────────────────
    //
    // Non-uniform logarithmic buckets:
    //   ≤256 KiB  → 256 KiB
    //   ≤1 MiB    → ceil to 256 KiB
    //   ≤16 MiB   → ceil to 1 MiB
    //   ≤256 MiB  → ceil to 16 MiB
    //   ≤1 GiB    → ceil to 64 MiB
    //   ≤8 GiB    → ceil to 256 MiB
    //   >8 GiB    → ceil to 1 GiB

    static size_t get_bucket_size(size_t bytes) {
        if (bytes <= MIN_BUCKET_SIZE) return MIN_BUCKET_SIZE;
        if (bytes <= 1024 * 1024)
            return ((bytes + 256 * 1024 - 1) / (256 * 1024)) * (256 * 1024);
        if (bytes <= 16 * 1024 * 1024)
            return ((bytes + 1024 * 1024 - 1) / (1024 * 1024)) * (1024 * 1024);
        if (bytes <= 256 * 1024 * 1024)
            return ((bytes + 16 * 1024 * 1024 - 1) / (16 * 1024 * 1024)) *
                   (16 * 1024 * 1024);
        if (bytes <= 1024ULL * 1024 * 1024)
            return ((bytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) *
                   (64 * 1024 * 1024);
        if (bytes <= 8ULL * 1024 * 1024 * 1024)
            return ((bytes + 256ULL * 1024 * 1024 - 1) /
                    (256ULL * 1024 * 1024)) *
                   (256ULL * 1024 * 1024);
        return ((bytes + 1024ULL * 1024 * 1024 - 1) / (1024ULL * 1024 * 1024)) *
               (1024ULL * 1024 * 1024);
    }

    static size_t get_bucket_index(size_t bucket_size) {
        if (bucket_size <= 1024 * 1024) return (bucket_size / (256 * 1024)) - 1;
        if (bucket_size <= 16 * 1024 * 1024)
            return 4 + (bucket_size / (1024 * 1024)) - 1;
        if (bucket_size <= 256 * 1024 * 1024)
            return 20 + (bucket_size / (16 * 1024 * 1024)) - 1;
        if (bucket_size <= 1024ULL * 1024 * 1024)
            return 36 + (bucket_size / (64 * 1024 * 1024)) - 4;
        if (bucket_size <= 8ULL * 1024 * 1024 * 1024)
            return 48 + (bucket_size / (256ULL * 1024 * 1024)) - 4;
        const size_t idx = 76 + (bucket_size / (1024ULL * 1024 * 1024)) - 8;
        return std::min(idx, NUM_BUCKETS - 1);
    }

    static size_t max_cached_entries_for_bucket(size_t bucket_size) {
        if (bucket_size <= 16ULL * 1024 * 1024)
            return CACHE_SIZE_PER_BUCKET;  // 4
        if (bucket_size <= 64ULL * 1024 * 1024) return 3;
        if (bucket_size <= 256ULL * 1024 * 1024) return 2;
        return 1;
    }

    static size_t cache_budget_for_total_memory(size_t total_bytes) {
        if (total_bytes == 0) return MAX_CACHE_BUDGET;
        return std::clamp(total_bytes / 96, MIN_CACHE_BUDGET, MAX_CACHE_BUDGET);
    }

    // ── Primary interface
    // ──────────────────────────────────────────────────────

    // Try to serve `bytes` from the cache.  Returns nullptr on miss (caller
    // should allocate fresh).  On a cross-stream hit, bridges the old stream
    // into `stream` and retags the entry.
    void* try_allocate_cached(size_t bytes, cudaStream_t stream = nullptr) {
        const size_t bucket_size = get_bucket_size(bytes);
        const size_t bucket_idx = get_bucket_index(bucket_size);
        if (bucket_idx >= NUM_BUCKETS) return nullptr;

        {
            std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
            Bucket& bucket = buckets_[bucket_idx];
            bucket.bucket_size = bucket_size;
            if (!bucket.cache.empty()) {
                // Prefer same-stream entry (LIFO, safe by stream ordering).
                size_t pick = bucket.cache.size() - 1;
                for (size_t i = bucket.cache.size(); i-- > 0;) {
                    if (bucket.cache[i].stream == stream) {
                        pick = i;
                        break;
                    }
                }
                const CachedBlock block = bucket.cache[pick];
                bucket.cache.erase(bucket.cache.begin() + pick);
                if (block.stream != stream) {
                    ::zt::cuda::bridgeStreams(block.stream, stream);
                    stats_.cross_stream_reuse.fetch_add(
                        1, std::memory_order_relaxed);
                }
                bucket.cached_bytes -= bucket_size;
                bucket.hits++;
                bucket.last_hit_epoch =
                    reuse_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
                stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_cached.fetch_sub(bucket_size,
                                              std::memory_order_relaxed);
                stats_.bytes_wasted.fetch_add(bucket_size - bytes,
                                              std::memory_order_relaxed);
                return block.ptr;
            }
            bucket.misses++;
        }
        stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    // Attempt to cache a freed block.  `stream` must be the stream the block's
    // last use is ordered on.  Returns true if cached, false if the block was
    // freed immediately (bucket index out of range, or probationary eviction).
    bool cache_free(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
        const size_t bucket_size = get_bucket_size(bytes);
        const size_t bucket_idx = get_bucket_index(bucket_size);
        if (bucket_idx >= NUM_BUCKETS) return false;

        {
            std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
            Bucket& bucket = buckets_[bucket_idx];
            bucket.bucket_size = bucket_size;

            // Large probationary heuristic: skip caching one-shot giant allocs.
            const size_t budget = current_cache_budget();
            const bool large_probationary = bucket_size > budget / 2 &&
                                            bucket.hits == 0 &&
                                            bucket.misses < 2;
            if (large_probationary) {
                ZT_CUDA_CHECK_SOFT(cudaFreeAsync(ptr, stream));
                return true;  // freed, not cached
            }

            // Evict oldest entries if the per-bucket cap is reached.
            const size_t max_entries =
                max_cached_entries_for_bucket(bucket_size);
            while (bucket.cache.size() >= max_entries) {
                const CachedBlock old = bucket.cache.front();
                bucket.cache.erase(bucket.cache.begin());
                bucket.cached_bytes -= bucket_size;
                stats_.bytes_cached.fetch_sub(bucket_size,
                                              std::memory_order_relaxed);
                ZT_CUDA_CHECK_SOFT(cudaFreeAsync(old.ptr, old.stream));
            }

            bucket.cache.push_back({ptr, stream});
            bucket.cached_bytes += bucket_size;
            stats_.free_count.fetch_add(1, std::memory_order_relaxed);
            stats_.bytes_cached.fetch_add(bucket_size,
                                          std::memory_order_relaxed);
        }

        enforce_cache_budget();
        return true;
    }

    // Allocate `bytes` from the bucketed pool (try cache first, then allocate).
    void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
        void* ptr = try_allocate_cached(bytes, stream);
        if (ptr) return ptr;

        const size_t bucket_size = get_bucket_size(bytes);
        cudaError_t err = cudaMallocAsync(&ptr, bucket_size, stream);
        if (err != cudaSuccess) {
            trim_cache();
            err = cudaMallocAsync(&ptr, bucket_size, stream);
            if (err != cudaSuccess) {
                (void)cudaGetLastError();  // clear sticky error
                return nullptr;
            }
        }
        stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_wasted.fetch_add(bucket_size - bytes,
                                      std::memory_order_relaxed);
        return ptr;
    }

    // Deallocate `bytes` at `ptr`, last used on `stream`.  Tries to cache;
    // frees immediately if caching is not possible.
    void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
        if (!ptr) return;
        if (!cache_free(ptr, bytes, stream)) {
            ZT_CUDA_CHECK_SOFT(cudaFreeAsync(ptr, stream));
        }
    }

    // ── Stream re-tagging
    // ──────────────────────────────────────────────────────

    // Re-tag cached entries from `from` to `to`.  Caller must have synchronised
    // `from` first — once the stream is idle, its entries can safely be reused
    // on `to` without an event edge.
    void retag_stream(cudaStream_t from, cudaStream_t to) {
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            std::lock_guard<std::mutex> lock(buckets_[i].mutex);
            for (CachedBlock& block : buckets_[i].cache) {
                if (block.stream == from) {
                    block.stream = to;
                }
            }
        }
    }

    // Re-tag every cached entry to `to`.  Caller must have synchronised the
    // device first.
    void retag_all_streams(cudaStream_t to) {
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            std::lock_guard<std::mutex> lock(buckets_[i].mutex);
            for (CachedBlock& block : buckets_[i].cache) {
                block.stream = to;
            }
        }
    }

    // ── Cache trimming
    // ─────────────────────────────────────────────────────────

    // Free all cached entries (stream-ordered on their last-use stream).
    void trim_cache() {
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            std::lock_guard<std::mutex> lock(buckets_[i].mutex);
            for (const CachedBlock& block : buckets_[i].cache) {
                ZT_CUDA_CHECK_SOFT(cudaFreeAsync(block.ptr, block.stream));
            }
            buckets_[i].cache.clear();
            buckets_[i].cached_bytes = 0;
        }
        stats_.bytes_cached.store(0, std::memory_order_relaxed);
    }

    // ── Statistics
    // ─────────────────────────────────────────────────────────────

    const Stats& stats() const { return stats_; }

    void print_stats() const {
        uint64_t hits = stats_.cache_hits.load();
        uint64_t misses = stats_.cache_misses.load();
        double hit_rate =
            (hits + misses > 0) ? (100.0 * hits / (hits + misses)) : 0.0;
        ZT_LOG_INFO(
            "SizeBucketedPool: hits={} ({:.1f}%) misses={} "
            "cached={:.2f} MiB wasted={:.2f} MiB cross_stream={}",
            hits,
            hit_rate,
            misses,
            stats_.bytes_cached.load() / (1024.0 * 1024.0),
            stats_.bytes_wasted.load() / (1024.0 * 1024.0),
            stats_.cross_stream_reuse.load());
    }

    // Waste percentage for a hypothetical allocation.
    static double get_waste_percentage(size_t bytes) {
        size_t bucket = get_bucket_size(bytes);
        return 100.0 * (bucket - bytes) / bucket;
    }

    SizeBucketedPool(const SizeBucketedPool&) = delete;
    SizeBucketedPool& operator=(const SizeBucketedPool&) = delete;

private:
    SizeBucketedPool() = default;
    ~SizeBucketedPool() { shutdown(); }

    // ── Internal helpers
    // ──────────────────────────────────────────────────────

    size_t current_cache_budget() {
        const size_t cached =
            cache_budget_bytes_.load(std::memory_order_acquire);
        if (cached != 0) return cached;

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        size_t budget = MAX_CACHE_BUDGET;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            budget = cache_budget_for_total_memory(total_bytes);
        }

        size_t expected = 0;
        if (cache_budget_bytes_.compare_exchange_strong(
                expected, budget, std::memory_order_release)) {
            return budget;
        }
        return cache_budget_bytes_.load(std::memory_order_acquire);
    }

    size_t cached_entry_count() {
        size_t count = 0;
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            std::lock_guard<std::mutex> lock(buckets_[i].mutex);
            count += buckets_[i].cache.size();
        }
        return count;
    }

    // Pick the least-recently-hit non-empty bucket (LRU-ish eviction).
    size_t choose_eviction_bucket() {
        size_t best = NUM_BUCKETS;
        uint64_t best_epoch = std::numeric_limits<uint64_t>::max();
        size_t best_bucket_size = 0;

        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            std::lock_guard<std::mutex> lock(buckets_[i].mutex);
            const Bucket& bucket = buckets_[i];
            if (bucket.cache.empty() || bucket.bucket_size == 0) continue;

            const uint64_t epoch = bucket.last_hit_epoch;
            if (best == NUM_BUCKETS || epoch < best_epoch ||
                (epoch == best_epoch &&
                 bucket.bucket_size > best_bucket_size)) {
                best = i;
                best_epoch = epoch;
                best_bucket_size = bucket.bucket_size;
            }
        }
        return best;
    }

    void enforce_cache_budget() {
        const size_t budget = current_cache_budget();
        while (stats_.bytes_cached.load(std::memory_order_relaxed) > budget) {
            if (cached_entry_count() <= 1) break;  // keep at least one entry

            const size_t victim_idx = choose_eviction_bucket();
            if (victim_idx >= NUM_BUCKETS) break;

            CachedBlock victim{};
            size_t victim_size = 0;
            {
                std::lock_guard<std::mutex> lock(buckets_[victim_idx].mutex);
                Bucket& bucket = buckets_[victim_idx];
                if (bucket.cache.empty() || bucket.bucket_size == 0) continue;
                victim = bucket.cache.front();
                bucket.cache.erase(bucket.cache.begin());
                victim_size = bucket.bucket_size;
                bucket.cached_bytes -= victim_size;
                stats_.bytes_cached.fetch_sub(victim_size,
                                              std::memory_order_relaxed);
            }
            ZT_CUDA_CHECK_SOFT(cudaFreeAsync(victim.ptr, victim.stream));
        }
    }

    // ── Data
    // ───────────────────────────────────────────────────────────────────

    struct CachedBlock {
        void* ptr = nullptr;
        cudaStream_t stream = nullptr;
    };

    struct Bucket {
        std::vector<CachedBlock> cache;
        std::mutex mutex;
        size_t cached_bytes{0};
        size_t bucket_size{0};
        uint64_t hits{0};
        uint64_t misses{0};
        uint64_t last_hit_epoch{0};

        Bucket() { cache.reserve(CACHE_SIZE_PER_BUCKET); }
    };

    std::array<Bucket, NUM_BUCKETS> buckets_;
    std::atomic<bool> shutdown_{false};
    std::atomic<size_t> cache_budget_bytes_{0};
    std::atomic<uint64_t> reuse_epoch_{1};
    Stats stats_;
};

}  // namespace zt

#endif  // BUILD_CUDA_MODULE
