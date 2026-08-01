// ztensor/core/CudaMemoryPool.h
//
// Four-tier CUDA memory pool: the single public entry point for all CUDA
// allocations.  Dispatches on request size:
//
//   Tier 1  ≤ 256 KiB  → GPUSlabAllocator (power-of-2 size classes, slabs)
//   Tier 2  ≤ 16 GiB   → SizeBucketedPool (128 log buckets, stream-tagged LIFO)
//   Tier 3  > 16 GiB    → cudaMallocAsync (driver stream-ordered pool)
//          or bucket miss
//   Tier 4  fallback    → cudaMalloc (direct allocation)
//
// Every allocation is tracked in allocation_map_ so deallocate() routes the
// pointer back to the correct backend.  Stream-aware: record_stream() /
// rehome_stream() / release_stream() manage cross-stream lifetimes.
//
// Design (§8.5.E E-3c): Meyers singleton, transparent to MemoryManager/Blob.
// Modeled on LichtFeld-Studio's CudaMemoryPool.
//
// This header is visible only when BUILD_CUDA_MODULE is enabled
// (same guard discipline as CUDAUtils.h / CUDAEventPool.h).

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace zt {

// ── CudaMemoryPool ─────────────────────────────────────────────────────────────

class CudaMemoryPool {
public:
    // ── Constants ──────────────────────────────────────────────────────────────

    static constexpr size_t SLAB_ALLOC_THRESHOLD = 256 * 1024;            // 256 KiB
    static constexpr size_t BUCKET_ALLOC_THRESHOLD = 16ULL * 1024ULL * 1024ULL * 1024ULL;  // 16 GiB

    // ── Allocation method tag ──────────────────────────────────────────────────

    enum class AllocMethod : uint8_t {
        Slab = 0,
        Bucketed,
        Async,
        Direct
    };

    // ── Singleton ──────────────────────────────────────────────────────────────

    static CudaMemoryPool& instance();

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    // Configure the driver memory pool (ReleaseThreshold = 64 MiB).
    // Called lazily on first allocate(); safe to call multiple times.
    void configure();

    // CAS-guarded; subsequent calls are no-ops.  Drains all sub-pools.
    void shutdown();

    // Signal that the process is exiting — skip further deallocations so the
    // OS / CUDA context can clean up without late-shutdown errors.
    void suspend_deallocations_for_process_exit() {
        suspend_deallocations_.store(true, std::memory_order_release);
    }

    // ── Primary interface ──────────────────────────────────────────────────────

    // Allocate `bytes` of CUDA memory, ordered after all prior work on `stream`.
    // Returns nullptr only when the driver truly cannot satisfy the request.
    void* allocate(size_t bytes, cudaStream_t stream = nullptr);

    // Return memory previously obtained from allocate().  `stream` must be the
    // stream the block's last use is ordered on.
    void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr);
    void deallocate(void* ptr, cudaStream_t stream = nullptr);

    // ── Stream-crossing helpers ────────────────────────────────────────────────

    // Mark `ptr` as used by `stream` in addition to its home stream.  On free,
    // the extra stream is bridged back to the home stream before recycling.
    void record_stream(void* ptr, cudaStream_t stream);

    // Move `ptr`'s home stream to `stream`.  The old home becomes a recorded
    // extra use.  Future frees on the new home are safe without bridging.
    void rehome_stream(void* ptr, cudaStream_t stream);

    // Sever every allocator reference to `stream` so it can be safely destroyed
    // via cudaStreamDestroy.  Synchronises the stream, drops it from all
    // recorded uses, re-homes live allocations to nullptr, and migrates cached
    // free-list entries.  Must be called before cudaStreamDestroy on any stream
    // that has touched pool memory.
    void release_stream(cudaStream_t stream);

    // ── Maintenance ────────────────────────────────────────────────────────────

    // Trim cached memory across all tiers and the driver pool.
    void trim();

    // Full trim after device sync: flush deferred frees, clear extra streams,
    // merge all slab free-lists into virgin, retag all bucketed entries, trim
    // cache, and trim the driver pool to zero.
    void trim_cached_memory();

    // ── Statistics ─────────────────────────────────────────────────────────────

    struct Stats {
        std::atomic<uint64_t> slab_allocs{0};
        std::atomic<uint64_t> slab_bytes{0};
        std::atomic<uint64_t> bucket_allocs{0};
        std::atomic<uint64_t> bucket_cache_hits{0};
        std::atomic<uint64_t> bucket_bytes{0};
        std::atomic<uint64_t> bucket_waste{0};
        std::atomic<uint64_t> async_allocs{0};
        std::atomic<uint64_t> async_bytes{0};
        std::atomic<uint64_t> direct_allocs{0};
        std::atomic<uint64_t> direct_bytes{0};
    };

    const Stats& stats() const { return stats_; }

    std::string get_stats_string() const;
    void print_stats() const;

    // ── Cache bypass ───────────────────────────────────────────────────────────

    // True when ZT_DISABLE_CUDA_CACHE=1 is set in the environment.
    static bool cache_disabled_by_env();

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

private:
    // ── Internal structures ────────────────────────────────────────────────────

    struct AllocationInfo {
        size_t size = 0;
        AllocMethod method = AllocMethod::Direct;
        cudaStream_t home_stream = nullptr;
        std::vector<cudaStream_t> extra_streams;
    };

    // ── Construction ───────────────────────────────────────────────────────────

    CudaMemoryPool();
    ~CudaMemoryPool();

    // ── Internal helpers ──────────────────────────────────────────────────────

    void track_allocation(void* ptr,
                          size_t size,
                          AllocMethod method,
                          cudaStream_t stream = nullptr);

    // Remove `ptr` from allocation_map_ and return its info.  Returns true if
    // the pointer was tracked.
    bool take_allocation(void* ptr, AllocationInfo& info);

    // Bridge every recorded cross-stream use into the home stream, then free.
    void free_routed(void* ptr, const AllocationInfo& info);

    // Last-resort direct allocation with trim-and-retry.
    void* allocate_direct(size_t bytes);

    // ── Data members ──────────────────────────────────────────────────────────

    std::unordered_map<void*, AllocationInfo> allocation_map_;
    std::mutex map_mutex_;
    std::atomic<size_t> direct_alloc_count_{0};
    bool slab_enabled_{false};
    bool configured_{false};
    std::once_flag configure_once_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> suspend_deallocations_{false};
    Stats stats_;
};

}  // namespace zt

#endif  // BUILD_CUDA_MODULE
