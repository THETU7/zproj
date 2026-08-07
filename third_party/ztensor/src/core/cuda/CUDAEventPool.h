// ztensor/core/CUDAEventPool.h
//
// Process-wide pool of cudaEventDisableTiming events.  Cross-stream waits and
// deferred frees record events in hot paths; pooling avoids per-call
// cudaEventCreate / cudaEventDestroy overhead.
//
// Modeled on LichtFeld-Studio's CudaEventPool.
//
// This header is visible only when BUILD_CUDA_MODULE is enabled
// (same guard discipline as the public zt/cuda/{Exception,Guard,Stream}.h).

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "ztensor/zt/cuda/Vendor.h"

namespace zt {
namespace cuda {

// ── CudaEventPool ────────────────────────────────────────────────────────────

// Meyers-singleton pool of cudaEvent_t objects created with
// cudaEventDisableTiming.  acquire() returns a pooled event (or creates one
// when the pool is empty); release() returns it to the pool (or destroys it
// when the pool is full).  shutdown() drains and destroys all pooled events.
class CudaEventPool {
public:
    static constexpr size_t MAX_POOL_SIZE = 512;

    static CudaEventPool& instance();

    // Return a pooled event, or create one if the pool is empty.
    // Returns nullptr if event creation fails (no CUDA context, OOM).
    cudaEvent_t acquire();

    // Return `event` to the pool.  Safe to call with an event that has a
    // pending cudaStreamWaitEvent — the wait snapshots the record it saw;
    // later re-record or destroy does not affect it.
    void release(cudaEvent_t event);

    // Drain and destroy all pooled events.  CAS-guarded, safe to call
    // multiple times (subsequent calls are no-ops).
    void shutdown();

    // ── Statistics ───────────────────────────────────────────────────────────

    struct Stats {
        std::atomic<uint64_t> created{0};
        std::atomic<uint64_t> reused{0};
    };

    const Stats& stats() const { return stats_; }

    // Number of events currently in the pool (for tests/debugging).
    size_t pooled_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

    CudaEventPool(const CudaEventPool&) = delete;
    CudaEventPool& operator=(const CudaEventPool&) = delete;

private:
    CudaEventPool() = default;
    ~CudaEventPool();

    std::vector<cudaEvent_t> pool_;
    mutable std::mutex mutex_;
    std::atomic<bool> shutdown_{false};
    Stats stats_;
};

// ── bridgeStreams ────────────────────────────────────────────────────────────

// Order all work currently enqueued on `from` before future work on `to`
// using a pooled event edge with a host-sync fallback.  Unlike
// WaitForStream, a nullptr `from` (legacy default stream) IS bridged so
// that allocator reuse orders against legacy-stream work.
//
// No-op when `from == to`.
void bridgeStreams(cudaStream_t from, cudaStream_t to);

}  // namespace cuda
}  // namespace zt

#endif  // BUILD_CUDA_MODULE
