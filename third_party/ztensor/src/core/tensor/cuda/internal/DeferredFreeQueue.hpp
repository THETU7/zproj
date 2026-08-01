// ztensor/core/tensor/internal/DeferredFreeQueue.hpp
//
// Stream-ordered deferred free queue for periodic driver-memory reclamation.
// Uses pooled CUDA events to track pending frees without host sync; callers
// periodically invoke process() which polls cudaEventQuery and releases blocks
// whose GPU work has completed.
//
// This is NOT the primary reuse path — slab and bucketed pools reuse immediately
// via stream-tag + bridgeStreams (no polling).  DeferredFreeQueue is only for
// cleaning up cudaMallocAsync'd blocks that were evicted from the cache or
// bypassed the pool, so the driver memory pool can reclaim them.
//
// Modeled on LichtFeld-Studio's DeferredFreeQueue.
//
// This header is visible only when BUILD_CUDA_MODULE is enabled
// (same guard discipline as CUDAUtils.h / CUDAEventPool.h).

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda_runtime.h>

#include "core/cuda/CUDAEventPool.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace zt {

// ── DeferredFreeQueue ──────────────────────────────────────────────────────────

class DeferredFreeQueue {
public:
    static constexpr size_t INITIAL_CAPACITY = 1024;
    static constexpr size_t PROCESS_BATCH_SIZE = 64;

    // Callback invoked when a deferred block is ready to be freed.
    // ptr  — the memory block (allocated via cudaMallocAsync or cudaMalloc).
    // size — size in bytes (for stats; the callback itself decides how to free).
    using FreeCallback = void (*)(void* ptr, size_t size);

    static DeferredFreeQueue& instance();

    // Enqueue a block for deferred freeing.  Records an event on `stream` so
    // process() won't release the block until the stream has passed that point.
    // Falls back to a synchronous free if event acquisition fails or the queue
    // has been shut down.
    void defer_free(void* ptr,
                    size_t size,
                    cudaStream_t stream,
                    FreeCallback callback);

    // Poll for completed frees (non-blocking).  Calls cudaEventQuery on pending
    // entries up to `max_items` and invokes their callbacks for any that have
    // completed.  Returns the number of blocks freed this call.
    size_t process(size_t max_items = PROCESS_BATCH_SIZE);

    // Drain everything synchronously (calls cudaDeviceSynchronize first, then
    // invokes every pending callback).  Used during shutdown.
    void flush();

    // CAS-guarded; subsequent calls are no-ops.
    void shutdown();

    // ── Statistics ─────────────────────────────────────────────────────────────

    struct Stats {
        std::atomic<uint64_t> queued_count{0};
        std::atomic<uint64_t> freed_count{0};
        std::atomic<uint64_t> queued_bytes{0};
        std::atomic<uint64_t> freed_bytes{0};
    };

    const Stats& stats() const { return stats_; }

    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return pending_.size();
    }

    DeferredFreeQueue(const DeferredFreeQueue&) = delete;
    DeferredFreeQueue& operator=(const DeferredFreeQueue&) = delete;

private:
    DeferredFreeQueue() { pending_.reserve(INITIAL_CAPACITY); }
    ~DeferredFreeQueue() { shutdown(); }

    struct PendingFree {
        void* ptr = nullptr;
        size_t size = 0;
        cudaEvent_t event = nullptr;
        FreeCallback callback = nullptr;
    };

    std::vector<PendingFree> pending_;
    mutable std::mutex queue_mutex_;
    std::atomic<bool> shutdown_{false};
    Stats stats_;
};

// ── Inline definitions ─────────────────────────────────────────────────────────

inline DeferredFreeQueue& DeferredFreeQueue::instance() {
    static DeferredFreeQueue queue;
    return queue;
}

inline void DeferredFreeQueue::defer_free(void* ptr,
                                          size_t size,
                                          cudaStream_t stream,
                                          FreeCallback callback) {
    if (!ptr) return;
    if (shutdown_.load(std::memory_order_acquire)) {
        callback(ptr, size);
        return;
    }

    cudaEvent_t event = ::zt::cuda::CudaEventPool::instance().acquire();
    if (!event) {
        // Cannot get an event — synchronise the host and free immediately.
        cudaStreamSynchronize(stream);
        callback(ptr, size);
        return;
    }

    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        ::zt::cuda::CudaEventPool::instance().release(event);
        cudaStreamSynchronize(stream);
        callback(ptr, size);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_.push_back({ptr, size, event, callback});
        stats_.queued_count.fetch_add(1, std::memory_order_relaxed);
        stats_.queued_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

inline size_t DeferredFreeQueue::process(size_t max_items) {
    std::vector<PendingFree> to_free;
    to_free.reserve(max_items > 0 ? max_items : PROCESS_BATCH_SIZE);

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (pending_.empty()) return 0;

        size_t count = 0;
        size_t i = 0;
        while (i < pending_.size() && (max_items == 0 || count < max_items)) {
            const auto& item = pending_[i];
            const cudaError_t err = cudaEventQuery(item.event);
            if (err == cudaSuccess) {
                to_free.push_back(item);
                ::zt::cuda::CudaEventPool::instance().release(item.event);
                pending_[i] = pending_.back();
                pending_.pop_back();
                ++count;
            } else if (err == cudaErrorNotReady) {
                ++i;
            } else {
                // Unexpected error — skip this entry to avoid stalling.
                ++i;
            }
        }
    }

    for (const auto& item : to_free) {
        item.callback(item.ptr, item.size);
        stats_.freed_count.fetch_add(1, std::memory_order_relaxed);
        stats_.freed_bytes.fetch_add(item.size, std::memory_order_relaxed);
        stats_.queued_bytes.fetch_sub(item.size, std::memory_order_relaxed);
    }

    return to_free.size();
}

inline void DeferredFreeQueue::flush() {
    std::vector<PendingFree> to_free;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        to_free = std::move(pending_);
        pending_.clear();
    }

    cudaDeviceSynchronize();

    for (const auto& item : to_free) {
        ::zt::cuda::CudaEventPool::instance().release(item.event);
        item.callback(item.ptr, item.size);
        stats_.freed_count.fetch_add(1, std::memory_order_relaxed);
        stats_.freed_bytes.fetch_add(item.size, std::memory_order_relaxed);
    }

    stats_.queued_bytes.store(0, std::memory_order_relaxed);
}

inline void DeferredFreeQueue::shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) return;
    flush();
}

}  // namespace zt

#endif  // BUILD_CUDA_MODULE
