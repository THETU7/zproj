// ztensor/core/CUDAEventPool.cu
//
// Out-of-line definitions for CudaEventPool and bridgeStreams.  Compiled only
// under BUILD_CUDA_MODULE (NVCC).

#include "ztensor/zt/utility/Log.h"

#include "core/cuda/CUDAEventPool.h"

namespace zt {
namespace cuda {

// ── Meyers singleton ─────────────────────────────────────────────────────────

CudaEventPool& CudaEventPool::instance() {
    static CudaEventPool pool;
    return pool;
}

// ── acquire / release ────────────────────────────────────────────────────────

cudaEvent_t CudaEventPool::acquire() {
    if (!shutdown_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            cudaEvent_t event = pool_.back();
            pool_.pop_back();
            stats_.reused.fetch_add(1, std::memory_order_relaxed);
            return event;
        }
    }

    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) !=
        cudaSuccess) {
        return nullptr;
    }
    stats_.created.fetch_add(1, std::memory_order_relaxed);
    return event;
}

void CudaEventPool::release(cudaEvent_t event) {
    if (!event) return;
    if (!shutdown_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.size() < MAX_POOL_SIZE) {
            pool_.push_back(event);
            return;
        }
    }
    cudaEventDestroy(event);
}

// ── shutdown ─────────────────────────────────────────────────────────────────

void CudaEventPool::shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (cudaEvent_t event : pool_) {
        cudaEventDestroy(event);
    }
    pool_.clear();
}

CudaEventPool::~CudaEventPool() { shutdown(); }

// ── bridgeStreams ────────────────────────────────────────────────────────────

void bridgeStreams(cudaStream_t from, cudaStream_t to) {
    if (from == to) {
        return;
    }

    // Attempt the GPU-side event edge.
    if (cudaEvent_t edge = CudaEventPool::instance().acquire()) {
        const bool bridged = cudaEventRecord(edge, from) == cudaSuccess &&
                             cudaStreamWaitEvent(to, edge, 0) == cudaSuccess;
        CudaEventPool::instance().release(edge);
        if (bridged) {
            return;
        }
    }

    // Fallback: host-sync (correct but serialises the host).
    const cudaError_t sync_status = cudaStreamSynchronize(from);
    if (sync_status != cudaSuccess) {
        ZT_LOG_WARNING(
            "bridgeStreams: event edge and fallback sync both failed ({}); "
            "stream may have been destroyed with pending work",
            cudaGetErrorString(sync_status));
    }
}

}  // namespace cuda
}  // namespace zt
