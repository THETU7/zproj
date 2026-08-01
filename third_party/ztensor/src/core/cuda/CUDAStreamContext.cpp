// ztensor/core/CUDAStreamContext.cu
//
// Out-of-line definitions for the per-thread CUDA stream helpers declared in
// CUDAStreamContext.h.  Compiled only under BUILD_CUDA_MODULE (NVCC).

#include "core/cuda/CUDAEventPool.h"
#include "ztensor/zt/cuda/Stream.h"

namespace zt {
namespace cuda {

// ── Thread-local stream storage ──────────────────────────────────────────────
//
// Initialised to nullptr (== legacy default stream 0), so every thread starts
// with the default stream until explicitly changed.

static thread_local cudaStream_t tl_current_stream = nullptr;

cudaStream_t GetCurrentStream() { return tl_current_stream; }

void SetCurrentStream(cudaStream_t stream) { tl_current_stream = stream; }

// ── Stream synchronisation ───────────────────────────────────────────────────

void WaitForStream(cudaStream_t execution_stream,
                   cudaStream_t dependency_stream) {
    if (dependency_stream == nullptr || dependency_stream == execution_stream) {
        return;
    }

    // Delegate to bridgeStreams which uses a pooled event edge
    // (cudaEventRecord + cudaStreamWaitEvent) with a host-sync fallback
    // on failure (see CUDAEventPool.cpp).
    bridgeStreams(dependency_stream, execution_stream);
}

// ── CUDAStreamGuard ──────────────────────────────────────────────────────────

CUDAStreamGuard::CUDAStreamGuard(cudaStream_t stream)
    : prev_stream_(GetCurrentStream()) {
    SetCurrentStream(stream);
}

CUDAStreamGuard::~CUDAStreamGuard() { SetCurrentStream(prev_stream_); }

}  // namespace cuda
}  // namespace zt
