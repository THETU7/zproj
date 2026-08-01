// ztensor/core/CUDAStreamContext.h
//
// Per-thread CUDA stream state and RAII guard. Modeled on LichtFeld-Studio's
// cuda_stream_context.hpp and PyTorch's CUDAStreamGuard.
//
// This header is visible only when BUILD_CUDA_MODULE is enabled;
// .cpp TUs never see cudaStream_t or any of the declarations below (same guard
// discipline as CUDAUtils.h).

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda_runtime.h>

namespace zt {
namespace cuda {

// ── Per-thread current stream ────────────────────────────────────────────────
//
// The thread-local stream defaults to nullptr (== legacy default stream 0),
// so existing code sees zero behaviour change by default.  Call
// SetCurrentStream() or use CUDAStreamGuard to route work to a non-default
// stream.

// Return the active CUDA stream for the calling thread (thread_local).
cudaStream_t GetCurrentStream();

// Set the active CUDA stream for the calling thread.
void SetCurrentStream(cudaStream_t stream);

// ── Stream synchronisation ───────────────────────────────────────────────────

// Make `execution_stream` wait (GPU-side) for all work enqueued on
// `dependency_stream`.  No-op when the two streams are identical or
// dependency_stream is nullptr.
//
// Uses a pooled event edge (cudaEventRecord + cudaStreamWaitEvent) via
// bridgeStreams with a host-sync fallback on failure.
void WaitForStream(cudaStream_t execution_stream,
                   cudaStream_t dependency_stream);

// ── RAII stream guard ────────────────────────────────────────────────────────

// Saves the current thread-local stream on construction, sets a new stream,
// and restores the previous stream on destruction.  Deleted copy/move so the
// guard cannot be accidentally shared.
//
// Usage:
//   {
//       CUDAStreamGuard g(my_stream);
//       // ... all work here uses my_stream ...
//   }  // previous stream restored
class CUDAStreamGuard {
public:
    explicit CUDAStreamGuard(cudaStream_t stream);
    ~CUDAStreamGuard();

    CUDAStreamGuard(const CUDAStreamGuard&) = delete;
    CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
    CUDAStreamGuard(CUDAStreamGuard&&) = delete;
    CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

private:
    cudaStream_t prev_stream_;
};

}  // namespace cuda
}  // namespace zt

#endif  // BUILD_CUDA_MODULE
