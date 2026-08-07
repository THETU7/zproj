// ztensor/zt/cuda/Stream.h
//
// Per-thread CUDA stream state, RAII guard, and the active-stream accessors
// used when launching kernels. Public counterpart of the internal stream
// helpers, modeled on c10/cuda/CUDAStream.h.
//
// Visible only under BUILD_CUDA_MODULE. The thread-local stream defaults to
// nullptr (== legacy default stream 0), so by default kernels launch on
// stream 0 with zero behaviour change. Call SetCurrentStream() or use
// CUDAStreamGuard to route work to a non-default stream. Out-of-line
// definitions live in core/cuda/CUDAStreamContext.cpp.

#pragma once

#ifdef BUILD_CUDA_MODULE

#include "ztensor/zt/cuda/Vendor.h"

namespace zt {
namespace cuda {

// ── Per-thread current stream ────────────────────────────────────────────────
//
// The thread-local stream defaults to nullptr (== legacy default stream 0),
// so existing code sees zero behaviour change by default. Call
// SetCurrentStream() or use CUDAStreamGuard to route work to a non-default
// stream.

// Return the active CUDA stream for the calling thread (thread_local).
cudaStream_t GetCurrentStream();

// Set the active CUDA stream for the calling thread.
void SetCurrentStream(cudaStream_t stream);

// ── Stream synchronisation ───────────────────────────────────────────────────

// Make `execution_stream` wait (GPU-side) for all work enqueued on
// `dependency_stream`. No-op when the two streams are identical or
// dependency_stream is nullptr.
//
// Uses a pooled event edge (cudaEventRecord + cudaStreamWaitEvent) via
// bridgeStreams with a host-sync fallback on failure.
void WaitForStream(cudaStream_t execution_stream,
                   cudaStream_t dependency_stream);

// ── Stream accessors ─────────────────────────────────────────────────────────

// The legacy default stream (always 0). Retained for code that must
// explicitly pin to stream 0 regardless of the thread-local stream.
inline cudaStream_t GetDefaultStream() { return static_cast<cudaStream_t>(0); }

// The active stream for the calling thread. Returns the thread-local stream
// set via SetCurrentStream() / CUDAStreamGuard, defaulting to the legacy
// default stream (0). Pass this as the stream argument when launching kernels
// so they honour a caller-selected stream.
inline cudaStream_t GetStream() { return GetCurrentStream(); }

// ── RAII stream guard ────────────────────────────────────────────────────────

// Saves the current thread-local stream on construction, sets a new stream,
// and restores the previous stream on destruction. Deleted copy/move so the
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
