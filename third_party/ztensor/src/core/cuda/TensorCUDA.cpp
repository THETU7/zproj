// ztensor/core/TensorCUDA.cu
//
// CUDA-specific Tensor methods (§8.5.E E-4).  Compiled only under
// BUILD_CUDA_MODULE (NVCC).  Provides the static memory-pool diagnostics /
// lifecycle helpers (PrintMemoryPoolStats, TrimMemoryPool, ShutdownMemoryPool)
// and the per-tensor stream operations (record_stream, set_stream).
//
// These are thin delegates to CudaMemoryPool; the heavy logic lives in the
// four-tier allocator (§8.5.E E-3).

#include "ztensor/zt/Tensor.h"

#include "core/cuda/CudaMemoryPool.h"
#include "ztensor/zt/cuda/Stream.h"  // GetCurrentStream

namespace zt {

// ── Static memory-pool helpers ─────────────────────────────────────────────────

#ifdef BUILD_CUDA_MODULE

/*static*/ void Tensor::PrintMemoryPoolStats() {
    CudaMemoryPool::instance().print_stats();
}

/*static*/ void Tensor::TrimMemoryPool() {
    // Full trim: sync device, flush deferred frees, retag all streams,
    // trim bucketed cache, and trim the driver pool to zero.
    CudaMemoryPool::instance().trim_cached_memory();
}

/*static*/ void Tensor::ShutdownMemoryPool() {
    CudaMemoryPool::instance().shutdown();
}

#endif  // BUILD_CUDA_MODULE

// ── Per-tensor stream operations ───────────────────────────────────────────────

#ifdef __CUDACC__

void Tensor::record_stream(cudaStream_t stream) {
    if (!defined()) return;
    // If no stream is given, default to the current thread-local stream.
    cudaStream_t s = stream ? stream : cuda::GetCurrentStream();
    CudaMemoryPool::instance().record_stream(data_ptr_, s);
}

void Tensor::set_stream(cudaStream_t stream) {
    if (!defined()) return;
    cudaStream_t s = stream ? stream : cuda::GetCurrentStream();
    CudaMemoryPool::instance().rehome_stream(data_ptr_, s);
}

#endif  // __CUDACC__

}  // namespace zt
