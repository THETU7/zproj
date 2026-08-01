// ztensor/zt/cuda/Exception.h
//
// CUDA error-checking macros. Public counterpart of the internal CUDA error
// helpers, modeled on c10/cuda/Exception.h.
//
// The whole header is visible only under BUILD_CUDA_MODULE (it expands to
// nothing otherwise), so CPU-only TUs that pull it in transitively stay free
// of any CUDA header dependency. The [[noreturn]] helper is defined
// out-of-line in core/cuda/CUDAUtils.cpp so the macros expand to a plain call.

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda_runtime.h>

namespace zt {
namespace cuda {

// Translate a cudaError_t into a std::runtime_error with source provenance.
// Defined in CUDAUtils.cpp; used by ZT_CUDA_CHECK / ZT_CUDA_GET_LAST_ERROR.
[[noreturn]] void __ThrowCudaError(cudaError_t err,
                                   const char* expr,
                                   const char* file,
                                   int line);

}  // namespace cuda
}  // namespace zt

// Translate a returned cudaError_t into a std::runtime_error with source
// provenance. Use for APIs that return cudaError_t.
#define ZT_CUDA_CHECK(call)                                                   \
    do {                                                                      \
        const cudaError_t _zt_err = (call);                                   \
        if (_zt_err != cudaSuccess) {                                         \
            ::zt::cuda::__ThrowCudaError(_zt_err, #call, __FILE__, __LINE__); \
        }                                                                     \
    } while (0)

// Check the asynchronous "last error" (kernel-launch failures). Call this
// immediately after a <<<>>> launch.
#define ZT_CUDA_GET_LAST_ERROR(context)                 \
    do {                                                \
        const cudaError_t _zt_err = cudaGetLastError(); \
        if (_zt_err != cudaSuccess) {                   \
            ::zt::cuda::__ThrowCudaError(               \
                _zt_err, context, __FILE__, __LINE__);  \
        }                                               \
    } while (0)

#endif  // BUILD_CUDA_MODULE
