// ztensor/core/CUDAUtils.h
//
// Cross-execution-space annotations and CUDA host-side helpers. Modeled on
// Open3D's open3d/core/CUDAUtils.h.
//
// This header is consumed in two compile contexts:
//
//   * Plain C++ TUs (every .cpp, plus the headers they pull in). Only the
//     ZT_HOST_DEVICE annotation is visible here. None of the CUDA host-side
//     helpers (CUDAScopedDevice, cuda::GetStream, ZT_CUDA_CHECK, ...) are
//     declared, so the CPU-only build — and every non-kernel TU in the CUDA
//     build — stays free of any CUDA header dependency.
//
//   * CUDA-enabled TUs (compiled only under BUILD_CUDA_MODULE). BUILD_CUDA_MODULE
//     is defined, so the full CUDA surface is exposed: the runtime header, the
//     ZT_CUDA_CHECK / ZT_CUDA_GET_LAST_ERROR macros, the CUDAScopedDevice RAII
//     guard, and the cuda:: helpers (DeviceCount, Synchronize, GetStream, ...).
//     The out-of-line definitions live in CUDAUtils.cpp.
//
// Rule of thumb: CUDA host helpers are #ifdef'd on BUILD_CUDA_MODULE. They
// are never visible in a CPU-only build; in a CUDA build the host compiler
// sees them too (with -I/usr/local/cuda/include). The __CUDACC__ guard is
// reserved for code containing actual kernel syntax (__global__, <<<>>>).

#pragma once

// Mark a function/lambda callable from both host and device. On the CPU build
// this expands to nothing; under NVCC it becomes __host__ __device__. Kernel
// code (Indexer methods, element functors) is written against this so the same
// source compiles for both backends.
#ifdef __CUDACC__
#define ZT_HOST_DEVICE __host__ __device__
#else
#define ZT_HOST_DEVICE
#endif

#ifdef BUILD_CUDA_MODULE
// ──────────────────────────────────────────────────────────────────────────
// CUDA host-side helpers (visible under BUILD_CUDA_MODULE).
// ──────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>

#include <cstddef>

#include "core/cuda/CUDAStreamContext.h"

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

namespace zt {

// RAII current-device switcher. Construct with a Device (or device id); the
// CUDA current device is set for the lifetime of the guard and restored on
// destruction. Copy/move disabled (matches Open3D's CUDAScopedDevice).
//
// Implemented out-of-line in CUDAUtils.cpp.
class CUDAScopedDevice {
public:
    explicit CUDAScopedDevice(int device_id);
    explicit CUDAScopedDevice(const class Device& device);
    ~CUDAScopedDevice();
    CUDAScopedDevice(const CUDAScopedDevice&) = delete;
    CUDAScopedDevice& operator=(const CUDAScopedDevice&) = delete;

private:
    int prev_device_id_ = 0;
};

namespace cuda {

// [[noreturn]] helper used by the ZT_CUDA_CHECK / ZT_CUDA_GET_LAST_ERROR
// macros. Defined in CUDAUtils.cpp so the macros expand to a plain call.
[[noreturn]] void __ThrowCudaError(cudaError_t err,
                                   const char* expr,
                                   const char* file,
                                   int line);

// Number of CUDA-capable devices. Returns 0 if there is none or the runtime
// could not be initialized (never throws).
int DeviceCount();

// True iff DeviceCount() > 0.
bool IsAvailable();

// Synchronize every device (DeviceCount() == 0 -> no-op).
void Synchronize();

// Range-check `device_id` against DeviceCount() and throw std::runtime_error
// if it is out of range.
void AssertCUDADeviceAvailable(int device_id);

// The CUDA current device (cudaGetDevice). Defined in CUDAUtils.cpp.
int GetDevice();

// The legacy default stream (always 0). Retained for code that must
// explicitly pin to stream 0 regardless of the thread-local stream.
inline cudaStream_t GetDefaultStream() { return static_cast<cudaStream_t>(0); }

// The active stream for the calling thread. Returns the thread-local stream
// set via SetCurrentStream() / CUDAStreamGuard, defaulting to the legacy
// default stream (0). Inline so existing call sites need no changes — they
// automatically benefit from per-thread stream support.
inline cudaStream_t GetStream() { return GetCurrentStream(); }

}  // namespace cuda

}  // namespace zt

#endif  // BUILD_CUDA_MODULE
