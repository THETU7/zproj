// ztensor/core/CUDAUtils.cu
//
// Out-of-line definitions of the CUDA host-side helpers declared in the
// public zt/cuda/Exception.h and zt/cuda/Guard.h. Compiled only under
// BUILD_CUDA_MODULE. Modeled on Open3D's CUDAUtils.cpp.

#include <stdexcept>
#include <string>

#include "ztensor/zt/cuda/Exception.h"
#include "ztensor/zt/cuda/Guard.h"
#include "ztensor/zt/Device.h"
#include "ztensor/zt/utility/Log.h"

namespace zt {
namespace cuda {

[[noreturn]] void __ThrowCudaError(cudaError_t err,
                                   const char* expr,
                                   const char* file,
                                   int line) {
    const char* msg = cudaGetErrorString(err);
    const char* name = cudaGetErrorName(err);
    // ZT_LOG_ERROR is [[noreturn]] (throws std::runtime_error).
    (void)file;
    (void)line;
    ZT_LOG_ERROR("CUDA error '{}' ({}): {}", name, msg, expr);
}

int DeviceCount() {
    int count = 0;
    // cudaGetDeviceCount can fail before any device is initialized; treat
    // every failure as "no device" rather than throwing (matches Open3D).
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

bool IsAvailable() { return DeviceCount() > 0; }

void Synchronize() {
    const int count = DeviceCount();
    for (int i = 0; i < count; ++i) {
        CUDAScopedDevice scoped(i);
        ZT_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void AssertCUDADeviceAvailable(int device_id) {
    const int count = DeviceCount();
    if (count <= 0) {
        ZT_LOG_ERROR("CUDA device requested but no CUDA-capable device found");
    }
    // index == -1 means "the current default device"; resolve it before the
    // range check so the caller can pass a Device that did not pin an index.
    int id = device_id;
    if (id < 0) {
        id = GetDevice();
    }
    if (id < 0 || id >= count) {
        ZT_LOG_ERROR(
            "CUDA device id {} out of range (device count {})", id, count);
    }
}

int GetDevice() {
    int dev = 0;
    ZT_CUDA_CHECK(cudaGetDevice(&dev));
    return dev;
}

}  // namespace cuda

// ── CUDAScopedDevice ─────────────────────────────────────────────────────────

CUDAScopedDevice::CUDAScopedDevice(int device_id) {
    cuda::AssertCUDADeviceAvailable(device_id);
    prev_device_id_ = cuda::GetDevice();
    // Resolve the "current device" sentinel (-1) to a concrete id; otherwise
    // cudaSetDevice(-1) fails.
    const int id = (device_id < 0) ? prev_device_id_ : device_id;
    if (id != prev_device_id_) {
        ZT_CUDA_CHECK(cudaSetDevice(id));
    }
}

CUDAScopedDevice::CUDAScopedDevice(const Device& device) {
    int id = device.index();
    cuda::AssertCUDADeviceAvailable(id);  // resolves -1 internally
    prev_device_id_ = cuda::GetDevice();
    if (id < 0) id = prev_device_id_;
    if (id != prev_device_id_) {
        ZT_CUDA_CHECK(cudaSetDevice(id));
    }
}

CUDAScopedDevice::~CUDAScopedDevice() {
    // Best-effort restore: ignore errors (e.g. during stack unwinding).
    (void)cudaSetDevice(prev_device_id_);
}

}  // namespace zt
