// ztensor/zt/cuda/Guard.h
//
// CUDA device management: the RAII current-device switcher and the device
// query helpers. Public counterpart of the device portion of the internal
// CUDAUtils.h, modeled on c10/cuda/CUDAGuard.h.
//
// Visible only under BUILD_CUDA_MODULE. Out-of-line definitions live in
// core/cuda/CUDAUtils.cpp.

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda_runtime.h>

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

}  // namespace cuda

}  // namespace zt

#endif  // BUILD_CUDA_MODULE
