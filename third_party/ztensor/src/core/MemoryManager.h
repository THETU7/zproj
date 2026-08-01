// ztensor/core/MemoryManager.h
//
// Device-dispatched memory operations. Static facade that routes to a
// per-device implementation (CPU now; CUDA in phase 4). Modeled on Open3D's
// open3d::core::MemoryManager.
//
// This is an internal header (lives under src/); the public API only needs
// Blob, which wraps these calls.

#pragma once

#include <cstddef>

#include "ztensor/zt/Device.h"

namespace zt {

class MemoryManager {
public:
    // Allocate `byte_size` bytes on `device`. Returns nullptr when byte_size
    // is zero. Throws std::runtime_error on failure or unsupported device.
    static void* Malloc(std::size_t byte_size, const Device& device);

    // Release memory previously returned by Malloc (no-op for nullptr).
    static void Free(void* ptr, const Device& device);

    // Copy `byte_size` bytes between arbitrary devices.
    static void Memcpy(const Device& dst_device,
                       void* dst,
                       const Device& src_device,
                       const void* src,
                       std::size_t byte_size);

    // Convenience wrappers for the most common host <-> device patterns.
    static void MemcpyFromHost(void* dst,
                               const Device& dst_device,
                               const void* src,
                               std::size_t byte_size) {
        Memcpy(dst_device, dst, Device(kCPU), src, byte_size);
    }
    static void MemcpyToHost(void* dst,
                             const void* src,
                             const Device& src_device,
                             std::size_t byte_size) {
        Memcpy(Device(kCPU), dst, src_device, src, byte_size);
    }
};

}  // namespace zt
