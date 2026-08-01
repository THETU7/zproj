// ztensor/core/MemoryManager.cpp
//
// Static facade implementation. Dispatches on the Device's type to the
// appropriate backend (CPU now; CUDA via MemoryManagerCUDA in phase 4).

#include "core/MemoryManager.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ztensor/zt/utility/Log.h"

namespace zt {

// ---- CPU backend (defined here for phase 2; may be split out later) -------
namespace {

void* CPU_Malloc(std::size_t byte_size) {
    if (byte_size == 0) return nullptr;
    void* ptr = std::malloc(byte_size);
    if (ptr == nullptr) {
        throw std::runtime_error("zt::MemoryManager(CPU): out of memory");
    }
    return ptr;
}

void CPU_Free(void* ptr) { std::free(ptr); }

void CPU_Memcpy(void* dst, const void* src, std::size_t n) {
    if (n == 0) return;
    std::memcpy(dst, src, n);
}

}  // namespace

#ifdef BUILD_CUDA_MODULE
// Forward-declared in MemoryManagerCUDA.cpp.
void* CUDA_Malloc(std::size_t byte_size, int device_index);
void CUDA_Free(void* ptr, int device_index);
void CUDA_MemcpyH2D(void* dst,
                    const void* src,
                    std::size_t n,
                    int device_index);
void CUDA_MemcpyD2H(void* dst,
                    const void* src,
                    std::size_t n,
                    int device_index);
void CUDA_MemcpyD2D(void* dst,
                    const void* src,
                    std::size_t n,
                    int device_index);
#endif

// ---- facade dispatch -------------------------------------------------------

void* MemoryManager::Malloc(std::size_t byte_size, const Device& device) {
    if (byte_size == 0) return nullptr;
    if (device.is_cpu()) {
        return CPU_Malloc(byte_size);
    }
#ifdef BUILD_CUDA_MODULE
    if (device.is_cuda()) {
        return CUDA_Malloc(byte_size, device.index());
    }
#endif
    ZT_LOG_ERROR("MemoryManager::Malloc: unsupported device '{}'",
                 device.string());
}

void MemoryManager::Free(void* ptr, const Device& device) {
    if (ptr == nullptr) return;
    if (device.is_cpu()) {
        CPU_Free(ptr);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (device.is_cuda()) {
        CUDA_Free(ptr, device.index());
        return;
    }
#endif
    ZT_LOG_ERROR("MemoryManager::Free: unsupported device '{}'",
                 device.string());
}

void MemoryManager::Memcpy(const Device& dst_device,
                           void* dst,
                           const Device& src_device,
                           const void* src,
                           std::size_t byte_size) {
    if (byte_size == 0) return;

    const bool d_cpu = dst_device.is_cpu();
    const bool d_cuda = dst_device.is_cuda();
    const bool s_cpu = src_device.is_cpu();
    const bool s_cuda = src_device.is_cuda();

    if (d_cpu && s_cpu) {
        CPU_Memcpy(dst, src, byte_size);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (d_cuda && s_cpu) {
        CUDA_MemcpyH2D(dst, src, byte_size, dst_device.index());
        return;
    }
    if (d_cpu && s_cuda) {
        CUDA_MemcpyD2H(dst, src, byte_size, src_device.index());
        return;
    }
    if (d_cuda && s_cuda) {
        CUDA_MemcpyD2D(dst, src, byte_size, dst_device.index());
        return;
    }
#endif
    (void)d_cuda;
    (void)s_cuda;
    ZT_LOG_ERROR(
        "MemoryManager::Memcpy: unsupported transfer '{}'(dst) <- '{}'(src)",
        dst_device.string(),
        src_device.string());
}

}  // namespace zt
