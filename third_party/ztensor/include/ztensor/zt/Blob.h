// ztensor/zt/Blob.h
//
// Blob: the sole owner of a tensor's backing memory. A Tensor is a "view" of
// a Blob; slicing/transposing/viewing shares the same Blob via shared_ptr, so
// those operations are zero-copy.
//
// Modeled on Open3D's open3d::core::Blob. Two construction modes:
//   1) (byte_size, device)       -> allocate internally via MemoryManager.
//   2) (device, data_ptr, deleter) -> wrap external memory (from_blob, DLPack,
//                                     moved std::vector, ...); the deleter is
//                                     invoked on destruction.
//
// Intentionally does NOT store its own byte_size (Open3D parity): negative-
// stride views can refer to addresses before the base, so callers track size.

#pragma once

#include <cstddef>
#include <functional>
#include <utility>

#include "ztensor/zt/Device.h"

namespace zt {

class Blob {
public:
    // Allocate `byte_size` bytes on `device` via the MemoryManager.
    Blob(std::size_t byte_size, const Device& device);

    // Wrap external memory located at `data_ptr` on `device`. When the Blob is
    // destroyed, `deleter(data_ptr)` is invoked (if non-null). The Blob does
    // NOT take ownership semantics beyond invoking the deleter.
    Blob(const Device& device,
         void* data_ptr,
         std::function<void(void*)> deleter = nullptr);

    ~Blob();

    // Non-copyable: ownership is shared via shared_ptr<Blob>.
    Blob(const Blob&) = delete;
    Blob& operator=(const Blob&) = delete;
    Blob(Blob&&) noexcept = delete;
    Blob& operator=(Blob&&) noexcept = delete;

    void* GetDataPtr() const noexcept { return data_ptr_; }
    const Device& GetDevice() const noexcept { return device_; }

private:
    void* data_ptr_ = nullptr;
    Device device_;
    // Invoked with data_ptr_ on destruction when non-null (external memory).
    // For internally-allocated memory this stays null and MemoryManager::Free
    // is used instead.
    std::function<void(void*)> deleter_;
    // True only for the (byte_size, device) internal-allocation ctor, which
    // owns its memory and must call MemoryManager::Free. External-memory blobs
    // (the (device, data_ptr, deleter) ctor) never free via the manager even
    // when the deleter is null/empty — a null deleter on external memory means
    // "from_blob with no cleanup, do not free" (AGENTS.md invariant #3).
    bool owns_memory_ = false;
};

}  // namespace zt
