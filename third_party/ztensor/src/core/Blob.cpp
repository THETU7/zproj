// ztensor/core/Blob.cpp

#include "ztensor/zt/Blob.h"

#include "core/MemoryManager.h"

namespace zt {

Blob::Blob(std::size_t byte_size, const Device& device)
    : device_(device), owns_memory_(true) {
    data_ptr_ = MemoryManager::Malloc(byte_size, device);
    // No deleter: MemoryManager::Free is used on destruction.
}

Blob::Blob(const Device& device,
           void* data_ptr,
           std::function<void(void*)> deleter)
    : data_ptr_(data_ptr),
      device_(device),
      deleter_(std::move(deleter)),
      owns_memory_(false) {}

Blob::~Blob() {
    if (deleter_) {
        deleter_(data_ptr_);
    } else if (owns_memory_ && data_ptr_ != nullptr) {
        // Only the internal-allocation ctor owns the memory; external memory
        // passed with a null/empty deleter (from_blob with no cleanup) must
        // NOT be freed here (AGENTS.md invariant #3, "external memory (e.g.
        // from_blob with no deleter); not freed by the Blob").
        MemoryManager::Free(data_ptr_, device_);
    }
}

}  // namespace zt
