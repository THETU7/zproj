// ztensor/kernel/CheckIndexBounds.cpp
//
// Device dispatch for the scatter/gather index bounds check.

#include "kernel/CheckIndexBounds.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void CheckIndexBounds(const Tensor& index, int64_t dim_size, const char* op) {
    if (index.numel() == 0) {
        return;  // nothing to validate; the kernels are no-ops over 0 elements
    }
    if (index.is_cpu()) {
        CheckIndexBoundsCPU(index, dim_size, op);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (index.is_cuda()) {
        CheckIndexBoundsCUDA(index, dim_size, op);
        return;
    }
#endif
    ZT_LOG_ERROR("CheckIndexBounds: unsupported device {}",
                 index.device().string());
}

}  // namespace zt::kernel
