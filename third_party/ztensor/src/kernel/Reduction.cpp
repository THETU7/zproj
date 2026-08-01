// ztensor/kernel/Reduction.cpp
//
// Device dispatch for Reduction. Phase 3 wires only the CPU path; the CUDA
// branch is added in phase 4 behind BUILD_CUDA_MODULE.

#include "kernel/Reduction.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void Reduction(const Tensor& src,
               const Tensor& dst,
               IntArrayRef reduction_dims,
               ReductionOpCode op) {
    if (src.is_cpu() && dst.is_cpu()) {
        ReductionCPU(src, dst, reduction_dims, op);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && dst.is_cuda()) {
        ReductionCUDA(src, dst, reduction_dims, op);
        return;
    }
#endif
    ZT_LOG_ERROR("Reduction: unsupported device (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

}  // namespace zt::kernel
