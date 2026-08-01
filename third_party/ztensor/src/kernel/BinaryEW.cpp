// ztensor/kernel/BinaryEW.cpp
//
// Device dispatch for BinaryEW. Phase 3 wires only the CPU path; the CUDA
// branch is added in phase 4 behind BUILD_CUDA_MODULE.

#include "kernel/BinaryEW.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              const Tensor& dst,
              BinaryEWOpCode op) {
    if (lhs.is_cpu() && rhs.is_cpu() && dst.is_cpu()) {
        BinaryEWCPU(lhs, rhs, dst, op);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (lhs.is_cuda() && rhs.is_cuda() && dst.is_cuda()) {
        BinaryEWCUDA(lhs, rhs, dst, op);
        return;
    }
#endif
    ZT_LOG_ERROR("BinaryEW: unsupported device (lhs {}, rhs {}, dst {})",
                 lhs.device().string(),
                 rhs.device().string(),
                 dst.device().string());
}

}  // namespace zt::kernel
