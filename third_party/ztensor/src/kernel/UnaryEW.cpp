// ztensor/kernel/UnaryEW.cpp
//
// Device dispatch for UnaryEW. Phase 3 wires only the CPU path; the CUDA
// branch is added in phase 4 behind BUILD_CUDA_MODULE.

#include "kernel/UnaryEW.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void UnaryEW(const Tensor& src, const Tensor& dst, UnaryEWOpCode op) {
    if (src.is_cpu() && dst.is_cpu()) {
        UnaryEWCPU(src, dst, op);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && dst.is_cuda()) {
        UnaryEWCUDA(src, dst, op);
        return;
    }
#endif
    ZT_LOG_ERROR("UnaryEW: unsupported device (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

}  // namespace zt::kernel
