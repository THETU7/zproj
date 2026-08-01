// ztensor/kernel/TernaryEW.cpp
//
// Device dispatch for TernaryEW.

#include "kernel/TernaryEW.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void TernaryEW(const Tensor& cond,
               const Tensor& a,
               const Tensor& b,
               const Tensor& dst,
               TernaryEWOpCode op) {
    if (cond.is_cpu() && a.is_cpu() && b.is_cpu() && dst.is_cpu()) {
        TernaryEWCPU(cond, a, b, dst, op);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (cond.is_cuda() && a.is_cuda() && b.is_cuda() && dst.is_cuda()) {
        TernaryEWCUDA(cond, a, b, dst, op);
        return;
    }
#endif
    ZT_LOG_ERROR(
        "TernaryEW: unsupported device (cond {}, a {}, b {}, dst {})",
        cond.device().string(),
        a.device().string(),
        b.device().string(),
        dst.device().string());
}

}  // namespace zt::kernel
