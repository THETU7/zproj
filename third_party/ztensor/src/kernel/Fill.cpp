// ztensor/kernel/Fill.cpp
//
// Device dispatch for the Fill kernel. Phase 3 wires only the CPU path; the
// CUDA branch is added in phase 4 behind BUILD_CUDA_MODULE.

#include "kernel/Fill.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void Fill(const Tensor& dst, Scalar v) {
    if (dst.is_cpu()) {
        FillCPU(dst, v);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (dst.is_cuda()) {
        FillCUDA(dst, v);
        return;
    }
#endif
    ZT_LOG_ERROR("Fill: unsupported device {}", dst.device().string());
}

}  // namespace zt::kernel
