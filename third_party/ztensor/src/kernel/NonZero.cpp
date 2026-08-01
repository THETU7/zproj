// ztensor/kernel/NonZero.cpp
//
// Device dispatch for NonZero.

#include "kernel/NonZero.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

std::vector<Tensor> NonZero(const Tensor& mask) {
    ZT_CHECK(mask.scalar_type() == ScalarType::Bool,
             "NonZero: mask must be Bool, got {}",
             toString(mask.scalar_type()));
    if (mask.is_cpu()) {
        return NonZeroCPU(mask);
    }
#ifdef BUILD_CUDA_MODULE
    if (mask.is_cuda()) {
        return NonZeroCUDA(mask);
    }
#endif
    ZT_LOG_ERROR("NonZero: unsupported device {}", mask.device().string());
}

}  // namespace zt::kernel
