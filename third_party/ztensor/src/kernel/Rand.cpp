// ztensor/kernel/Rand.cpp
//
// Device dispatch for the Rand / RandN / RandInt kernels.

#include "kernel/Rand.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void Rand(const Tensor& dst, double from, double to, const Generator& gen) {
    if (dst.is_cpu()) {
        RandCPU(dst, from, to, gen);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (dst.is_cuda()) {
        RandCUDA(dst, from, to, gen);
        return;
    }
#endif
    ZT_LOG_ERROR("Rand: unsupported device {}", dst.device().string());
}

void RandN(const Tensor& dst,
           double mean,
           double stddev,
           const Generator& gen) {
    if (dst.is_cpu()) {
        RandNCPU(dst, mean, stddev, gen);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (dst.is_cuda()) {
        RandNCUDA(dst, mean, stddev, gen);
        return;
    }
#endif
    ZT_LOG_ERROR("RandN: unsupported device {}", dst.device().string());
}

void RandInt(const Tensor& dst,
             int64_t low,
             int64_t high,
             const Generator& gen) {
    if (dst.is_cpu()) {
        RandIntCPU(dst, low, high, gen);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (dst.is_cuda()) {
        RandIntCUDA(dst, low, high, gen);
        return;
    }
#endif
    ZT_LOG_ERROR("RandInt: unsupported device {}", dst.device().string());
}

}  // namespace zt::kernel
