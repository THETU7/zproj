// ztensor/kernel/Scatter.cpp
//
// Device dispatch for Gather / Scatter. Operands must already share a device
// (Tensor::gather / scatter_ / scatter_add_ stage cross-device values via Copy
// before calling here).

#include "kernel/Scatter.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void Gather(const Tensor& src,
            const Tensor& index,
            const Tensor& dst,
            int64_t dim) {
    if (src.is_cpu() && index.is_cpu() && dst.is_cpu()) {
        GatherCPU(src, index, dst, dim);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && index.is_cuda() && dst.is_cuda()) {
        GatherCUDA(src, index, dst, dim);
        return;
    }
#endif
    ZT_LOG_ERROR("Gather: unsupported device (src {}, index {}, dst {})",
                 src.device().string(),
                 index.device().string(),
                 dst.device().string());
}

void Scatter(const Tensor& src,
             const Tensor& index,
             const Tensor& dst,
             int64_t dim,
             bool accumulate) {
    if (src.is_cpu() && index.is_cpu() && dst.is_cpu()) {
        ScatterCPU(src, index, dst, dim, accumulate);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && index.is_cuda() && dst.is_cuda()) {
        ScatterCUDA(src, index, dst, dim, accumulate);
        return;
    }
#endif
    ZT_LOG_ERROR("Scatter: unsupported device (src {}, index {}, dst {})",
                 src.device().string(),
                 index.device().string(),
                 dst.device().string());
}

}  // namespace zt::kernel
