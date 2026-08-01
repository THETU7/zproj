// ztensor/kernel/ArgReduce.cpp
//
// Device dispatch for ArgReduce.

#include "kernel/ArgReduce.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void ArgReduce(const Tensor& src,
               const Tensor& dst,
               IntArrayRef reduction_dims,
               ArgReduceOp op) {
    if (src.is_cpu() && dst.is_cpu()) {
        ArgReduceCPU(src, dst, reduction_dims, op);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && dst.is_cuda()) {
        ArgReduceCUDA(src, dst, reduction_dims, op);
        return;
    }
#endif
    ZT_LOG_ERROR("ArgReduce: unsupported device (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

}  // namespace zt::kernel
