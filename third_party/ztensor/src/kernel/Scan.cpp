// ztensor/kernel/Scan.cpp
//
// Device dispatch for Scan.

#include "kernel/Scan.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void Scan(const Tensor& src, Tensor& dst, int64_t dim, ScanOpCode op) {
    if (src.is_cpu() && dst.is_cpu()) {
        ScanCPU(src, dst, dim, op);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && dst.is_cuda()) {
        ScanCUDA(src, dst, dim, op);
        return;
    }
#endif
    ZT_LOG_ERROR("Scan: unsupported device (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

}  // namespace zt::kernel
