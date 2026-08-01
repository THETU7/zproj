// ztensor/kernel/Index.cpp
//
// Device dispatch for IndexGet / IndexSet. Both operands must already be on
// the same device (Tensor::index/index_put_ handle cross-device staging via
// Copy before calling here).

#include "kernel/Index.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void IndexGet(const Tensor& src,
              const Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const std::vector<int64_t>& indexed_shape,
              const std::vector<int64_t>& indexed_strides) {
    if (src.is_cpu() && dst.is_cpu()) {
        IndexGetCPU(src, dst, index_tensors, indexed_shape, indexed_strides);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && dst.is_cuda()) {
        IndexGetCUDA(src, dst, index_tensors, indexed_shape, indexed_strides);
        return;
    }
#endif
    ZT_LOG_ERROR("IndexGet: unsupported device (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

void IndexSet(const Tensor& src,
              const Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const std::vector<int64_t>& indexed_shape,
              const std::vector<int64_t>& indexed_strides) {
    if (src.is_cpu() && dst.is_cpu()) {
        IndexSetCPU(src, dst, index_tensors, indexed_shape, indexed_strides);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && dst.is_cuda()) {
        IndexSetCUDA(src, dst, index_tensors, indexed_shape, indexed_strides);
        return;
    }
#endif
    ZT_LOG_ERROR("IndexSet: unsupported device (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

void IndexAdd(const Tensor& src,
              const Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const std::vector<int64_t>& indexed_shape,
              const std::vector<int64_t>& indexed_strides) {
    if (src.is_cpu() && dst.is_cpu()) {
        IndexAddCPU(src, dst, index_tensors, indexed_shape, indexed_strides);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (src.is_cuda() && dst.is_cuda()) {
        IndexAddCUDA(src, dst, index_tensors, indexed_shape, indexed_strides);
        return;
    }
#endif
    ZT_LOG_ERROR("IndexAdd: unsupported device (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

}  // namespace zt::kernel
