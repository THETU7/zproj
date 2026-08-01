// ztensor/kernel/Copy.cpp
//
// Device dispatch for the Copy kernel.
//
//   * same device (CPU<->CPU or CUDA<->CUDA): the matching backend runs the
//     per-element cast directly (honors strides).
//   * cross device (CPU<->CUDA): staged for correctness. A GPU kernel may not
//     dereference a host pointer, so we first materialize a contiguous
//     same-dtype copy of `src` on src's device (src.contiguous()), bytewise
//     Memcpy that contiguous buffer onto dst's device, then run the (now
//     same-device) per-element cast honoring dst's strides.

#include "kernel/Copy.h"

#include "ztensor/zt/utility/Log.h"

#include "core/MemoryManager.h"

namespace zt {
namespace kernel {

#ifdef BUILD_CUDA_MODULE
namespace {

// Cross-device transfer: src and dst live on different devices. Stage a
// contiguous same-dtype buffer of src on src's device, blit it onto dst's
// device, then cast in place honoring dst's strides. src's strides are
// honored by src.contiguous() (a real copy unless already contiguous); dst's
// strides are honored by the final Copy() on dst's device.
void stage_and_cast(const Tensor& src, const Tensor& dst) {
    // 1) Contiguous same-dtype clone of src on src's device.
    const Tensor src_contig = src.contiguous();

    // 2) Allocate the same-dtype, same-shape, contiguous buffer on dst's
    // device and bytewise transfer.
    Tensor moved(Tensor::ShapeVector(src_contig.sizes().begin(),
                                     src_contig.sizes().end()),
                 src_contig.scalar_type(),
                 dst.device());
    MemoryManager::Memcpy(dst.device(),
                          moved.data_ptr(),
                          src_contig.device(),
                          src_contig.data_ptr(),
                          static_cast<std::size_t>(src_contig.numel()) *
                              src_contig.element_size());

    // 3) Per-element cast on dst's device (same-device now). Honors dst's
    // strides; the source (`moved`) is contiguous.
    Copy(moved, dst);  // same device -> matching backend, no re-entry.
}

}  // namespace
#endif

void Copy(const Tensor& src, const Tensor& dst) {
    const bool src_cpu = src.is_cpu();
    const bool dst_cpu = dst.is_cpu();
    if (src_cpu && dst_cpu) {
        CopyCPU(src, dst);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    const bool src_cuda = src.is_cuda();
    const bool dst_cuda = dst.is_cuda();
    if (src_cpu != dst_cpu && (src_cuda || dst_cuda)) {
        stage_and_cast(src, dst);
        return;
    }
    if (src_cuda && dst_cuda) {
        CopyCUDA(src, dst);
        return;
    }
#endif
    ZT_LOG_ERROR("Copy: unsupported device pair (src {}, dst {})",
                 src.device().string(),
                 dst.device().string());
}

}  // namespace kernel
}  // namespace zt
