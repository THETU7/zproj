// ztensor/kernel/CopyCUDA.cu
//
// CUDA Copy kernel: element-wise cast src.dtype() -> dst.dtype() with strides,
// on a single CUDA device. A double dtype-dispatch expands the per-element
// read-cast-write as a __device__ lambda over an Indexer + ParallelFor.
//
// Cross-device (CPU<->CUDA) pairs never reach this file: kernel::Copy's device
// dispatch routes them through stage_and_cast() in Copy.cpp, which materializes
// a contiguous same-dtype buffer of src on the device and casts in place, so
// this kernel only ever sees both operands on the same (CUDA) device.

#include "ztensor/zt/cuda/Guard.h"
#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/Copy.h"

namespace zt {
namespace kernel {
namespace {

// Per-element copy with cast, templated on (src_t, dst_t). CUDA device lambda.
template<typename src_t, typename dst_t>
void copy_element_wise(const Tensor& src, const Tensor& dst) {
    core::Indexer indexer({src}, dst, core::DtypePolicy::NONE);
    core::ParallelFor(
        dst.device(), indexer.NumWorkloads(), [=] __device__(int64_t i) {
            const src_t v = *indexer.GetInputPtr<src_t>(0, i);
            *indexer.GetOutputPtr<dst_t>(i) = static_cast<dst_t>(v);
        });
}

// Inner dispatch on dst dtype; expands copy_element_wise<src_t, dst_t>.
#define ZT_COPY_INNER(DST_CPP)                           \
    return ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL( \
        src.scalar_type(),                               \
        [&] { copy_element_wise<scalar_t, DST_CPP>(src, dst); });

void copy_dispatch(const Tensor& src, const Tensor& dst) {
    switch (dst.scalar_type()) {
        case ScalarType::Bool:
            ZT_COPY_INNER(bool)
        case ScalarType::Byte:
            ZT_COPY_INNER(std::uint8_t)
        case ScalarType::Char:
            ZT_COPY_INNER(std::int8_t)
        case ScalarType::Short:
            ZT_COPY_INNER(std::int16_t)
        case ScalarType::Int:
            ZT_COPY_INNER(std::int32_t)
        case ScalarType::Long:
            ZT_COPY_INNER(std::int64_t)
        case ScalarType::Half:
            ZT_COPY_INNER(zt::Half)
        case ScalarType::BFloat16:
            ZT_COPY_INNER(zt::BFloat16)
        case ScalarType::Float:
            ZT_COPY_INNER(float)
        case ScalarType::Double:
            ZT_COPY_INNER(double)
        default:
            break;
    }
#undef ZT_COPY_INNER
    ZT_LOG_ERROR("CopyCUDA: unsupported dst dtype {}",
                 toString(dst.scalar_type()));
}

}  // namespace

void CopyCUDA(const Tensor& src, const Tensor& dst) {
    CUDAScopedDevice scoped(dst.device());
    copy_dispatch(src, dst);
}

}  // namespace kernel
}  // namespace zt
