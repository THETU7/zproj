// ztensor/kernel/FillCPU.cpp
//
// CPU Fill kernel: dispatch on the destination dtype, then either a contiguous
// fast path (std::fill_n over the raw buffer) or a strided path that walks the
// destination's layout via an Indexer + ParallelFor.

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "ztensor/zt/BFloat16.h"
#include "ztensor/zt/Half.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/Fill.h"

namespace zt::kernel {
namespace {

// Cast a Scalar to the dispatched scalar_t. Exactly one branch applies per
// scalar_t (floats take the double path, bool takes toBool, every other
// integer type takes the int64 path); the rest are discarded by if constexpr.
template<typename scalar_t>
inline scalar_t CastScalar(const Scalar& v) {
    if constexpr (std::is_floating_point_v<scalar_t>) {
        return static_cast<scalar_t>(v.toDouble());
    }
    if constexpr (std::is_same_v<scalar_t, bool>) {
        return v.toBool();
    }
    if constexpr (std::is_same_v<scalar_t, Half> ||
                  std::is_same_v<scalar_t, BFloat16>) {
        return static_cast<scalar_t>(static_cast<float>(v.toDouble()));
    }
    return static_cast<scalar_t>(v.toInt64());
}

}  // namespace

void FillCPU(const Tensor& dst, Scalar v) {
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(dst.scalar_type(), [&] {
        const auto casted = CastScalar<scalar_t>(v);

        if (dst.is_contiguous()) {
            // Output kernels take a const Tensor& and write through it,
            // mirroring TensorRef's const_cast-on-construct (Open3D
            // convention). For the contiguous case std::fill_n lowers to
            // memset-class code.
            auto* base =
                static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));
            std::fill_n(base, static_cast<std::size_t>(dst.numel()), casted);
            return;
        }

        // Strided: bind dst as both the (unread) input and the output so the
        // Indexer tracks dst's strides, then write the constant per workload.
        core::Indexer indexer({dst}, dst, core::DtypePolicy::NONE);
        core::ParallelFor(dst.device(), indexer.NumWorkloads(), [&](int64_t i) {
            *indexer.GetOutputPtr<scalar_t>(i) = casted;
        });
    });
}

}  // namespace zt::kernel
