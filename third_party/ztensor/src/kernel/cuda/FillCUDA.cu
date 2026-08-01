// ztensor/kernel/FillCUDA.cu
//
// CUDA Fill kernel. Mirrors FillCPU.cpp's structure (dispatch on dst dtype,
// contiguous fast path vs strided Indexer path) but drives the work through
// core::ParallelFor, which launches the grid-stride ElementWiseKernel_.
//
// The contiguous fast path fills via a CUDA kernel rather than cudaMemset, so
// the fill shares the default stream with the rest of the op (no implicit
// synchronization) and supports the same dtype set as the strided path. The
// value is cast through the dispatched scalar_t with the same CastScalar<> as
// the CPU path, so numerical results are identical.

#include <cstddef>
#include <type_traits>

#include "ztensor/zt/BFloat16.h"
#include "ztensor/zt/Half.h"
#include "ztensor/zt/Scalar.h"

#include "core/cuda/CUDAUtils.h"
#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/Fill.h"

namespace zt {
namespace kernel {
namespace {

// Cast a Scalar to the dispatched scalar_t. Shared with FillCPU.cpp; kept here
// (rather than in a shared Impl header) because the per-op set is tiny and the
// duplication is mechanical. Exactly one branch applies per scalar_t.
template<typename scalar_t>
__host__ __device__ inline scalar_t CastScalar(const Scalar& v) {
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

void FillCUDA(const Tensor& dst, Scalar v) {
    CUDAScopedDevice scoped(dst.device());
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(dst.scalar_type(), [&] {
        const auto casted = CastScalar<scalar_t>(v);

        if (dst.is_contiguous()) {
            // Contiguous fast path: one fill kernel over the raw buffer.
            scalar_t* base =
                static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));
            const int64_t n = dst.numel();
            core::ParallelFor(dst.device(), n, [=] __device__(int64_t i) {
                base[i] = casted;
            });
            return;
        }

        // Strided: bind dst as both the (unread) input and the output so the
        // Indexer tracks dst's strides, then write the constant per workload.
        core::Indexer indexer({dst}, dst, core::DtypePolicy::NONE);
        core::ParallelFor(
            dst.device(), indexer.NumWorkloads(), [=] __device__(int64_t i) {
                *indexer.GetOutputPtr<scalar_t>(i) = casted;
            });
    });
}

}  // namespace kernel
}  // namespace zt
