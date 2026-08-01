// ztensor/kernel/ArgReduceCPU.cpp
//
// CPU ArgReduce kernel.  The Indexer is built in reduction mode with
// DtypePolicy::INPUT_SAME, so output strides are zeroed on reduced axes.
// For each output slot we track the best (value, index) pair seen so far;
// tie-breaking prefers the smallest flat index within the reduced subspace.
// Implementation is serial scatter-reduce (correctness-first, matching the
// existing ReductionCPU strategy).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "kernel/ArgReduce.h"

namespace zt::kernel {
namespace {

// Identity value for the best-value tracker (not written to dst).
template<typename T>
T identity_value(ArgReduceOp op) {
    switch (op) {
        case ArgReduceOp::ArgMin:
            return std::numeric_limits<T>::max();
        case ArgReduceOp::ArgMax:
            return std::numeric_limits<T>::lowest();
    }
    ZT_LOG_ERROR("ArgReduceCPU: unsupported op {}", static_cast<int>(op));
}

// Compute the flat index within the reduced subspace for workload `wi`.
// `indexer` is already in reduction mode (output stride 0 on reduced axes).
// We decode coordinates from `wi` via primary strides, then only accumulate
// over reduced dimensions (where output stride == 0).
int64_t compute_reduced_flat_index(const core::Indexer& indexer, int64_t wi) {
    const int64_t ndim = indexer.NumDims();
    int64_t coord = wi;
    int64_t flat_idx = 0;
    int64_t stride = 1;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        const int64_t size = indexer.GetPrimaryShape()[d];
        const int64_t c = coord % size;
        coord /= size;
        if (indexer.IsReductionDim(d)) {
            flat_idx += c * stride;
            stride *= size;
        }
    }
    return flat_idx;
}

}  // namespace

void ArgReduceCPU(const Tensor& src,
                  const Tensor& dst,
                  IntArrayRef reduction_dims,
                  ArgReduceOp op) {
    ZT_CHECK(src.numel() > 0,
             "ArgReduceCPU: zero-size tensor has no argmin/argmax");

    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        const int64_t nout = dst.numel();

        // Seed the index output with -1 (sentinel: not yet assigned).
        auto* idx_out =
            static_cast<int64_t*>(const_cast<void*>(dst.data_ptr()));
        std::fill_n(idx_out, static_cast<std::size_t>(nout), int64_t{-1});

        // Best-value tracker (one per output slot), seeded with identity.
        std::vector<scalar_t> best_val(static_cast<std::size_t>(nout),
                                       identity_value<scalar_t>(op));

        // Build Indexer in reduction mode.  INPUT_SAME is used because the
        // output dtype (Long) differs from the input dtype (scalar_t).
        core::Indexer indexer(
            {src}, dst, core::DtypePolicy::INPUT_SAME, reduction_dims);
        const int64_t n = indexer.NumWorkloads();

        for (int64_t i = 0; i < n; ++i) {
            const scalar_t v = *indexer.GetInputPtr<scalar_t>(0, i);
            const int64_t out_off = indexer.GetOutputPtr<int64_t>(i) - idx_out;
            const int64_t flat_idx = compute_reduced_flat_index(indexer, i);
            bool better = false;
            const auto s = static_cast<std::size_t>(out_off);
            if (op == ArgReduceOp::ArgMin) {
                better = (v < best_val[s]) ||
                         (v == best_val[s] && flat_idx < idx_out[s]);
            } else {
                better = (v > best_val[s]) ||
                         (v == best_val[s] && flat_idx < idx_out[s]);
            }
            if (better) {
                best_val[s] = v;
                idx_out[s] = flat_idx;
            }
        }
    });
}

}  // namespace zt::kernel
