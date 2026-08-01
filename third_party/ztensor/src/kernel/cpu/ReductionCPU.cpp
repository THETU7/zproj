// ztensor/kernel/ReductionCPU.cpp
//
// CPU Reduction kernel. The Indexer is built in reduction mode (output strides
// zeroed on reduced axes), so each workload reads one source element and maps
// to its output slot; we scatter-reduce serially into a dst pre-seeded with the
// op identity. Mean accumulates as Sum, then scales. This is correctness-first
// (matches NumPy); the TwoPass / ParallelDim strategies are a future
// optimization for large inputs.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "kernel/Reduction.h"

namespace zt::kernel {
namespace {

// NaN detection helper: integral types are never NaN; floating-point types
// use std::isnan; Half/BFloat16 convert to float for the check.
template<typename T>
constexpr bool is_nan(T /*v*/) {
    return false;
}
template<>
inline bool is_nan<float>(float v) {
    return std::isnan(v);
}
template<>
inline bool is_nan<double>(double v) {
    return std::isnan(v);
}
template<>
inline bool is_nan<Half>(Half v) {
    return std::isnan(static_cast<float>(v));
}
template<>
inline bool is_nan<BFloat16>(BFloat16 v) {
    return std::isnan(static_cast<float>(v));
}

// Identity element for `op`. Min/Max error on empty inputs at the call site.
template<typename T>
T identity_value(ReductionOpCode op) {
    switch (op) {
        case ReductionOpCode::Sum:
        case ReductionOpCode::Mean:
            return static_cast<T>(0);
        case ReductionOpCode::Min:
        case ReductionOpCode::NanMin:
            return std::numeric_limits<T>::max();
        case ReductionOpCode::Max:
        case ReductionOpCode::NanMax:
            return std::numeric_limits<T>::lowest();
        case ReductionOpCode::Prod:
            return static_cast<T>(1);
        case ReductionOpCode::All:
            return static_cast<T>(true);
        case ReductionOpCode::Any:
            return static_cast<T>(false);
    }
    ZT_LOG_ERROR("ReductionCPU: unsupported op code {}", static_cast<int>(op));
}

// Fold one source value `v` into the running accumulator `acc`.
template<typename T>
T combine_value(ReductionOpCode op, T acc, T v) {
    switch (op) {
        case ReductionOpCode::Sum:
        case ReductionOpCode::Mean:
            return static_cast<T>(acc + v);
        case ReductionOpCode::Min:
            return v < acc ? v : acc;
        case ReductionOpCode::Max:
            return v > acc ? v : acc;
        case ReductionOpCode::Prod:
            return static_cast<T>(acc * v);
        case ReductionOpCode::NanMin:
            if (is_nan(v)) {
                return acc;
            }
            return v < acc ? v : acc;
        case ReductionOpCode::NanMax:
            if (is_nan(v)) {
                return acc;
            }
            return v > acc ? v : acc;
        case ReductionOpCode::All:
            return static_cast<T>(acc && v);
        case ReductionOpCode::Any:
            return static_cast<T>(acc || v);
    }
    ZT_LOG_ERROR("ReductionCPU: unsupported op code {}", static_cast<int>(op));
}

}  // namespace

void ReductionCPU(const Tensor& src,
                  const Tensor& dst,
                  IntArrayRef reduction_dims,
                  ReductionOpCode op) {
    // Sum/Prod/All/Any of an empty tensor is the identity; Min/Max/Mean are
    // undefined on an empty reduction set.
    if (src.numel() == 0 && op != ReductionOpCode::Sum &&
        op != ReductionOpCode::Prod && op != ReductionOpCode::All &&
        op != ReductionOpCode::Any) {
        ZT_LOG_ERROR("ReductionCPU: zero-size tensor has no identity for op {}",
                     static_cast<int>(op));
    }

    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        const auto identity = identity_value<scalar_t>(op);

        // Seed dst with the identity (dst is freshly allocated / contiguous).
        auto* dptr = static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));
        const int64_t nout = dst.numel();
        std::fill_n(dptr, static_cast<std::size_t>(nout), identity);

        // Serial scatter-reduce: NumWorkloads == src.numel(); reduced axes map
        // many workloads onto the same output slot (output stride 0).
        core::Indexer indexer(
            {src}, dst, core::DtypePolicy::ALL_SAME, reduction_dims);
        const int64_t n = indexer.NumWorkloads();
        for (int64_t i = 0; i < n; ++i) {
            auto* out = indexer.GetOutputPtr<scalar_t>(i);
            *out =
                combine_value(op, *out, *indexer.GetInputPtr<scalar_t>(0, i));
        }

        // Mean: scale the accumulated sums by 1 / reduction_count.
        if (op == ReductionOpCode::Mean) {
            const int64_t count = nout > 0 ? n / nout : 1;
            const double inv = 1.0 / static_cast<double>(count);
            for (int64_t j = 0; j < nout; ++j) {
                const auto uj = static_cast<std::size_t>(j);
                const double v = static_cast<double>(dptr[uj]) * inv;
                dptr[uj] = static_cast<scalar_t>(v);
            }
        }
    });
}

}  // namespace zt::kernel
