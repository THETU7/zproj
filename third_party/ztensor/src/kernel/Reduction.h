// ztensor/kernel/Reduction.h
//
// Reduction kernel (Sum / Mean / Min / Max) over arbitrary axes. The CPU path
// is a serial scatter-reduce seeded with the op's identity; Mean accumulates as
// Sum and scales by 1/reduction_count afterwards.

#pragma once

#include <cstdint>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

enum class ReductionOpCode : std::uint8_t {
    Sum,
    Mean,  // Sum + post-scale by 1/reduction_count.
    Min,
    Max,
    Prod,    // Product (identity = 1, combine = acc * v).
    NanMin,  // Min that skips NaN (identity = max).
    NanMax,  // Max that skips NaN (identity = lowest).
    All,     // Logical AND — Bool input, Bool output (identity = true).
    Any,     // Logical OR  — Bool input, Bool output (identity = false).
};

/// Reduce `src` over `reduction_dims` into `dst`.
///
/// `dst` must already carry the keepdim reduction shape (size-1 on the reduced
/// axes); the caller squeezes afterwards if keepdim=false is desired.
/// `src.scalar_type() == dst.scalar_type()` (DtypePolicy::ALL_SAME) — the
/// caller pre-casts `src` to the intended output dtype.
///
/// Dispatches on the device (CPU / CUDA under BUILD_CUDA_MODULE).
void Reduction(const Tensor& src,
               const Tensor& dst,
               IntArrayRef reduction_dims,
               ReductionOpCode op);

/// CPU implementation.
void ReductionCPU(const Tensor& src,
                  const Tensor& dst,
                  IntArrayRef reduction_dims,
                  ReductionOpCode op);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void ReductionCUDA(const Tensor& src,
                   const Tensor& dst,
                   IntArrayRef reduction_dims,
                   ReductionOpCode op);
#endif

}  // namespace zt::kernel
