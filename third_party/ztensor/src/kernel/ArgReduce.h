// ztensor/kernel/ArgReduce.h
//
// ArgReduce kernel: argmin / argmax over arbitrary axes. Returns int64
// indices of the extreme values.  Tie-breaking always selects the smallest
// flat index within the reduced subspace (matching PyTorch/NumPy).
//
// The output dtype is always ScalarType::Long (int64), regardless of the
// input dtype.  Uses DtypePolicy::INPUT_SAME inside the Indexer.

#pragma once

#include <cstdint>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

enum class ArgReduceOp : std::uint8_t {
    ArgMin,
    ArgMax,
};

/// Reduce `src` over `reduction_dims` into `dst` (int64 indices).
///
/// `dst` must already carry the keepdim reduction shape (size-1 on the
/// reduced axes); the caller squeezes afterwards if keepdim=false is desired.
/// `src.scalar_type()` is numeric; `dst.scalar_type()` is always Long.
///
/// Dispatches on the device (CPU / CUDA under BUILD_CUDA_MODULE).
void ArgReduce(const Tensor& src,
               const Tensor& dst,
               IntArrayRef reduction_dims,
               ArgReduceOp op);

/// CPU implementation.
void ArgReduceCPU(const Tensor& src,
                  const Tensor& dst,
                  IntArrayRef reduction_dims,
                  ArgReduceOp op);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void ArgReduceCUDA(const Tensor& src,
                   const Tensor& dst,
                   IntArrayRef reduction_dims,
                   ArgReduceOp op);
#endif

}  // namespace zt::kernel
