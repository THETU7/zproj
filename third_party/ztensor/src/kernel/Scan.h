// ztensor/kernel/Scan.h
//
// Inclusive scan (prefix sum / cumulative product / cumulative min/max)
// along a single dimension.  The output has the same shape and dtype as the
// input.  CumMax and CumMin produce the running extreme *value*; the caller
// (Tensor method) is responsible for computing the corresponding index
// tensor when needed.

#pragma once

#include <cstdint>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

enum class ScanOpCode : std::uint8_t {
    CumSum,
    CumProd,
    CumMax,
    CumMin,
};

/// Inclusive scan of `src` along `dim`, writing into `dst`.
///
/// `dst` must have the same shape and dtype as `src` (DtypePolicy::ALL_SAME).
/// The first element along `dim` is unchanged (identity for the scan).
///
/// Dispatches on the device (CPU / CUDA under BUILD_CUDA_MODULE).
void Scan(const Tensor& src, Tensor& dst, int64_t dim, ScanOpCode op);

/// CPU implementation.
void ScanCPU(const Tensor& src, Tensor& dst, int64_t dim, ScanOpCode op);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void ScanCUDA(const Tensor& src, Tensor& dst, int64_t dim, ScanOpCode op);
#endif

}  // namespace zt::kernel
