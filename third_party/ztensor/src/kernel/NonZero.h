// ztensor/kernel/NonZero.h
//
// NonZero helper: given a bool tensor, return the coordinates of its true
// elements as a vector of int64 tensors — one per dim of the input, each of
// shape {num_true}. This is the form Open3D/PyTorch's nonzero uses to expand a
// boolean mask into advanced-indexing operands.
//
// Internal-only (not exposed on Tensor); consumed by the advanced-indexing
// preprocessor's bool-key expansion.

#pragma once

#include <vector>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

/// `out[d]` = the d-th coordinate of every true element of `mask`, int64,
/// shape {num_true}. `mask` must be Bool. Dispatches on the device.
std::vector<Tensor> NonZero(const Tensor& mask);

/// CPU implementation.
std::vector<Tensor> NonZeroCPU(const Tensor& mask);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
std::vector<Tensor> NonZeroCUDA(const Tensor& mask);
#endif

}  // namespace zt::kernel
