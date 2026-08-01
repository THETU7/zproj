// ztensor/kernel/CheckIndexBounds.h
//
// Bounds validation for the dim-based scatter/gather index tensors
// (DESIGN §8.6.D). Every entry of `index` must be in [-size, size) — negative
// values wrap in the kernels, anything outside that range would read/write
// out-of-bounds memory. This runs BEFORE the scatter/gather kernel so an
// invalid index throws with the operand untouched.
//
// The check is compiled out of release builds (NDEBUG); see
// Tensor.cpp::check_index_bounds. Debug builds keep it because catching a
// corrupt index is worth more than throughput.

#pragma once

#include <cstdint>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

/// Throw (via ZT_LOG_ERROR) if any entry of `index` (int64, contiguous) is
/// outside [-dim_size, dim_size). `op` names the caller for the error message.
/// Dispatches on the device.
void CheckIndexBounds(const Tensor& index, int64_t dim_size, const char* op);

/// CPU implementation: parallel scan (OpenMP) with an atomic error flag.
void CheckIndexBoundsCPU(const Tensor& index, int64_t dim_size, const char* op);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE): grid-stride
/// device scan, then an O(1) {flag, bad-value} D2H copy + host throw.
#ifdef BUILD_CUDA_MODULE
void CheckIndexBoundsCUDA(const Tensor& index,
                          int64_t dim_size,
                          const char* op);
#endif

}  // namespace zt::kernel
