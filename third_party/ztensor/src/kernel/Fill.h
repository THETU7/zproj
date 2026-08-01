// ztensor/kernel/Fill.h
//
// Fill kernel: write a scalar into every element of a (possibly strided)
// destination view. First concrete instance of the op-quartet pattern
// (Op{.h,.cpp,CPU.cpp}); the CPU path is implemented now, the CUDA path arrives
// in phase 4 behind the usual #ifdef.

#pragma once

#include "ztensor/zt/Scalar.h"
#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

/// Fill `dst` with the scalar value `v` (broadcast across every element).
/// Handles both contiguous and strided views. Dispatches on the destination's
/// device (CPU / CUDA under BUILD_CUDA_MODULE).
void Fill(const Tensor& dst, Scalar v);

/// CPU implementation. Exposed so the CUDA dispatcher can fall back to it and
/// so unit tests can target the CPU path directly.
void FillCPU(const Tensor& dst, Scalar v);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void FillCUDA(const Tensor& dst, Scalar v);
#endif

}  // namespace zt::kernel
