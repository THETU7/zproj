// ztensor/kernel/Copy.h
//
// Element-wise copy with dtype cast and (optionally) device transfer.
// `Tensor::copy_` is the single entry point for dtype casts and for CPU<->
// CUDA data movement between views with arbitrary strides.
//
// The same-dtype, both-contiguous, both-same-device case is handled by a plain
// MemoryManager::Memcpy inside Tensor::copy_ (a bytewise blit). Everything
// else — dtype cast, strided decode, cross-device transfer — is routed here so
// the per-element read/cast/write logic lives in exactly one place per backend.

#pragma once

#include "ztensor/zt/Tensor.h"

namespace zt {
namespace kernel {

/// Copy `src` into `dst` element-wise, casting each element through double and
/// honoring both tensors' strides. Shapes must already match. Used for dtype
/// casts and (when one operand is on CUDA) cross-device element transfers.
///
/// Device dispatch: CPU<->CPU and CUDA<->CUDA take the matching backend;
/// CPU<->CUDA transfers route through the CUDA backend, which stages each
/// element through the device (cast happens on the GPU) — see Copy.cpp.
void Copy(const Tensor& src, const Tensor& dst);

/// CPU implementation.
void CopyCPU(const Tensor& src, const Tensor& dst);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void CopyCUDA(const Tensor& src, const Tensor& dst);
#endif

}  // namespace kernel
}  // namespace zt
