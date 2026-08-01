// ztensor/kernel/TernaryEW.h
//
// Ternary element-wise kernel (§8.6.A).  Currently only `Where` is defined
// (cond ? a : b).  The structure mirrors BinaryEW/UnaryEW.

#pragma once

#include <cstdint>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

enum class TernaryEWOpCode : std::uint8_t {
    Where,  // dst = cond ? a : b   (cond is Bool, a/b broadcast → dst)
};

/// `dst = op(cond, a, b)` element-wise with broadcasting.  `cond`, `a`, and
/// `b` are broadcast together; the output dtype equals the promoted dtype of
/// `a` and `b`.  Dispatches on device.
void TernaryEW(const Tensor& cond,
               const Tensor& a,
               const Tensor& b,
               const Tensor& dst,
               TernaryEWOpCode op);

/// CPU implementation.
void TernaryEWCPU(const Tensor& cond,
                  const Tensor& a,
                  const Tensor& b,
                  const Tensor& dst,
                  TernaryEWOpCode op);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void TernaryEWCUDA(const Tensor& cond,
                   const Tensor& a,
                   const Tensor& b,
                   const Tensor& dst,
                   TernaryEWOpCode op);
#endif

}  // namespace zt::kernel
