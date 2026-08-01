// ztensor/kernel/BinaryEW.h
//
// Binary element-wise kernel covering arithmetic (Add/Sub/Mul/Div),
// comparisons (Eq/Ne/Lt/Le/Gt/Ge), extended arithmetic (Pow/Fmod/...),
// logical (And/Or/Xor → Bool output), and bitwise ops (§8.6.A).

#pragma once

#include <cstdint>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

enum class BinaryEWOpCode : std::uint8_t {
    // ── arithmetic (dst dtype == lhs/rhs dtype) ─────────────────────────
    Add,
    Sub,
    Mul,
    Div,

    // ── comparisons (dst dtype == Bool) ─────────────────────────────────
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // ── extended arithmetic (ALL_SAME dispatch) ────────────────────────
    Pow,
    Fmod,
    Remainder,
    Maximum,
    Minimum,
    Atan2,
    Hypot,

    // ── logical (Bool output, same as comparisons) ─────────────────────
    LogicalAnd,
    LogicalOr,
    LogicalXor,

    // ── bitwise (ALL_SAME, integral only) ──────────────────────────────
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Lshift,
    Rshift,
};

/// `dst = op(lhs, rhs)` element-wise with broadcasting (lhs/rhs broadcast to
/// dst's shape). Dispatches on the device (CPU / CUDA under BUILD_CUDA_MODULE).
void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              const Tensor& dst,
              BinaryEWOpCode op);

/// CPU implementation.
void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 const Tensor& dst,
                 BinaryEWOpCode op);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void BinaryEWCUDA(const Tensor& lhs,
                  const Tensor& rhs,
                  const Tensor& dst,
                  BinaryEWOpCode op);
#endif

}  // namespace zt::kernel
