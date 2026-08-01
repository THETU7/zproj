// ztensor/kernel/UnaryEW.h
//
// Unary element-wise kernel covering arithmetic, trigonometric, hyperbolic,
// rounding, logical, and bitwise ops (§8.6.A). The op code is resolved once
// before the element loop (CPU: function pointer, CUDA: __device__ switch).

#pragma once

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

enum class UnaryEWOpCode : std::uint8_t {
    // ── arithmetic ───────────────────────────────────────────────────────
    Neg,         //  -a
    Abs,         // |a|
    Sqrt,        // √a            (float only)
    Rsqrt,       //  1/√a         (float only)
    Exp,         //  e^a          (float only)
    Expm1,       //  e^a - 1      (float only)
    Log,         // ln(a)         (float only)
    Log2,        // log₂(a)       (float only)
    Log10,       // log₁₀(a)      (float only)
    Log1p,       // ln(1 + a)     (float only)
    Reciprocal,  //  1/a
    Sigmoid,     //  1/(1 + e^-a) (float only)
    Frac,        // a - floor(a)  (float only)

    // ── trigonometric ────────────────────────────────────────────────────
    Sin,         // (float only)
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,

    // ── hyperbolic ───────────────────────────────────────────────────────
    Sinh,        // (float only)
    Cosh,
    Tanh,

    // ── rounding ─────────────────────────────────────────────────────────
    Floor,       // ⌊a⌋  (float only)
    Ceil,        // ⌈a⌉  (float only)
    Round,       // round-to-nearest-even (float only)
    Trunc,       // integer part  (float only)
    Sign,        // -1/0/+1

    // ── logical / bitwise ────────────────────────────────────────────────
    LogicalNot,  // !a
    BitwiseNot,  // ~a  (integer only)
};

/// `dst = op(src)` element-wise (broadcasting not applicable: 1:1 shape).
/// Dispatches on the device (CPU / CUDA under BUILD_CUDA_MODULE).
void UnaryEW(const Tensor& src, const Tensor& dst, UnaryEWOpCode op);

/// CPU implementation.
void UnaryEWCPU(const Tensor& src, const Tensor& dst, UnaryEWOpCode op);

/// CUDA implementation (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void UnaryEWCUDA(const Tensor& src, const Tensor& dst, UnaryEWOpCode op);
#endif

}  // namespace zt::kernel
