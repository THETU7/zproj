// ztensor/zt/BFloat16.h
//
// Google Brain Float 16 type. Modeled on PyTorch's c10::BFloat16
// (c10/util/BFloat16.h), restricted to the subset ztensor needs:
//
//   * 16-bit POD (alignas(2), uint16_t storage), binary-compatible with CUDA
//     __nv_bfloat16.
//   * BF16 is simply the top 16 bits of an IEEE 754 binary32: 1 sign,
//     8 exponent, 7 mantissa. This makes float <-> BF16 conversion trivial
//     (shift + memcpy) and preserves the full dynamic range of float32 at
//     the cost of reduced mantissa precision.
//   * Arithmetic (+, -, *, /) via float promotion (same strategy as Half).
//   * Full comparison operators, unary minus, compound assignment.
//   * std::numeric_limits<zt::BFloat16> specialization.

#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <ostream>

#include "ztensor/zt/Macros.h"

#if defined(__CUDACC__)
#include <cuda_bf16.h>
#elif defined(__HIPCC__)
#include <hip/hip_bf16.h>
// HIP exposes bf16 as __hip_bfloat16; alias it to the CUDA typename so the
// __nv_bfloat16 interop below compiles unchanged under HIP.
using __nv_bfloat16 = __hip_bfloat16;
#endif

namespace zt {

// ============================================================================
// BFloat16
// ============================================================================
// Layout: 1 sign :: 8 exponent (bias 127, same as float32) :: 7 mantissa
//          Bit 15     Bits 7-14                                 Bits 0-6
//
// BF16 is a truncated float32: the mantissa is 7 bits vs 23, but the
// exponent range is identical (including the bias of 127). This means BF16
// can represent the same magnitude range as float32 — it cannot overflow
// or underflow where float32 would not.

struct alignas(2) BFloat16 {
    uint16_t x_ = 0;

    // ---- lifecycle ----------------------------------------------------------

    BFloat16() = default;

    /// Construct from IEEE 754 binary32. Rounds to nearest even (truncation
    /// of mantissa from 23 to 7 bits with rounding).
    explicit ZT_HOST_DEVICE BFloat16(float value);

    /// Reinterpret raw bits.
    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return from_bits_t{}; }
    constexpr BFloat16(uint16_t bits, from_bits_t) : x_(bits) {}

    // ---- conversion to/from float ------------------------------------------

    /// Implicit conversion to float (exact — BF16 is a subset of float32).
    ZT_HOST_DEVICE operator float() const;

    // ---- CUDA __nv_bfloat16 interop -----------------------------------------
    // Under HIP __nv_bfloat16 is aliased to __hip_bfloat16 (see the include
    // block above), so this interop is source-identical for both compilers.
#if defined(__CUDACC__) || defined(__HIPCC__)
    /// Implicit: construct from CUDA __nv_bfloat16 (binary-compatible).
    ZT_HOST_DEVICE BFloat16(__nv_bfloat16 v)
        : x_(*reinterpret_cast<const uint16_t*>(&v)) {}

    /// Explicit: reinterpret as CUDA __nv_bfloat16.
    explicit ZT_HOST_DEVICE operator __nv_bfloat16() const {
        return *reinterpret_cast<const __nv_bfloat16*>(&x_);
    }
#endif
};

// ============================================================================
// Free-standing conversion helpers (detail)
// ============================================================================
namespace detail {

/// Convert bfloat16 bits to float. BF16 is the top 16 bits of float32, so
/// left-shift by 16 and bit-cast.
ZT_HOST_DEVICE inline float bf16_bits_to_float(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

/// Convert float to bfloat16 bits with round-to-nearest-even.
/// BF16 truncates the mantissa from 23 to 7 bits. The rounding logic adds
/// a bias that rounds the truncated bits to nearest even.
/// Uses bit-manipulation throughout (no std::isnan) to stay __host__ __device__
/// compatible.
ZT_HOST_DEVICE inline uint16_t float_to_bf16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Check for NaN: exponent == 0xFF and mantissa != 0
    const uint32_t exp = (bits >> 23) & 0xFFU;
    const uint32_t mant = bits & 0x7FFFFFU;
    if (exp == 0xFFU && mant != 0U) {
        return UINT16_C(0x7FC0);
    }
    // Round to nearest even: the rounding_bias is 0x7FFF plus the LSB of
    // the truncated result (bit 16 of the float), which implements the
    // "round to nearest, ties to even" rule.
    const uint32_t rounding_bias = ((bits >> 16) & 1U) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

}  // namespace detail

// ============================================================================
// BFloat16 member implementations
// ============================================================================

inline ZT_HOST_DEVICE BFloat16::BFloat16(float value)
    : x_(detail::float_to_bf16_bits(value)) {}

inline ZT_HOST_DEVICE BFloat16::operator float() const {
    return detail::bf16_bits_to_float(x_);
}

// ============================================================================
// Arithmetic operators (float promotion)
// ============================================================================

ZT_HOST_DEVICE inline BFloat16 operator+(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) + static_cast<float>(b));
}
ZT_HOST_DEVICE inline BFloat16 operator-(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) - static_cast<float>(b));
}
ZT_HOST_DEVICE inline BFloat16 operator*(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) * static_cast<float>(b));
}
ZT_HOST_DEVICE inline BFloat16 operator/(BFloat16 a, BFloat16 b) {
    return BFloat16(static_cast<float>(a) / static_cast<float>(b));
}

ZT_HOST_DEVICE inline BFloat16 operator-(BFloat16 a) {
    return BFloat16(-static_cast<float>(a));
}

// Compound assignment
ZT_HOST_DEVICE inline BFloat16& operator+=(BFloat16& a, BFloat16 b) {
    a = a + b;
    return a;
}
ZT_HOST_DEVICE inline BFloat16& operator-=(BFloat16& a, BFloat16 b) {
    a = a - b;
    return a;
}
ZT_HOST_DEVICE inline BFloat16& operator*=(BFloat16& a, BFloat16 b) {
    a = a * b;
    return a;
}
ZT_HOST_DEVICE inline BFloat16& operator/=(BFloat16& a, BFloat16 b) {
    a = a / b;
    return a;
}

// ============================================================================
// Comparison operators
// ============================================================================

ZT_HOST_DEVICE inline bool operator==(BFloat16 a, BFloat16 b) {
    return static_cast<float>(a) == static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator!=(BFloat16 a, BFloat16 b) {
    return static_cast<float>(a) != static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator<(BFloat16 a, BFloat16 b) {
    return static_cast<float>(a) < static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator<=(BFloat16 a, BFloat16 b) {
    return static_cast<float>(a) <= static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator>(BFloat16 a, BFloat16 b) {
    return static_cast<float>(a) > static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator>=(BFloat16 a, BFloat16 b) {
    return static_cast<float>(a) >= static_cast<float>(b);
}

// ============================================================================
// Stream output
// ============================================================================

inline std::ostream& operator<<(std::ostream& out, BFloat16 value) {
    return out << static_cast<float>(value);
}

}  // namespace zt

// ============================================================================
// std::numeric_limits<zt::BFloat16>
// ============================================================================
namespace std {

template<>
class numeric_limits<zt::BFloat16> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr std::float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;  // BF16 is not IEEE 754
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 8;    // 7 mantissa + 1 implicit
    static constexpr int digits10 = 2;  // floor(7 * log10(2))
    static constexpr int max_digits10 = 4;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;  // same as float32: 1 - 127 + 1
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;  // same as float32
    static constexpr int max_exponent10 = 38;
    static constexpr bool traps = true;
    static constexpr bool tinyness_before = true;

    static constexpr ZT_HOST_DEVICE zt::BFloat16(min)() noexcept {
        return zt::BFloat16(0x0080, zt::BFloat16::from_bits());  // ~1.2e-38
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(lowest)() noexcept {
        return zt::BFloat16(0xFF7F,
                            zt::BFloat16::from_bits());  // -3.39e38
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(max)() noexcept {
        return zt::BFloat16(0x7F7F,
                            zt::BFloat16::from_bits());  // 3.39e38
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(epsilon)() noexcept {
        return zt::BFloat16(0x3C00, zt::BFloat16::from_bits());  // 2^-7
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(round_error)() noexcept {
        return zt::BFloat16(0x3F00, zt::BFloat16::from_bits());  // 0.5
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(infinity)() noexcept {
        return zt::BFloat16(0x7F80, zt::BFloat16::from_bits());
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(quiet_NaN)() noexcept {
        return zt::BFloat16(0x7FFF, zt::BFloat16::from_bits());
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(signaling_NaN)() noexcept {
        return zt::BFloat16(0x7FDF, zt::BFloat16::from_bits());
    }
    static constexpr ZT_HOST_DEVICE zt::BFloat16(denorm_min)() noexcept {
        return zt::BFloat16(0x0001, zt::BFloat16::from_bits());  // ~9.2e-41
    }
};

}  // namespace std
