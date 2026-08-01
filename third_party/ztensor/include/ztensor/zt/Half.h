// ztensor/zt/Half.h
//
// IEEE 754 binary16 (half-precision) floating-point type. Modeled on PyTorch's
// c10::Half (c10/util/Half.h), restricted to the subset ztensor needs:
//
//   * 16-bit POD (alignas(2), uint16_t storage), binary-compatible with CUDA
//     __half and every other IEEE 754 binary16 type.
//   * Lossless float <-> Half conversion via bit manipulation (no FPU ops in
//     the conversion path — safe for __host__ and __device__).
//   * Arithmetic (+, -, *, /) via float promotion: convert operands to float,
//     operate in float32, round result back to Half. This is the strategy
//     PyTorch uses for element-wise kernels; the CUDA half intrinsics aren't
//     efficient on all GPUs for memory-bound workloads.
//   * Full comparison operators, unary minus, compound assignment.
//   * std::numeric_limits<zt::Half> specialization so Reduction kernels that
//     call std::numeric_limits<T>::max() / ::lowest() compile.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <ostream>

#include "ztensor/zt/Macros.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace zt {

// ============================================================================
// Half — IEEE 754 binary16
// ============================================================================
// Layout: 1 sign :: 5 exponent (bias 15) :: 10 mantissa
//          Bit 15     Bits 10-14              Bits 0-9
//
// Special values:
//   exponent == 0     : ±0 (mantissa == 0) or subnormal (mantissa != 0)
//   exponent == 0x1F  : ±Inf (mantissa == 0) or NaN  (mantissa != 0)

struct alignas(2) Half {
    uint16_t x_ = 0;

    // ---- lifecycle ----------------------------------------------------------

    Half() = default;

    /// Construct from IEEE 754 binary32. Rounds to nearest even; subnormals
    /// in the float input are rounded (may flush to zero).
    explicit ZT_HOST_DEVICE Half(float value);

    /// Reinterpret raw bits as a Half (no conversion). Use from_bits() for
    /// readability:  Half h = Half::from_bits(0x3C00);  // 1.0
    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return from_bits_t{}; }
    constexpr Half(uint16_t bits, from_bits_t) : x_(bits) {}

    // ---- conversion to/from float ------------------------------------------

    /// Implicit conversion to float (exact — half fits in float).
    ZT_HOST_DEVICE operator float() const;

    // ---- CUDA __half interop ------------------------------------------------
#ifdef __CUDACC__
    /// Implicit: construct from CUDA __half (binary-compatible, no conversion).
    ZT_HOST_DEVICE Half(__half v)
        : x_(*reinterpret_cast<const uint16_t*>(&v)) {}

    /// Explicit: reinterpret as CUDA __half (binary-compatible).
    explicit ZT_HOST_DEVICE operator __half() const {
        return *reinterpret_cast<const __half*>(&x_);
    }
#endif
};

// ============================================================================
// Free-standing conversion helpers (detail)
// ============================================================================
namespace detail {

/// Convert a 16-bit IEEE half-precision pattern to a float32 bit pattern.
/// Pure integer math — safe for __host__ and __device__.
ZT_HOST_DEVICE inline uint32_t fp16_to_fp32_bits(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000U) << 16;
    const uint32_t exp16 = (h >> 10) & 0x1FU;
    const uint32_t mant16 = h & 0x3FFU;

    if (exp16 == 0) {
        // Zero or subnormal
        if (mant16 == 0) {
            return sign;  // ±0
        }
        // Subnormal: find the leading 1 in mantissa, normalize.
        uint32_t m = mant16;
        int32_t e = -14;  // 1 - bias(15) for subnormals: exponent is -14
        while (!(m & 0x400U)) {
            m <<= 1;
            --e;
        }
        m &= 0x3FFU;  // clear the implicit leading 1
        return sign | ((uint32_t)(e + 127) << 23) | (m << 13);
    }

    if (exp16 == 0x1FU) {
        // NaN or Inf
        return sign | (0xFFU << 23) | (mant16 << 13);
    }

    // Normal case: rebias exponent from 15 to 127
    return sign | ((uint32_t)(exp16 - 15 + 127) << 23) | (mant16 << 13);
}

/// Convert a float32 bit pattern to a 16-bit IEEE half-precision pattern.
/// Rounds to nearest even. Pure integer math.
ZT_HOST_DEVICE inline uint16_t fp32_to_fp16_bits(uint32_t f_bits) {
    const uint32_t sign = (f_bits >> 16) & 0x8000U;
    const int32_t exp32 = (int32_t)((f_bits >> 23) & 0xFFU);
    const uint32_t mant32 = f_bits & 0x7FFFFFU;

    // NaN or Inf
    if (exp32 == 0xFF) {
        if (mant32 == 0) return static_cast<uint16_t>(sign | 0x7C00U);  // Inf
        // NaN: preserve the top bits of the mantissa to distinguish quiet/signaling
        return static_cast<uint16_t>(
            sign | 0x7C00U | ((mant32 >> 13) & 0x3FFU) | 0x200U);
    }

    // Normal number
    int32_t exp16 = exp32 - 127 + 15;

    if (exp16 >= 0x1F) {
        // Overflow → Inf
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (exp16 <= 0) {
        // Subnormal or zero
        if (exp16 < -10) {
            // Too small — flush to ±0
            return static_cast<uint16_t>(sign);
        }
        // Add the implicit leading 1 and shift right to create a subnormal
        uint32_t mant = (mant32 | 0x800000U) >> (1 - exp16 + 13);
        // Round to nearest even
        const uint32_t remainder = (mant32 >> (1 - exp16 + 12)) & 1U;
        const uint32_t sticky =
            (mant32 & ((1U << (1 - exp16 + 12)) - 1U)) != 0 ? 1U : 0U;
        if (remainder && (sticky || (mant & 1U))) {
            mant++;
        }
        if (mant > 0x3FFU) mant = 0x3FFU;  // clamp
        return static_cast<uint16_t>(sign | mant);
    }

    // Normal: round 23-bit mantissa to 10 bits
    uint32_t mant16 = mant32 >> 13;
    const uint32_t round_bit = (mant32 >> 12) & 1U;
    const uint32_t sticky = (mant32 & 0xFFFU) != 0 ? 1U : 0U;
    if (round_bit && (sticky || (mant16 & 1U))) {
        mant16++;
    }

    // Handle carry from rounding
    if (mant16 > 0x3FFU) {
        mant16 = 0;
        exp16++;
        if (exp16 >= 0x1F) {
            return static_cast<uint16_t>(sign | 0x7C00U);  // overflowed to Inf
        }
    }

    return static_cast<uint16_t>(sign | ((uint32_t)exp16 << 10) | mant16);
}

/// Convert IEEE half-precision bits to float.
ZT_HOST_DEVICE inline float fp16_bits_to_float(uint16_t h) {
    uint32_t bits = fp16_to_fp32_bits(h);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

/// Convert float to IEEE half-precision bits (round to nearest even).
ZT_HOST_DEVICE inline uint16_t float_to_fp16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return fp32_to_fp16_bits(bits);
}

}  // namespace detail

// ============================================================================
// Half member implementations
// ============================================================================

inline ZT_HOST_DEVICE Half::Half(float value)
    : x_(detail::float_to_fp16_bits(value)) {}

inline ZT_HOST_DEVICE Half::operator float() const {
    return detail::fp16_bits_to_float(x_);
}

// ============================================================================
// Arithmetic operators (float promotion, PyTorch convention)
// ============================================================================

ZT_HOST_DEVICE inline Half operator+(Half a, Half b) {
    return Half(static_cast<float>(a) + static_cast<float>(b));
}
ZT_HOST_DEVICE inline Half operator-(Half a, Half b) {
    return Half(static_cast<float>(a) - static_cast<float>(b));
}
ZT_HOST_DEVICE inline Half operator*(Half a, Half b) {
    return Half(static_cast<float>(a) * static_cast<float>(b));
}
ZT_HOST_DEVICE inline Half operator/(Half a, Half b) {
    return Half(static_cast<float>(a) / static_cast<float>(b));
}

ZT_HOST_DEVICE inline Half operator-(Half a) {
    return Half(-static_cast<float>(a));
}

// Compound assignment
ZT_HOST_DEVICE inline Half& operator+=(Half& a, Half b) {
    a = a + b;
    return a;
}
ZT_HOST_DEVICE inline Half& operator-=(Half& a, Half b) {
    a = a - b;
    return a;
}
ZT_HOST_DEVICE inline Half& operator*=(Half& a, Half b) {
    a = a * b;
    return a;
}
ZT_HOST_DEVICE inline Half& operator/=(Half& a, Half b) {
    a = a / b;
    return a;
}

// ============================================================================
// Comparison operators
// ============================================================================

ZT_HOST_DEVICE inline bool operator==(Half a, Half b) {
    return static_cast<float>(a) == static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator!=(Half a, Half b) {
    return static_cast<float>(a) != static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator<(Half a, Half b) {
    return static_cast<float>(a) < static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator<=(Half a, Half b) {
    return static_cast<float>(a) <= static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator>(Half a, Half b) {
    return static_cast<float>(a) > static_cast<float>(b);
}
ZT_HOST_DEVICE inline bool operator>=(Half a, Half b) {
    return static_cast<float>(a) >= static_cast<float>(b);
}

// ============================================================================
// Stream output
// ============================================================================

inline std::ostream& operator<<(std::ostream& out, Half value) {
    return out << static_cast<float>(value);
}

}  // namespace zt

// ============================================================================
// std::numeric_limits<zt::Half>
// ============================================================================
namespace std {

template <>
class numeric_limits<zt::Half> {
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
    static constexpr std::float_round_style round_style =
        std::round_to_nearest;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11;       // 10 mantissa + 1 implicit
    static constexpr int digits10 = 3;      // floor((10) * log10(2))
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;  // subnormal: -(14+10-1)
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;   // 1 past max exponent value
    static constexpr int max_exponent10 = 4;
    static constexpr bool traps = true;
    static constexpr bool tinyness_before = true;

    static constexpr ZT_HOST_DEVICE zt::Half(min)() noexcept {
        return zt::Half(0x0400, zt::Half::from_bits());  // 2^-14 ≈ 6.1e-5
    }
    static constexpr ZT_HOST_DEVICE zt::Half(lowest)() noexcept {
        return zt::Half(0xFBFF,
                        zt::Half::from_bits());  // -(2-2^-10)*2^15 ≈ -65504
    }
    static constexpr ZT_HOST_DEVICE zt::Half(max)() noexcept {
        return zt::Half(0x7BFF,
                        zt::Half::from_bits());  // (2-2^-10)*2^15 ≈ 65504
    }
    static constexpr ZT_HOST_DEVICE zt::Half(epsilon)() noexcept {
        return zt::Half(0x1400, zt::Half::from_bits());  // 2^-10 ≈ 9.77e-4
    }
    static constexpr ZT_HOST_DEVICE zt::Half(round_error)() noexcept {
        return zt::Half(0x3800, zt::Half::from_bits());  // 0.5
    }
    static constexpr ZT_HOST_DEVICE zt::Half(infinity)() noexcept {
        return zt::Half(0x7C00, zt::Half::from_bits());
    }
    static constexpr ZT_HOST_DEVICE zt::Half(quiet_NaN)() noexcept {
        return zt::Half(0x7FFF, zt::Half::from_bits());
    }
    static constexpr ZT_HOST_DEVICE zt::Half(signaling_NaN)() noexcept {
        return zt::Half(0x7DFF, zt::Half::from_bits());
    }
    static constexpr ZT_HOST_DEVICE zt::Half(denorm_min)() noexcept {
        return zt::Half(0x0001, zt::Half::from_bits());  // 2^-24 ≈ 5.96e-8
    }
};

}  // namespace std
