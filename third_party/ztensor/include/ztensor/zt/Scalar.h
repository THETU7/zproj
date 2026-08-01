// ztensor/zt/Scalar.h
//
// A tagged 64-bit scalar value, modeled on (a heavily trimmed-down version of)
// PyTorch's c10::Scalar. It carries at most one of {bool, int64_t, double}
// plus the originating ScalarType, and is used as the "scalar operand" type
// in element-wise overloads (add(Scalar), fill_(Scalar), etc.).

#pragma once

#include <cstdint>

#include "ztensor/zt/BFloat16.h"
#include "ztensor/zt/Half.h"
#include "ztensor/zt/ScalarType.h"

namespace zt {

class Scalar {
public:
    // Default: 0 of type Long (mirrors PyTorch's behavior).
    constexpr Scalar() noexcept = default;
    constexpr Scalar(std::int64_t v) noexcept  // NOLINT(runtime/explicit)
        : type_(ScalarType::Long), as_int_(v) {}
    // Note: we deliberately do NOT provide a separate Scalar(std::int32_t)
    // overload — on targets where `int` is 32-bit it would collide with the
    // `int` overload below. `int` covers the 32-bit case portably.
    constexpr Scalar(int v) noexcept  // NOLINT(runtime/explicit)
        : type_(ScalarType::Int), as_int_(v) {}
    constexpr Scalar(double v) noexcept  // NOLINT(runtime/explicit)
        : type_(ScalarType::Double), as_double_(v) {}
    constexpr Scalar(float v) noexcept  // NOLINT(runtime/explicit)
        : type_(ScalarType::Float), as_double_(static_cast<double>(v)) {}
    constexpr Scalar(bool v) noexcept  // NOLINT(runtime/explicit)
        : type_(ScalarType::Bool), as_int_(v ? 1 : 0) {}
    /* implicit */ Scalar(Half v) noexcept  // NOLINT(runtime/explicit)
        : type_(ScalarType::Half),
          as_double_(static_cast<double>(static_cast<float>(v))) {}
    /* implicit */ Scalar(BFloat16 v) noexcept  // NOLINT(runtime/explicit)
        : type_(ScalarType::BFloat16),
          as_double_(static_cast<double>(static_cast<float>(v))) {}

    constexpr ScalarType type() const noexcept { return type_; }
    constexpr bool isIntegral() const noexcept {
        return zt::isIntegralType(type_) || zt::isBooleanType(type_);
    }
    constexpr bool isFloatingPoint() const noexcept {
        return zt::isFloatingType(type_);
    }

    // Typed accessors. Behavior is undefined if the stored value cannot be
    // losslessly represented in the requested type.
    constexpr bool toBool() const noexcept {
        return isFloatingPoint() ? as_double_ != 0.0 : as_int_ != 0;
    }
    constexpr std::int64_t toInt64() const noexcept {
        return isFloatingPoint() ? static_cast<std::int64_t>(as_double_)
                                 : as_int_;
    }
    constexpr double toDouble() const noexcept {
        return isFloatingPoint() ? as_double_ : static_cast<double>(as_int_);
    }

private:
    ScalarType type_ = ScalarType::Long;
    union {
        // `as_int_` is the active member when type_ == Long (the default), so
        // it carries the default-member-initializer. A defaulted default ctor
        // of a class with an anonymous union member can be `constexpr` in
        // C++17 only if exactly one union member has a default initializer;
        // clang enforces this (C++23 relaxes the rule).
        std::int64_t as_int_ = 0;
        double as_double_;
    };
};

}  // namespace zt
