// ztensor/zt/ScalarType.h
//
// Element type enumeration for zt::Tensor. Modeled on PyTorch's
// c10::ScalarType, restricted to the numeric subset ztensor supports.
//
// Phase 1 scope: Byte/Char/Short/Int/Long/Half/Float/Double/Bool.
// BFloat16 added in phase 5 (§8.5.B).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace zt {

// Forward declarations for types that map to ScalarType values.
struct Half;
struct BFloat16;

// Numeric element types of a Tensor. Order is stable: a subset of PyTorch's
// ScalarType ordering is preserved so that future expansion is painless.
enum class ScalarType : int8_t {
    Undefined = 0,
    Byte,    // uint8_t
    Char,    // int8_t
    Short,   // int16_t
    Int,     // int32_t
    Long,    // int64_t
    Half,    // 16-bit float
    BFloat16, // brain float 16
    Float,   // float
    Double,  // double
    Bool,    // bool
    NumScalarTypes,
};

// PyTorch-style constexpr aliases (kByte, kFloat, ...).
inline constexpr ScalarType kUndefined = ScalarType::Undefined;
inline constexpr ScalarType kByte = ScalarType::Byte;
inline constexpr ScalarType kChar = ScalarType::Char;
inline constexpr ScalarType kShort = ScalarType::Short;
inline constexpr ScalarType kInt = ScalarType::Int;
inline constexpr ScalarType kLong = ScalarType::Long;
inline constexpr ScalarType kHalf = ScalarType::Half;
inline constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
inline constexpr ScalarType kFloat = ScalarType::Float;
inline constexpr ScalarType kDouble = ScalarType::Double;
inline constexpr ScalarType kBool = ScalarType::Bool;

// Category of a scalar type (mirrors Open3D's DtypeCode concept).
enum class ScalarTypeCode : int8_t {
    Undefined = 0,
    Bool,
    Int,
    UInt,
    Float,
};

// Size of one element of `dtype` in bytes. Returns 0 for Undefined.
constexpr std::size_t elementSize(ScalarType dtype);

// Human-readable name (e.g. "Float", "Int64" in PyTorch's spelling).
constexpr std::string_view toString(ScalarType dtype);

// Category classification.
constexpr bool isIntegralType(ScalarType dtype);
constexpr bool isFloatingType(ScalarType dtype);
constexpr bool isSignedType(ScalarType dtype);
constexpr bool isBooleanType(ScalarType dtype);
constexpr ScalarTypeCode scalarTypeCode(ScalarType dtype);

// Can a value of `src` be safely promoted/stored into `dst`?
constexpr bool canCast(ScalarType src, ScalarType dst);

// NumPy-style type promotion of two scalar types.
constexpr ScalarType promoteTypes(ScalarType a, ScalarType b);

// C++ type <-> ScalarType mappings.
//   ScalarTypeToCppType<dt>::type   (static)
//   CppTypeToScalarType<T>          (constexpr)
template<ScalarType dtype>
struct ScalarTypeToCppType;  // primary undefined; specializations below.

template<typename T>
struct CppTypeToScalarType;  // primary undefined; specializations below.

#define ZT_DEFINE_SCALAR_TRAITS(cpp_type, enum_val)               \
    template<>                                                    \
    struct ScalarTypeToCppType<ScalarType::enum_val> {            \
        using type = cpp_type;                                    \
    };                                                            \
    template<>                                                    \
    struct CppTypeToScalarType<cpp_type> {                        \
        static constexpr ScalarType value = ScalarType::enum_val; \
    }

ZT_DEFINE_SCALAR_TRAITS(uint8_t, Byte);
ZT_DEFINE_SCALAR_TRAITS(int8_t, Char);
ZT_DEFINE_SCALAR_TRAITS(int16_t, Short);
ZT_DEFINE_SCALAR_TRAITS(int32_t, Int);
ZT_DEFINE_SCALAR_TRAITS(int64_t, Long);
ZT_DEFINE_SCALAR_TRAITS(float, Float);
ZT_DEFINE_SCALAR_TRAITS(double, Double);
ZT_DEFINE_SCALAR_TRAITS(bool, Bool);
ZT_DEFINE_SCALAR_TRAITS(zt::Half, Half);
ZT_DEFINE_SCALAR_TRAITS(zt::BFloat16, BFloat16);

#undef ZT_DEFINE_SCALAR_TRAITS

}  // namespace zt

// ---------------------------------------------------------------------------
// Definitions (constexpr, header-only by design).
// ---------------------------------------------------------------------------

namespace zt {

constexpr std::size_t elementSize(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::Bool:
        case ScalarType::Byte:
        case ScalarType::Char:
            return 1;
        case ScalarType::Short:
            return 2;
        case ScalarType::Half:
        case ScalarType::BFloat16:
            return 2;
        case ScalarType::Int:
        case ScalarType::Float:
            return 4;
        case ScalarType::Long:
        case ScalarType::Double:
            return 8;
        case ScalarType::Undefined:
        case ScalarType::NumScalarTypes:
            return 0;
    }
    return 0;
}

constexpr ScalarTypeCode scalarTypeCode(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::Bool:
            return ScalarTypeCode::Bool;
        case ScalarType::Byte:
            return ScalarTypeCode::UInt;
        case ScalarType::Char:
        case ScalarType::Short:
        case ScalarType::Int:
        case ScalarType::Long:
            return ScalarTypeCode::Int;
        case ScalarType::Half:
        case ScalarType::BFloat16:
        case ScalarType::Float:
        case ScalarType::Double:
            return ScalarTypeCode::Float;
        case ScalarType::Undefined:
        case ScalarType::NumScalarTypes:
            return ScalarTypeCode::Undefined;
    }
    return ScalarTypeCode::Undefined;
}

constexpr bool isBooleanType(ScalarType dtype) {
    return dtype == ScalarType::Bool;
}
constexpr bool isIntegralType(ScalarType dtype) {
    const auto code = scalarTypeCode(dtype);
    return code == ScalarTypeCode::Int || code == ScalarTypeCode::UInt;
}
constexpr bool isFloatingType(ScalarType dtype) {
    return scalarTypeCode(dtype) == ScalarTypeCode::Float;
}
constexpr bool isSignedType(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::Char:
        case ScalarType::Short:
        case ScalarType::Int:
        case ScalarType::Long:
        case ScalarType::Half:
        case ScalarType::BFloat16:
        case ScalarType::Float:
        case ScalarType::Double:
            return true;
        default:
            return false;
    }
}

constexpr std::string_view toString(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::Undefined:
            return "Undefined";
        case ScalarType::Byte:
            return "UInt8";
        case ScalarType::Char:
            return "Int8";
        case ScalarType::Short:
            return "Int16";
        case ScalarType::Int:
            return "Int32";
        case ScalarType::Long:
            return "Int64";
        case ScalarType::Half:
            return "Float16";
        case ScalarType::BFloat16:
            return "BFloat16";
        case ScalarType::Float:
            return "Float32";
        case ScalarType::Double:
            return "Float64";
        case ScalarType::Bool:
            return "Bool";
        case ScalarType::NumScalarTypes:
            return "NumScalarTypes";
    }
    return "Unknown";
}

// Conservative cast rules: same-category and float<-int are allowed; never
// downcast floats to narrower ints implicitly.
constexpr bool canCast(ScalarType src, ScalarType dst) {
    if (src == ScalarType::Undefined || dst == ScalarType::Undefined)
        return false;
    if (src == dst) return true;
    const auto sc = scalarTypeCode(src);
    const auto dc = scalarTypeCode(dst);
    if (sc == dc) {
        // Same category: allow if dst is at least as wide.
        return elementSize(dst) >= elementSize(src);
    }
    // float <- (int/uint) allowed; int <- float disallowed.
    if (dc == ScalarTypeCode::Float) {
        return elementSize(dst) >= elementSize(src) ||
               true;  // floats accept ints
    }
    return false;
}

constexpr ScalarType promoteTypes(ScalarType a, ScalarType b) {
    if (a == ScalarType::Undefined) return b;
    if (b == ScalarType::Undefined) return a;
    // Simple rules: same category -> wider; mixing int/float -> float.
    if (scalarTypeCode(a) == scalarTypeCode(b)) {
        return elementSize(a) >= elementSize(b) ? a : b;
    }
    if (isFloatingType(a) || isFloatingType(b)) {
        if (a == ScalarType::Double || b == ScalarType::Double)
            return ScalarType::Double;
        return ScalarType::Float;
    }
    // int vs uint -> wider int
    const ScalarType wider = elementSize(a) >= elementSize(b) ? a : b;
    switch (wider) {
        case ScalarType::Byte:
            return ScalarType::Short;
        case ScalarType::Char:
            return ScalarType::Char;
        case ScalarType::Short:
            return ScalarType::Short;
        case ScalarType::Int:
            return ScalarType::Int;
        case ScalarType::Long:
            return ScalarType::Long;
        default:
            return wider;
    }
}

}  // namespace zt
