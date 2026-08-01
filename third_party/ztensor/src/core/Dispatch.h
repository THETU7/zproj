// ztensor/core/Dispatch.h
//
// Dtype -> C++ template dispatch macros. Modeled on Open3D's
// open3d/core/Dispatch.h (itself inspired by PyTorch's ATen/Dispatch.h).
//
// Usage:
//   ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(dtype, [&] {
//       // scalar_t is in scope here (float/double/int32_t/...)
//       RunKernel<scalar_t>(...);
//   });
//
// Variants:
//   _TO_TEMPLATE            numeric types only (no Bool)
//   _TO_TEMPLATE_WITH_BOOL  adds Bool
//   _FLOAT_ONLY             Float / Double
//   _INT_ONLY               Byte/Char/Short/Int/Long
//
// On an unsupported dtype the trailing ZT_LOG_ERROR throws (it is
// [[noreturn]]). Each macro is a chain of independent `if` blocks (not
// if/else) so that call sites stay clean under readability-else-after-return.

#pragma once

#include <cstdint>

#include "ztensor/zt/BFloat16.h"
#include "ztensor/zt/Half.h"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/utility/Log.h"

// Numeric types (Byte/Char/Short/Int/Long/Float/Double). Bool excluded.
#define ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(DTYPE, ...)                  \
    [&] {                                                               \
        if (DTYPE == ::zt::ScalarType::Float) {                         \
            using scalar_t = float;                                     \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::Double) {                        \
            using scalar_t = double;                                    \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::Char) {                          \
            using scalar_t = std::int8_t;                               \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::Short) {                         \
            using scalar_t = std::int16_t;                              \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::Int) {                           \
            using scalar_t = std::int32_t;                              \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::Long) {                          \
            using scalar_t = std::int64_t;                              \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::Byte) {                          \
            using scalar_t = std::uint8_t;                              \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::Half) {                          \
            using scalar_t = ::zt::Half;                                \
            return __VA_ARGS__();                                       \
        }                                                               \
        if (DTYPE == ::zt::ScalarType::BFloat16) {                      \
            using scalar_t = ::zt::BFloat16;                            \
            return __VA_ARGS__();                                       \
        }                                                               \
        ZT_LOG_ERROR("Unsupported ScalarType {} for template dispatch", \
                     ::zt::toString(DTYPE));                            \
    }()

// Numeric types plus Bool.
#define ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(DTYPE, ...)       \
    [&] {                                                              \
        if (DTYPE == ::zt::ScalarType::Bool) {                         \
            using scalar_t = bool;                                     \
            return __VA_ARGS__();                                      \
        }                                                              \
        return ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(DTYPE, __VA_ARGS__); \
    }()

// Floating-point only (Float / Double).
#define ZT_DISPATCH_SCALARTYPE_FLOAT_ONLY(DTYPE, ...)       \
    [&] {                                                   \
        if (DTYPE == ::zt::ScalarType::Float) {             \
            using scalar_t = float;                         \
            return __VA_ARGS__();                           \
        }                                                   \
        if (DTYPE == ::zt::ScalarType::Double) {            \
            using scalar_t = double;                        \
            return __VA_ARGS__();                           \
        }                                                   \
        if (DTYPE == ::zt::ScalarType::Half) {              \
            using scalar_t = ::zt::Half;                    \
            return __VA_ARGS__();                           \
        }                                                   \
        if (DTYPE == ::zt::ScalarType::BFloat16) {          \
            using scalar_t = ::zt::BFloat16;                \
            return __VA_ARGS__();                           \
        }                                                   \
        ZT_LOG_ERROR("ScalarType {} is not floating-point", \
                     ::zt::toString(DTYPE));                \
    }()

// Integral only (Byte/Char/Short/Int/Long).
#define ZT_DISPATCH_SCALARTYPE_INT_ONLY(DTYPE, ...)                           \
    [&] {                                                                     \
        if (DTYPE == ::zt::ScalarType::Char) {                                \
            using scalar_t = std::int8_t;                                     \
            return __VA_ARGS__();                                             \
        }                                                                     \
        if (DTYPE == ::zt::ScalarType::Short) {                               \
            using scalar_t = std::int16_t;                                    \
            return __VA_ARGS__();                                             \
        }                                                                     \
        if (DTYPE == ::zt::ScalarType::Int) {                                 \
            using scalar_t = std::int32_t;                                    \
            return __VA_ARGS__();                                             \
        }                                                                     \
        if (DTYPE == ::zt::ScalarType::Long) {                                \
            using scalar_t = std::int64_t;                                    \
            return __VA_ARGS__();                                             \
        }                                                                     \
        if (DTYPE == ::zt::ScalarType::Byte) {                                \
            using scalar_t = std::uint8_t;                                    \
            return __VA_ARGS__();                                             \
        }                                                                     \
        ZT_LOG_ERROR("ScalarType {} is not integral", ::zt::toString(DTYPE)); \
    }()
