// ztensor/kernel/BinaryEWCPU.cpp
//
// CPU BinaryEW kernel. The op code resolves once to a captureless element
// functor (function pointer) before the loop, so the hot path is a plain
// indirect call. Dispatch shapes:
//   * arithmetic  (Add/Sub/Mul/Div + Pow/Fmod/...): ALL_SAME, write scalar_t.
//   * comparisons (Eq/Ne/.../Ge): INPUT_SAME_OUTPUT_BOOL, write bool.
//   * logical     (And/Or/Xor):     INPUT_SAME_OUTPUT_BOOL, write bool.
//   * bitwise     (And/Or/Xor/Lshift/Rshift): ALL_SAME, integral only
//     (rejected at runtime for floats; compile-time SFINAE avoids illegal
//      &/|/^/<< on float/double).

#include <cmath>
#include <cstdint>

#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/BinaryEW.h"

namespace zt::kernel {
namespace {

// ── Op category helpers ────────────────────────────────────────────────────

inline bool is_comparison(BinaryEWOpCode op) noexcept {
    return op >= BinaryEWOpCode::Eq && op <= BinaryEWOpCode::Ge;
}

inline bool is_logical(BinaryEWOpCode op) noexcept {
    return op >= BinaryEWOpCode::LogicalAnd &&
           op <= BinaryEWOpCode::LogicalXor;
}

inline bool is_bitwise(BinaryEWOpCode op) noexcept {
    return op >= BinaryEWOpCode::BitwiseAnd;
}

// ── Bitwise helper: only integral T gets real operators ────────────────────

template <typename T, bool Integral = std::is_integral_v<T>>
struct BitwiseOps {
    static T and_(T a, T b) { return static_cast<T>(a & b); }
    static T or_(T a, T b)  { return static_cast<T>(a | b); }
    static T xor_(T a, T b) { return static_cast<T>(a ^ b); }
    static T lshift(T a, T b) { return static_cast<T>(a << b); }
    static T rshift(T a, T b) { return static_cast<T>(a >> b); }
};

// Float / Half / BFloat16 — never called (rejected at runtime).
template <typename T>
struct BitwiseOps<T, false> {
    static T and_(T a, T) { return a; }
    static T or_(T a, T)  { return a; }
    static T xor_(T a, T) { return a; }
    static T lshift(T a, T) { return a; }
    static T rshift(T a, T) { return a; }
};

// ── Functor tables ─────────────────────────────────────────────────────────

template <typename T>
using ArithFn = T (*)(T, T);

template <typename T>
ArithFn<T> select_arith_fn(BinaryEWOpCode op) {
    switch (op) {
        case BinaryEWOpCode::Add:
            return +[](T a, T b) { return static_cast<T>(a + b); };
        case BinaryEWOpCode::Sub:
            return +[](T a, T b) { return static_cast<T>(a - b); };
        case BinaryEWOpCode::Mul:
            return +[](T a, T b) { return static_cast<T>(a * b); };
        case BinaryEWOpCode::Div:
            return +[](T a, T b) { return static_cast<T>(a / b); };
        case BinaryEWOpCode::Pow:
            return +[](T a, T b) {
                return static_cast<T>(
                    std::pow(static_cast<float>(a), static_cast<float>(b)));
            };
        case BinaryEWOpCode::Fmod:
            return +[](T a, T b) {
                return static_cast<T>(
                    std::fmod(static_cast<float>(a), static_cast<float>(b)));
            };
        case BinaryEWOpCode::Remainder:
            return +[](T a, T b) {
                return static_cast<T>(
                    std::remainder(static_cast<float>(a),
                                   static_cast<float>(b)));
            };
        case BinaryEWOpCode::Maximum:
            return +[](T a, T b) { return a > b ? a : b; };
        case BinaryEWOpCode::Minimum:
            return +[](T a, T b) { return a < b ? a : b; };
        case BinaryEWOpCode::Atan2:
            return +[](T a, T b) {
                return static_cast<T>(
                    std::atan2(static_cast<float>(a), static_cast<float>(b)));
            };
        case BinaryEWOpCode::Hypot:
            return +[](T a, T b) {
                return static_cast<T>(
                    std::hypot(static_cast<float>(a), static_cast<float>(b)));
            };
        case BinaryEWOpCode::BitwiseAnd:
            return +[](T a, T b) { return BitwiseOps<T>::and_(a, b); };
        case BinaryEWOpCode::BitwiseOr:
            return +[](T a, T b) { return BitwiseOps<T>::or_(a, b); };
        case BinaryEWOpCode::BitwiseXor:
            return +[](T a, T b) { return BitwiseOps<T>::xor_(a, b); };
        case BinaryEWOpCode::Lshift:
            return +[](T a, T b) { return BitwiseOps<T>::lshift(a, b); };
        case BinaryEWOpCode::Rshift:
            return +[](T a, T b) { return BitwiseOps<T>::rshift(a, b); };
        default:
            break;
    }
    ZT_LOG_ERROR("BinaryEWCPU: op code {} is not arithmetic/bitwise",
                 static_cast<int>(op));
}

template <typename T>
using BoolFn = bool (*)(T, T);

template <typename T>
BoolFn<T> select_cmp_fn(BinaryEWOpCode op) {
    switch (op) {
        case BinaryEWOpCode::Eq:
            return +[](T a, T b) { return a == b; };
        case BinaryEWOpCode::Ne:
            return +[](T a, T b) { return a != b; };
        case BinaryEWOpCode::Lt:
            return +[](T a, T b) { return a < b; };
        case BinaryEWOpCode::Le:
            return +[](T a, T b) { return a <= b; };
        case BinaryEWOpCode::Gt:
            return +[](T a, T b) { return a > b; };
        case BinaryEWOpCode::Ge:
            return +[](T a, T b) { return a >= b; };
        default:
            break;
    }
    ZT_LOG_ERROR("BinaryEWCPU: op code {} is not a comparison",
                 static_cast<int>(op));
}

template <typename T>
BoolFn<T> select_logical_fn(BinaryEWOpCode op) {
    switch (op) {
        case BinaryEWOpCode::LogicalAnd:
            return +[](T a, T b) { return a && b; };
        case BinaryEWOpCode::LogicalOr:
            return +[](T a, T b) { return a || b; };
        case BinaryEWOpCode::LogicalXor:
            return +[](T a, T b) {
                return static_cast<bool>(a) != static_cast<bool>(b);
            };
        default:
            break;
    }
    ZT_LOG_ERROR("BinaryEWCPU: op code {} is not a logical op",
                 static_cast<int>(op));
}

}  // namespace

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 const Tensor& dst,
                 BinaryEWOpCode op) {
    // ── Logical ops: INPUT_SAME_OUTPUT_BOOL ────────────────────────────
    if (is_logical(op)) {
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(lhs.scalar_type(), [&] {
            const auto f = select_logical_fn<scalar_t>(op);
            core::Indexer indexer(
                {lhs, rhs}, dst, core::DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
            core::ParallelFor(
                dst.device(), indexer.NumWorkloads(), [&](int64_t i) {
                    *indexer.GetOutputPtr<bool>(i) =
                        f(*indexer.GetInputPtr<scalar_t>(0, i),
                          *indexer.GetInputPtr<scalar_t>(1, i));
                });
        });
        return;
    }

    // ── Comparison ops: INPUT_SAME_OUTPUT_BOOL ────────────────────────
    if (is_comparison(op)) {
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(lhs.scalar_type(), [&] {
            const auto f = select_cmp_fn<scalar_t>(op);
            core::Indexer indexer(
                {lhs, rhs}, dst, core::DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
            core::ParallelFor(
                dst.device(), indexer.NumWorkloads(), [&](int64_t i) {
                    *indexer.GetOutputPtr<bool>(i) =
                        f(*indexer.GetInputPtr<scalar_t>(0, i),
                          *indexer.GetInputPtr<scalar_t>(1, i));
                });
        });
        return;
    }

    // ── Bitwise ops: ALL_SAME, integral only ──────────────────────────
    if (is_bitwise(op)) {
        if (zt::isFloatingType(lhs.scalar_type())) {
            ZT_LOG_ERROR(
                "BinaryEWCPU: bitwise op {} requires integral dtype, got {}",
                static_cast<int>(op),
                zt::toString(lhs.scalar_type()));
        }
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(lhs.scalar_type(), [&] {
            const auto f = select_arith_fn<scalar_t>(op);
            core::Indexer indexer({lhs, rhs}, dst, core::DtypePolicy::ALL_SAME);
            core::ParallelFor(
                dst.device(), indexer.NumWorkloads(), [&](int64_t i) {
                    *indexer.GetOutputPtr<scalar_t>(i) =
                        f(*indexer.GetInputPtr<scalar_t>(0, i),
                          *indexer.GetInputPtr<scalar_t>(1, i));
                });
        });
        return;
    }

    // ── Arithmetic (including extended: Pow/Fmod/...): ALL_SAME ──────
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(lhs.scalar_type(), [&] {
        const auto f = select_arith_fn<scalar_t>(op);
        core::Indexer indexer({lhs, rhs}, dst, core::DtypePolicy::ALL_SAME);
        core::ParallelFor(
            dst.device(), indexer.NumWorkloads(), [&](int64_t i) {
                *indexer.GetOutputPtr<scalar_t>(i) =
                    f(*indexer.GetInputPtr<scalar_t>(0, i),
                      *indexer.GetInputPtr<scalar_t>(1, i));
            });
    });
}

}  // namespace zt::kernel
