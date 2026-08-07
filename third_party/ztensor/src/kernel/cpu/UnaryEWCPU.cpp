// ztensor/kernel/UnaryEWCPU.cpp
//
// CPU UnaryEW kernel: resolve the op code to a captureless element functor
// once (function pointer), then walk src/dst with an Indexer + ParallelFor.
// src and dst share shape & dtype (DtypePolicy::ALL_SAME).
//
// Float-only ops (sqrt/exp/log/sin/...) are rejected for integral dtypes at
// runtime via is_float_only_op(). The corresponding functor bodies still
// compile for integral scalars (using an implicit float cast path), but they
// are never invoked on an integral-dtype tensor.
//
// Transcendental functors call <cmath> directly on the operand and let
// overload resolution pick the right precision: `T=double` → std::sqrt(double)
// (full double precision), `T=float`/`Half`/`BFloat16` → std::sqrt(float)
// (Half/BFloat16 have only `operator float()`, so they round through float at
// their native precision). Never route `double` math through `float` — that
// silently drops ~9 significant digits.

#include <cmath>

#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/UnaryEW.h"

namespace zt::kernel {
namespace {

// ── Helpers ────────────────────────────────────────────────────────────────

inline bool is_float_only_op(UnaryEWOpCode op) noexcept {
    switch (op) {
        case UnaryEWOpCode::Sqrt:
        case UnaryEWOpCode::Rsqrt:
        case UnaryEWOpCode::Exp:
        case UnaryEWOpCode::Expm1:
        case UnaryEWOpCode::Log:
        case UnaryEWOpCode::Log2:
        case UnaryEWOpCode::Log10:
        case UnaryEWOpCode::Log1p:
        case UnaryEWOpCode::Sin:
        case UnaryEWOpCode::Cos:
        case UnaryEWOpCode::Tan:
        case UnaryEWOpCode::Asin:
        case UnaryEWOpCode::Acos:
        case UnaryEWOpCode::Atan:
        case UnaryEWOpCode::Sinh:
        case UnaryEWOpCode::Cosh:
        case UnaryEWOpCode::Tanh:
        case UnaryEWOpCode::Floor:
        case UnaryEWOpCode::Ceil:
        case UnaryEWOpCode::Round:
        case UnaryEWOpCode::Trunc:
        case UnaryEWOpCode::Sigmoid:
        case UnaryEWOpCode::Frac:
            return true;
        default:
            return false;
    }
}

template<typename T>
using UnaryFn = T (*)(T);

// ── Per-op functors (every op has a definition for every T) ────────────

template<typename T>
T fn_neg(T a) {
    return -a;
}

template<typename T>
T fn_abs(T a) {
    return a < T{} ? -a : a;
}

template<typename T>
T fn_sign(T a) {
    return static_cast<T>((T{} < a) - (a < T{}));
}

template<typename T>
T fn_logical_not(T a) {
    return static_cast<T>(!a);
}

// `~bool` would be a logical NOT in PyTorch semantics (`~True == False`), but
// `~integral` is bitwise. Key on `IsBool` so `T=bool` returns `!a` (the only
// sane Boolean "bitwise" NOT) while other integrals apply the real `~` op.
// The float specialization below rejects non-integral T at runtime.
template<typename T,
         bool Integral = std::is_integral_v<T>,
         bool IsBool = std::is_same_v<T, bool>>
struct BitwiseHelper {
    static T not_(T a) {
        if constexpr (IsBool) {
            return static_cast<T>(!a);
        } else {
            return static_cast<T>(~a);
        }
    }
};
template<typename T>
struct BitwiseHelper<T, false, false> {
    static T not_(T a) { return a; }  // never called; rejected at runtime
};

template<typename T>
T fn_bitwise_not(T a) {
    return BitwiseHelper<T>::not_(a);
}

template<typename T>
T fn_reciprocal(T a) {
    return static_cast<T>(1) / a;
}

// ── <cmath> wrappers ────────────────────────────────────────────────────
// Call std::FN(a) directly; overload resolution selects double for T=double
// (full precision) and float for T=float/Half/BFloat16 (Half/BFloat16
// promote via their `operator float()`).
template<typename T>
T fn_sqrt(T a) {
    return static_cast<T>(std::sqrt(a));
}
template<typename T>
T fn_rsqrt(T a) {
    return static_cast<T>(static_cast<T>(1) / static_cast<T>(std::sqrt(a)));
}
template<typename T>
T fn_exp(T a) {
    return static_cast<T>(std::exp(a));
}
template<typename T>
T fn_expm1(T a) {
    return static_cast<T>(std::expm1(a));
}
template<typename T>
T fn_log(T a) {
    return static_cast<T>(std::log(a));
}
template<typename T>
T fn_log2(T a) {
    return static_cast<T>(std::log2(a));
}
template<typename T>
T fn_log10(T a) {
    return static_cast<T>(std::log10(a));
}
template<typename T>
T fn_log1p(T a) {
    return static_cast<T>(std::log1p(a));
}
template<typename T>
T fn_sigmoid(T a) {
    return static_cast<T>(static_cast<T>(1) /
                          (static_cast<T>(1) + static_cast<T>(std::exp(-a))));
}
template<typename T>
T fn_frac(T a) {
    return static_cast<T>(a - static_cast<T>(std::floor(a)));
}
template<typename T>
T fn_sin(T a) {
    return static_cast<T>(std::sin(a));
}
template<typename T>
T fn_cos(T a) {
    return static_cast<T>(std::cos(a));
}
template<typename T>
T fn_tan(T a) {
    return static_cast<T>(std::tan(a));
}
template<typename T>
T fn_asin(T a) {
    return static_cast<T>(std::asin(a));
}
template<typename T>
T fn_acos(T a) {
    return static_cast<T>(std::acos(a));
}
template<typename T>
T fn_atan(T a) {
    return static_cast<T>(std::atan(a));
}
template<typename T>
T fn_sinh(T a) {
    return static_cast<T>(std::sinh(a));
}
template<typename T>
T fn_cosh(T a) {
    return static_cast<T>(std::cosh(a));
}
template<typename T>
T fn_tanh(T a) {
    return static_cast<T>(std::tanh(a));
}
template<typename T>
T fn_floor_(T a) {
    return static_cast<T>(std::floor(a));
}
template<typename T>
T fn_ceil(T a) {
    return static_cast<T>(std::ceil(a));
}
template<typename T>
T fn_round_(T a) {
    return static_cast<T>(std::round(a));
}
template<typename T>
T fn_trunc(T a) {
    return static_cast<T>(std::trunc(a));
}

// ── Unified dispatch ───────────────────────────────────────────────────────

template<typename T>
UnaryFn<T> select_unary_fn(UnaryEWOpCode op) {
    switch (op) {
        case UnaryEWOpCode::Neg:
            return fn_neg<T>;
        case UnaryEWOpCode::Abs:
            return fn_abs<T>;
        case UnaryEWOpCode::Sqrt:
            return fn_sqrt<T>;
        case UnaryEWOpCode::Rsqrt:
            return fn_rsqrt<T>;
        case UnaryEWOpCode::Exp:
            return fn_exp<T>;
        case UnaryEWOpCode::Expm1:
            return fn_expm1<T>;
        case UnaryEWOpCode::Log:
            return fn_log<T>;
        case UnaryEWOpCode::Log2:
            return fn_log2<T>;
        case UnaryEWOpCode::Log10:
            return fn_log10<T>;
        case UnaryEWOpCode::Log1p:
            return fn_log1p<T>;
        case UnaryEWOpCode::Reciprocal:
            return fn_reciprocal<T>;
        case UnaryEWOpCode::Sigmoid:
            return fn_sigmoid<T>;
        case UnaryEWOpCode::Frac:
            return fn_frac<T>;
        case UnaryEWOpCode::Sin:
            return fn_sin<T>;
        case UnaryEWOpCode::Cos:
            return fn_cos<T>;
        case UnaryEWOpCode::Tan:
            return fn_tan<T>;
        case UnaryEWOpCode::Asin:
            return fn_asin<T>;
        case UnaryEWOpCode::Acos:
            return fn_acos<T>;
        case UnaryEWOpCode::Atan:
            return fn_atan<T>;
        case UnaryEWOpCode::Sinh:
            return fn_sinh<T>;
        case UnaryEWOpCode::Cosh:
            return fn_cosh<T>;
        case UnaryEWOpCode::Tanh:
            return fn_tanh<T>;
        case UnaryEWOpCode::Floor:
            return fn_floor_<T>;
        case UnaryEWOpCode::Ceil:
            return fn_ceil<T>;
        case UnaryEWOpCode::Round:
            return fn_round_<T>;
        case UnaryEWOpCode::Trunc:
            return fn_trunc<T>;
        case UnaryEWOpCode::Sign:
            return fn_sign<T>;
        case UnaryEWOpCode::LogicalNot:
            return fn_logical_not<T>;
        case UnaryEWOpCode::BitwiseNot:
            return fn_bitwise_not<T>;
    }
    ZT_LOG_ERROR("UnaryEWCPU: unsupported op code {}", static_cast<int>(op));
}

}  // namespace

void UnaryEWCPU(const Tensor& src, const Tensor& dst, UnaryEWOpCode op) {
    // Reject float-only ops on integral dtypes (they'd compile but produce
    // nonsense — PyTorch semantics).
    if (is_float_only_op(op) && zt::isIntegralType(src.scalar_type())) {
        ZT_LOG_ERROR(
            "UnaryEWCPU: op {} requires a floating-point dtype, got {}",
            static_cast<int>(op),
            zt::toString(src.scalar_type()));
    }
    // Reject bitwise_not on float dtypes.
    if (op == UnaryEWOpCode::BitwiseNot &&
        zt::isFloatingType(src.scalar_type())) {
        ZT_LOG_ERROR(
            "UnaryEWCPU: bitwise_not requires an integral dtype, got {}",
            zt::toString(src.scalar_type()));
    }

    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        const auto f = select_unary_fn<scalar_t>(op);
        core::Indexer indexer({src}, dst, core::DtypePolicy::ALL_SAME);
        core::ParallelFor(dst.device(), indexer.NumWorkloads(), [&](int64_t i) {
            *indexer.GetOutputPtr<scalar_t>(i) =
                f(*indexer.GetInputPtr<scalar_t>(0, i));
        });
    });
}

}  // namespace zt::kernel
