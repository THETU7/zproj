// ztensor/kernel/UnaryEWCUDA.cu
//
// CUDA UnaryEW kernel: resolve the op code to a __device__ element functor
// switch once per element, then walk src/dst with an Indexer + ParallelFor
// (grid-stride). Mirrors UnaryEWCPU.cpp.
//
// Float-only ops reject integral dtypes at runtime, matching CPU semantics.
//
// Transcendental functors call the overloaded `::FN(a)` (no `f` suffix). For
// T=double this selects the double-precision device runtime function
// (preserving full precision); for T=float/Half/BFloat16 the argument
// converts to float (Half/BFloat16 have only `operator float()`) and selects
// the single-precision overload. Routing `double` through `::sqrtf` would
// silently drop ~9 significant digits.

#include "ztensor/zt/cuda/Guard.h"
#include "ztensor/zt/cuda/Vendor.h"
#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/UnaryEW.h"

namespace zt {
namespace kernel {
namespace {

__host__ __device__ inline bool is_float_only_op(UnaryEWOpCode op) noexcept {
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

// ── Per-element __device__ functor ─────────────────────────────────────────

// Separate the ~ case so it only compiles for integral types.
template<typename T, bool Integral = std::is_integral_v<T>>
struct DeviceUnary;

template<typename T>
struct DeviceUnary<T, true> {
    static __device__ T bitwise_not(T a) {
        // PyTorch semantics: `~True == False` (logical NOT) for bool,
        // bitwise `~` for other integrals.
        if constexpr (std::is_same_v<T, bool>) {
            return static_cast<T>(!a);
        } else {
            return static_cast<T>(~a);
        }
    }
};

template<typename T>
struct DeviceUnary<T, false> {
    static __device__ T bitwise_not(T a) { return a; }  // never called
};

template<typename T>
__device__ T unary_element(UnaryEWOpCode op, T a) {
    switch (op) {
        case UnaryEWOpCode::Neg:
            return -a;
        case UnaryEWOpCode::Abs:
            return a < T{} ? -a : a;
        case UnaryEWOpCode::Sqrt:
            return static_cast<T>(::sqrt(a));
        case UnaryEWOpCode::Rsqrt:
            return static_cast<T>(T{1} / static_cast<T>(::sqrt(a)));
        case UnaryEWOpCode::Exp:
            return static_cast<T>(::exp(a));
        case UnaryEWOpCode::Expm1:
            return static_cast<T>(::expm1(a));
        case UnaryEWOpCode::Log:
            return static_cast<T>(::log(a));
        case UnaryEWOpCode::Log2:
            return static_cast<T>(::log2(a));
        case UnaryEWOpCode::Log10:
            return static_cast<T>(::log10(a));
        case UnaryEWOpCode::Log1p:
            return static_cast<T>(::log1p(a));
        case UnaryEWOpCode::Reciprocal:
            return static_cast<T>(1) / a;
        case UnaryEWOpCode::Sigmoid:
            return static_cast<T>(
                static_cast<T>(1) /
                (static_cast<T>(1) + static_cast<T>(::exp(-a))));
        case UnaryEWOpCode::Frac:
            return static_cast<T>(a - static_cast<T>(::floor(a)));
        case UnaryEWOpCode::Sin:
            return static_cast<T>(::sin(a));
        case UnaryEWOpCode::Cos:
            return static_cast<T>(::cos(a));
        case UnaryEWOpCode::Tan:
            return static_cast<T>(::tan(a));
        case UnaryEWOpCode::Asin:
            return static_cast<T>(::asin(a));
        case UnaryEWOpCode::Acos:
            return static_cast<T>(::acos(a));
        case UnaryEWOpCode::Atan:
            return static_cast<T>(::atan(a));
        case UnaryEWOpCode::Sinh:
            return static_cast<T>(::sinh(a));
        case UnaryEWOpCode::Cosh:
            return static_cast<T>(::cosh(a));
        case UnaryEWOpCode::Tanh:
            return static_cast<T>(::tanh(a));
        case UnaryEWOpCode::Floor:
            return static_cast<T>(::floor(a));
        case UnaryEWOpCode::Ceil:
            return static_cast<T>(::ceil(a));
        case UnaryEWOpCode::Round:
            return static_cast<T>(::round(a));
        case UnaryEWOpCode::Trunc:
            return static_cast<T>(::trunc(a));
        case UnaryEWOpCode::Sign:
            return static_cast<T>((T{} < a) - (a < T{}));
        case UnaryEWOpCode::LogicalNot:
            return static_cast<T>(!a);
        case UnaryEWOpCode::BitwiseNot:
            return DeviceUnary<T>::bitwise_not(a);
    }
    return static_cast<T>(0);
}

}  // namespace

void UnaryEWCUDA(const Tensor& src, const Tensor& dst, UnaryEWOpCode op) {
    // Reject float-only ops on integral dtypes at the host level.
    if (is_float_only_op(op) && zt::isIntegralType(src.scalar_type())) {
        ZT_LOG_ERROR(
            "UnaryEWCUDA: op {} requires a floating-point dtype, got {}",
            static_cast<int>(op),
            zt::toString(src.scalar_type()));
    }
    if (op == UnaryEWOpCode::BitwiseNot &&
        zt::isFloatingType(src.scalar_type())) {
        ZT_LOG_ERROR(
            "UnaryEWCUDA: bitwise_not requires an integral dtype, got {}",
            zt::toString(src.scalar_type()));
    }

    CUDAScopedDevice scoped(dst.device());
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        core::Indexer indexer({src}, dst, core::DtypePolicy::ALL_SAME);
        core::ParallelFor(
            dst.device(), indexer.NumWorkloads(), [=] __device__(int64_t i) {
                *indexer.GetOutputPtr<scalar_t>(i) = unary_element<scalar_t>(
                    op, *indexer.GetInputPtr<scalar_t>(0, i));
            });
    });
}

}  // namespace kernel
}  // namespace zt
