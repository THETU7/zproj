// ztensor/kernel/BinaryEWCUDA.cu
//
// CUDA BinaryEW kernel. Mirrors BinaryEWCPU.cpp's dispatch shapes:
//   * arithmetic  (Add/Sub/Mul/Div + Pow/Fmod/...): ALL_SAME
//   * comparisons (Eq/Ne/.../Ge): INPUT_SAME_OUTPUT_BOOL
//   * logical     (And/Or/Xor):     INPUT_SAME_OUTPUT_BOOL
//   * bitwise     (And/Or/Xor/Lshift/Rshift): ALL_SAME, integral only
// The per-element math runs in a __device__ lambda over an Indexer +
// ParallelFor (grid-stride).

#include <cuda_runtime.h>

#include "ztensor/zt/utility/Log.h"

#include "ztensor/zt/cuda/Guard.h"
#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/BinaryEW.h"

namespace zt {
namespace kernel {
namespace {

// ── Op category helpers ────────────────────────────────────────────────────

__host__ __device__ inline bool is_comparison(BinaryEWOpCode op) noexcept {
    return op >= BinaryEWOpCode::Eq && op <= BinaryEWOpCode::Ge;
}

__host__ __device__ inline bool is_logical(BinaryEWOpCode op) noexcept {
    return op >= BinaryEWOpCode::LogicalAnd &&
           op <= BinaryEWOpCode::LogicalXor;
}

__host__ __device__ inline bool is_bitwise(BinaryEWOpCode op) noexcept {
    return op >= BinaryEWOpCode::BitwiseAnd;
}

// ── Per-element __device__ functors ────────────────────────────────────────

template <typename T>
__device__ T arith_element(BinaryEWOpCode op, T a, T b) {
    switch (op) {
        case BinaryEWOpCode::Add:
            return static_cast<T>(a + b);
        case BinaryEWOpCode::Sub:
            return static_cast<T>(a - b);
        case BinaryEWOpCode::Mul:
            return static_cast<T>(a * b);
        case BinaryEWOpCode::Div:
            return static_cast<T>(a / b);
        case BinaryEWOpCode::Pow:
            return static_cast<T>(
                ::powf(static_cast<float>(a), static_cast<float>(b)));
        case BinaryEWOpCode::Fmod:
            return static_cast<T>(
                ::fmodf(static_cast<float>(a), static_cast<float>(b)));
        case BinaryEWOpCode::Remainder:
            return static_cast<T>(
                ::remainderf(static_cast<float>(a), static_cast<float>(b)));
        case BinaryEWOpCode::Maximum:
            return a > b ? a : b;
        case BinaryEWOpCode::Minimum:
            return a < b ? a : b;
        case BinaryEWOpCode::Atan2:
            return static_cast<T>(
                ::atan2f(static_cast<float>(a), static_cast<float>(b)));
        case BinaryEWOpCode::Hypot:
            return static_cast<T>(
                ::hypotf(static_cast<float>(a), static_cast<float>(b)));
        default:
            return static_cast<T>(0);
    }
}

template <typename T>
__device__ T bitwise_element(BinaryEWOpCode op, T a, T b) {
    switch (op) {
        case BinaryEWOpCode::BitwiseAnd:
            return static_cast<T>(static_cast<int64_t>(a) & static_cast<int64_t>(b));
        case BinaryEWOpCode::BitwiseOr:
            return static_cast<T>(static_cast<int64_t>(a) | static_cast<int64_t>(b));
        case BinaryEWOpCode::BitwiseXor:
            return static_cast<T>(static_cast<int64_t>(a) ^ static_cast<int64_t>(b));
        case BinaryEWOpCode::Lshift:
            return static_cast<T>(static_cast<int64_t>(a) << static_cast<int64_t>(b));
        case BinaryEWOpCode::Rshift:
            return static_cast<T>(static_cast<int64_t>(a) >> static_cast<int64_t>(b));
        default:
            return static_cast<T>(0);
    }
}

template <typename T>
__device__ bool cmp_element(BinaryEWOpCode op, T a, T b) {
    switch (op) {
        case BinaryEWOpCode::Eq:
            return a == b;
        case BinaryEWOpCode::Ne:
            return a != b;
        case BinaryEWOpCode::Lt:
            return a < b;
        case BinaryEWOpCode::Le:
            return a <= b;
        case BinaryEWOpCode::Gt:
            return a > b;
        case BinaryEWOpCode::Ge:
            return a >= b;
        default:
            return false;
    }
}

template <typename T>
__device__ bool logical_element(BinaryEWOpCode op, T a, T b) {
    switch (op) {
        case BinaryEWOpCode::LogicalAnd:
            return a && b;
        case BinaryEWOpCode::LogicalOr:
            return a || b;
        case BinaryEWOpCode::LogicalXor:
            return static_cast<bool>(a) != static_cast<bool>(b);
        default:
            return false;
    }
}

}  // namespace

void BinaryEWCUDA(const Tensor& lhs,
                  const Tensor& rhs,
                  const Tensor& dst,
                  BinaryEWOpCode op) {
    CUDAScopedDevice scoped(dst.device());

    // ── Logical ops ───────────────────────────────────────────────────
    if (is_logical(op)) {
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(lhs.scalar_type(), [&] {
            core::Indexer indexer(
                {lhs, rhs}, dst, core::DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
            core::ParallelFor(dst.device(),
                              indexer.NumWorkloads(),
                              [=] __device__(int64_t i) {
                                  *indexer.GetOutputPtr<bool>(i) =
                                      logical_element<scalar_t>(
                                          op,
                                          *indexer.GetInputPtr<scalar_t>(0, i),
                                          *indexer.GetInputPtr<scalar_t>(1, i));
                              });
        });
        return;
    }

    // ── Comparison ops ────────────────────────────────────────────────
    if (is_comparison(op)) {
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(lhs.scalar_type(), [&] {
            core::Indexer indexer(
                {lhs, rhs}, dst, core::DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
            core::ParallelFor(dst.device(),
                              indexer.NumWorkloads(),
                              [=] __device__(int64_t i) {
                                  *indexer.GetOutputPtr<bool>(i) =
                                      cmp_element<scalar_t>(
                                          op,
                                          *indexer.GetInputPtr<scalar_t>(0, i),
                                          *indexer.GetInputPtr<scalar_t>(1, i));
                              });
        });
        return;
    }

    // ── Bitwise ops (integral only) ───────────────────────────────────
    if (is_bitwise(op)) {
        if (zt::isFloatingType(lhs.scalar_type())) {
            ZT_LOG_ERROR(
                "BinaryEWCUDA: bitwise op {} requires integral dtype, got {}",
                static_cast<int>(op),
                zt::toString(lhs.scalar_type()));
        }
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(lhs.scalar_type(), [&] {
            core::Indexer indexer({lhs, rhs}, dst, core::DtypePolicy::ALL_SAME);
            core::ParallelFor(dst.device(),
                              indexer.NumWorkloads(),
                              [=] __device__(int64_t i) {
                                  *indexer.GetOutputPtr<scalar_t>(i) =
                                      bitwise_element<scalar_t>(
                                          op,
                                          *indexer.GetInputPtr<scalar_t>(0, i),
                                          *indexer.GetInputPtr<scalar_t>(1, i));
                              });
        });
        return;
    }

    // ── Arithmetic (including extended) ───────────────────────────────
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(lhs.scalar_type(), [&] {
        core::Indexer indexer({lhs, rhs}, dst, core::DtypePolicy::ALL_SAME);
        core::ParallelFor(
            dst.device(), indexer.NumWorkloads(), [=] __device__(int64_t i) {
                *indexer.GetOutputPtr<scalar_t>(i) = arith_element<scalar_t>(
                    op,
                    *indexer.GetInputPtr<scalar_t>(0, i),
                    *indexer.GetInputPtr<scalar_t>(1, i));
            });
    });
}

}  // namespace kernel
}  // namespace zt
