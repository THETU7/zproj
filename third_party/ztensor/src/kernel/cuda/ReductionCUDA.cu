// ztensor/kernel/ReductionCUDA.cu
//
// CUDA Reduction kernel. Reductions aggregate many source workloads into one
// output slot, so a naive parallel scatter races. This implementation uses two
// correct strategies:
//
//   * atomic path (Sum/Mean/Min/Max over float/double/int32/int64): every
//     source workload atomically combines into its (identity-seeded) output
//     slot. atomicAdd covers Sum/Mean; a CAS loop covers Min/Max.
//   * serial path (every other dtype, e.g. 8/16-bit ints and bool): a single
//     thread walks every workload in order and folds into the seeded output,
//     reproducing the CPU scatter-reduce exactly. Slow, but correct and rare.
//
// Mean is Sum followed by a per-output scale (matches the CPU path). This is a
// correctness-first implementation; a PyTorch-style ReduceConfig
// (thread -> warp -> block -> global) is the documented future optimization.

#include <cstdint>
#include <limits>

#include "ztensor/zt/utility/Log.h"

#include "ztensor/zt/cuda/Guard.h"
#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/Fill.h"
#include "kernel/Reduction.h"

namespace zt {
namespace kernel {
namespace {

// True iff the atomic path supports (dtype, op).
bool atomic_supported(ScalarType dt) {
    return dt == ScalarType::Float || dt == ScalarType::Double ||
           dt == ScalarType::Int || dt == ScalarType::Long;
}

// ── atomic add (32/64-bit int, float, double) ────────────────────────────────
__device__ inline void atomic_add(std::int32_t* addr, std::int32_t v) {
    atomicAdd(addr, v);
}
__device__ inline void atomic_add(std::int64_t* addr, std::int64_t v) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(v));
}
__device__ inline void atomic_add(float* addr, float v) { atomicAdd(addr, v); }
__device__ inline void atomic_add(double* addr, double v) {
    atomicAdd(addr, v);
}

// ── atomic min / max via CAS loop over a 32- or 64-bit slot ──────────────────
// The CAS reads/writes the raw bits, so it works for signed types without a
// reinterpret dance beyond the pointer width.
__device__ inline void atomic_min(std::int32_t* addr, std::int32_t v) {
    int* p = reinterpret_cast<int*>(addr);
    int assumed;
    int old = *p;
    do {
        assumed = old;
        const int cur = *reinterpret_cast<const std::int32_t*>(&assumed);
        if (!(v < cur)) return;
        old = atomicCAS(p, assumed, *reinterpret_cast<const int*>(&v));
    } while (assumed != old);
}
__device__ inline void atomic_max(std::int32_t* addr, std::int32_t v) {
    int* p = reinterpret_cast<int*>(addr);
    int assumed;
    int old = *p;
    do {
        assumed = old;
        const int cur = *reinterpret_cast<const std::int32_t*>(&assumed);
        if (!(v > cur)) return;
        old = atomicCAS(p, assumed, *reinterpret_cast<const int*>(&v));
    } while (assumed != old);
}
__device__ inline void atomic_min(std::int64_t* addr, std::int64_t v) {
    unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *p;
    do {
        assumed = old;
        const std::int64_t cur =
            *reinterpret_cast<const std::int64_t*>(&assumed);
        if (!(v < cur)) return;
        old = atomicCAS(
            p, assumed, *reinterpret_cast<const unsigned long long*>(&v));
    } while (assumed != old);
}
__device__ inline void atomic_max(std::int64_t* addr, std::int64_t v) {
    unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *p;
    do {
        assumed = old;
        const std::int64_t cur =
            *reinterpret_cast<const std::int64_t*>(&assumed);
        if (!(v > cur)) return;
        old = atomicCAS(
            p, assumed, *reinterpret_cast<const unsigned long long*>(&v));
    } while (assumed != old);
}
__device__ inline void atomic_min(float* addr, float v) {
    int* p = reinterpret_cast<int*>(addr);
    int assumed;
    int old = *p;
    do {
        assumed = old;
        float cur = *reinterpret_cast<const float*>(&assumed);
        if (!(v < cur)) return;
        old = atomicCAS(p, assumed, *reinterpret_cast<const int*>(&v));
    } while (assumed != old);
}
__device__ inline void atomic_max(float* addr, float v) {
    int* p = reinterpret_cast<int*>(addr);
    int assumed;
    int old = *p;
    do {
        assumed = old;
        float cur = *reinterpret_cast<const float*>(&assumed);
        if (!(v > cur)) return;
        old = atomicCAS(p, assumed, *reinterpret_cast<const int*>(&v));
    } while (assumed != old);
}
__device__ inline void atomic_min(double* addr, double v) {
    unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *p;
    do {
        assumed = old;
        double cur = *reinterpret_cast<const double*>(&assumed);
        if (!(v < cur)) return;
        old = atomicCAS(
            p, assumed, *reinterpret_cast<const unsigned long long*>(&v));
    } while (assumed != old);
}
__device__ inline void atomic_max(double* addr, double v) {
    unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *p;
    do {
        assumed = old;
        double cur = *reinterpret_cast<const double*>(&assumed);
        if (!(v > cur)) return;
        old = atomicCAS(
            p, assumed, *reinterpret_cast<const unsigned long long*>(&v));
    } while (assumed != old);
}

// ── atomic multiply via CAS loop ─────────────────────────────────────────
__device__ inline void atomic_mul(float* addr, float v) {
    int* p = reinterpret_cast<int*>(addr);
    int assumed;
    int old = *p;
    do {
        assumed = old;
        float cur = *reinterpret_cast<const float*>(&assumed);
        float desired = cur * v;
        old = atomicCAS(p, assumed, *reinterpret_cast<const int*>(&desired));
    } while (assumed != old);
}
__device__ inline void atomic_mul(double* addr, double v) {
    unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *p;
    do {
        assumed = old;
        double cur = *reinterpret_cast<const double*>(&assumed);
        double desired = cur * v;
        old = atomicCAS(p, assumed,
                        *reinterpret_cast<const unsigned long long*>(&desired));
    } while (assumed != old);
}
__device__ inline void atomic_mul(std::int32_t* addr, std::int32_t v) {
    int* p = reinterpret_cast<int*>(addr);
    int assumed;
    int old = *p;
    do {
        assumed = old;
        const int cur = *reinterpret_cast<const std::int32_t*>(&assumed);
        const int desired = cur * v;
        old = atomicCAS(p, assumed, *reinterpret_cast<const int*>(&desired));
    } while (assumed != old);
}
__device__ inline void atomic_mul(std::int64_t* addr, std::int64_t v) {
    unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *p;
    do {
        assumed = old;
        const std::int64_t cur =
            *reinterpret_cast<const std::int64_t*>(&assumed);
        const std::int64_t desired = cur * v;
        old = atomicCAS(p, assumed,
                        *reinterpret_cast<const unsigned long long*>(&desired));
    } while (assumed != old);
}

// NaN detection helper: integral types are never NaN.
template <typename T>
__host__ __device__ constexpr bool is_nan(T /*v*/) {
    return false;
}
template <>
__host__ __device__ inline bool is_nan<float>(float v) {
    return isnan(v);
}
template <>
__host__ __device__ inline bool is_nan<double>(double v) {
    return isnan(v);
}
#ifdef __CUDACC__
template <>
__host__ __device__ inline bool is_nan<__half>(__half v) {
    return __hisnan(v);
}
template <>
__host__ __device__ inline bool is_nan<__nv_bfloat16>(__nv_bfloat16 v) {
    return __hisnan(v);
}
#endif  // __CUDACC__

// Combine one source value into the accumulator at `addr` per `op`.
template <typename T>
__device__ inline void combine_atomic(ReductionOpCode op, T* addr, T v) {
    switch (op) {
        case ReductionOpCode::Sum:
        case ReductionOpCode::Mean:
            atomic_add(addr, v);
            return;
        case ReductionOpCode::Min:
            atomic_min(addr, v);
            return;
        case ReductionOpCode::Max:
            atomic_max(addr, v);
            return;
        case ReductionOpCode::Prod:
            atomic_mul(addr, v);
            return;
        case ReductionOpCode::NanMin:
            if (!is_nan(v)) { atomic_min(addr, v); }
            return;
        case ReductionOpCode::NanMax:
            if (!is_nan(v)) { atomic_max(addr, v); }
            return;
        default:
            return;  // All/Any reach serial path only (bool not atomic).
    }
}

// Identity element for `op` (matches ReductionCPU.cpp). Callable from host
// (entry-point seeding) and device (serial fallback).
template <typename T>
__host__ __device__ T identity_value(ReductionOpCode op) {
    switch (op) {
        case ReductionOpCode::Sum:
        case ReductionOpCode::Mean:
            return static_cast<T>(0);
        case ReductionOpCode::Min:
        case ReductionOpCode::NanMin:
            return std::numeric_limits<T>::max();
        case ReductionOpCode::Max:
        case ReductionOpCode::NanMax:
            return std::numeric_limits<T>::lowest();
        case ReductionOpCode::Prod:
            return static_cast<T>(1);
        case ReductionOpCode::All:
            return static_cast<T>(true);
        case ReductionOpCode::Any:
            return static_cast<T>(false);
    }
    return static_cast<T>(0);  // unreachable for a valid op
}

// Fold one source value `v` into `acc` (used by the serial fallback path).
template <typename T>
__host__ __device__ T combine_value(ReductionOpCode op, T acc, T v) {
    switch (op) {
        case ReductionOpCode::Sum:
        case ReductionOpCode::Mean:
            return static_cast<T>(acc + v);
        case ReductionOpCode::Min:
            return v < acc ? v : acc;
        case ReductionOpCode::Max:
            return v > acc ? v : acc;
        case ReductionOpCode::Prod:
            return static_cast<T>(acc * v);
        case ReductionOpCode::NanMin:
            if (is_nan(v)) { return acc; }
            return v < acc ? v : acc;
        case ReductionOpCode::NanMax:
            if (is_nan(v)) { return acc; }
            return v > acc ? v : acc;
        case ReductionOpCode::All:
            return static_cast<T>(acc && v);
        case ReductionOpCode::Any:
            return static_cast<T>(acc || v);
    }
    return acc;  // unreachable for a valid op
}

}  // namespace

// Atomic path: only float/double/int32/int64 have primitives above.
#define ZT_REDUCTION_ATOMIC(DT, CPP)                                           \
    if (src.scalar_type() == DT) {                                             \
        using scalar_t = CPP;                                                  \
        const auto id = identity_value<scalar_t>(op);                          \
        Fill(dst, Scalar(static_cast<double>(id)));                           \
        core::Indexer indexer(                                                 \
            {src}, dst, core::DtypePolicy::ALL_SAME, reduction_dims);          \
        core::ParallelFor(                                                     \
            dst.device(), indexer.NumWorkloads(), [=] __device__(int64_t i) {  \
                scalar_t* out = indexer.GetOutputPtr<scalar_t>(i);             \
                combine_atomic<scalar_t>(                                      \
                    op, out, *indexer.GetInputPtr<scalar_t>(0, i));            \
            });                                                                \
        if (op == ReductionOpCode::Mean) {                                     \
            const int64_t nout = dst.numel();                                  \
            const int64_t count =                                              \
                nout > 0 ? indexer.NumWorkloads() / nout : 1;                  \
            const double inv = 1.0 / static_cast<double>(count);               \
            scalar_t* dptr =                                                   \
                static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));     \
            core::ParallelFor(dst.device(), nout, [=] __device__(int64_t j) {  \
                dptr[j] =                                                      \
                    static_cast<scalar_t>(static_cast<double>(dptr[j]) * inv); \
            });                                                                \
        }                                                                      \
        return;                                                                \
    }

void ReductionCUDA(const Tensor& src,
                   const Tensor& dst,
                   IntArrayRef reduction_dims,
                   ReductionOpCode op) {
    CUDAScopedDevice scoped(dst.device());

    if (atomic_supported(src.scalar_type())) {
        ZT_REDUCTION_ATOMIC(ScalarType::Float, float)
        ZT_REDUCTION_ATOMIC(ScalarType::Double, double)
        ZT_REDUCTION_ATOMIC(ScalarType::Int, std::int32_t)
        ZT_REDUCTION_ATOMIC(ScalarType::Long, std::int64_t)
    }

    // Serial fallback (8/16-bit ints, bool, ...): reproduce the CPU
    // scatter-reduce in a single thread. Slow but correct and rare. The
    // Indexer is a POD built on the HOST (its ctor needs std::vector) and
    // captured by value into the device lambda.
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        const auto id = identity_value<scalar_t>(op);
        const int64_t nout = dst.numel();
        // Build the Indexer on the host; it is POD-by-value-captureable.
        core::Indexer indexer(
            {src}, dst, core::DtypePolicy::ALL_SAME, reduction_dims);
        const int64_t n = indexer.NumWorkloads();
        scalar_t* dptr =
            static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));
        core::ParallelFor(dst.device(), 1, [=] __device__(int64_t /*b*/) {
            for (int64_t j = 0; j < nout; ++j) {
                dptr[j] = id;
            }
            for (int64_t i = 0; i < n; ++i) {
                scalar_t* out = indexer.GetOutputPtr<scalar_t>(i);
                *out = combine_value(
                    op, *out, *indexer.GetInputPtr<scalar_t>(0, i));
            }
            if (op == ReductionOpCode::Mean) {
                const int64_t count = nout > 0 ? n / nout : 1;
                const double inv = 1.0 / static_cast<double>(count);
                for (int64_t j = 0; j < nout; ++j) {
                    dptr[j] = static_cast<scalar_t>(
                        static_cast<double>(dptr[j]) * inv);
                }
            }
        });
    });
}
#undef ZT_REDUCTION_ATOMIC

}  // namespace kernel
}  // namespace zt
