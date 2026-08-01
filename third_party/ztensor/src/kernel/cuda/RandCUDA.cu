// ztensor/kernel/RandCUDA.cu
//
// CUDA random fill: mirrors RandCPU.cpp's structure but drives work through
// core::ParallelFor (which launches the grid-stride ElementWiseKernel_).
// The Philox engine and distribution transforms are ZT_HOST_DEVICE and
// shared with the CPU path through Generator.h, guaranteeing identical
// sequences for the same seed.

#include <cstdint>
#include <type_traits>

#include "ztensor/zt/BFloat16.h"
#include "ztensor/zt/cuda/Guard.h"
#include "ztensor/zt/Generator.h"
#include "ztensor/zt/Half.h"
#include "ztensor/zt/Macros.h"
#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/Rand.h"

namespace zt {
namespace kernel {
namespace {

// ── Distribution helpers (ZT_HOST_DEVICE, shared signatures with CPU) ───

template<typename scalar_t>
__host__ __device__ inline scalar_t UniformSample(
    const PhiloxEngine::Output& out, double from, double to) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        float u = philox_uniform_float_u32(out[0]);
        return static_cast<scalar_t>(static_cast<float>(from) +
                                     u * static_cast<float>(to - from));
    } else if constexpr (std::is_same_v<scalar_t, double>) {
        double u = philox_uniform_double_u64(out[0], out[1]);
        return static_cast<scalar_t>(from + u * (to - from));
    } else {
        float u = philox_uniform_float_u32(out[0]);
        return static_cast<scalar_t>(static_cast<float>(from) +
                                     u * static_cast<float>(to - from));
    }
}

template<typename scalar_t>
__host__ __device__ inline scalar_t IntSample(const PhiloxEngine::Output& out,
                                              int64_t low,
                                              uint64_t range) {
    uint64_t v = static_cast<uint64_t>(out[0]);
    return static_cast<scalar_t>(static_cast<int64_t>(v % range) + low);
}

// ── Box-Muller device helpers ────────────────────────────────────────────
//
// NVCC rejects `if constexpr` inside an extended __device__ lambda that
// first-captures variables from the enclosing scope ("An extended
// __device__ lambda cannot first-capture variable in constexpr-if
// context").  Work around this by hoisting the type-switched Box-Muller
// logic into ordinary __device__ function templates; these are not
// lambdas, so `if constexpr` is perfectly fine here.

template<typename scalar_t>
__device__ inline void BoxMullerWritePair(scalar_t* p0,
                                          scalar_t* p1,
                                          const PhiloxEngine::Output& out0,
                                          const PhiloxEngine::Output& out1,
                                          double mean,
                                          double stddev) {
    if constexpr (std::is_same_v<scalar_t, double>) {
        double du1 = philox_uniform_double_u64(out0[0], out0[1]);
        double du2 = philox_uniform_double_u64(out1[0], out1[1]);
        double dn1, dn2;
        philox_box_muller_double(du1, du2, dn1, dn2);
        *p0 = static_cast<scalar_t>(mean + dn1 * stddev);
        *p1 = static_cast<scalar_t>(mean + dn2 * stddev);
    } else {
        float u1 = philox_uniform_float_u32(out0[0]);
        float u2 = philox_uniform_float_u32(out1[0]);
        float nn1, nn2;
        philox_box_muller(u1, u2, nn1, nn2);
        *p0 = static_cast<scalar_t>(static_cast<float>(mean) +
                                    nn1 * static_cast<float>(stddev));
        *p1 = static_cast<scalar_t>(static_cast<float>(mean) +
                                    nn2 * static_cast<float>(stddev));
    }
}

template<typename scalar_t>
__device__ inline void BoxMullerWriteSingle(scalar_t* p0,
                                            const PhiloxEngine::Output& out,
                                            const PhiloxEngine::Output& out2,
                                            double mean,
                                            double stddev) {
    if constexpr (std::is_same_v<scalar_t, double>) {
        double du1 = philox_uniform_double_u64(out[0], out[1]);
        double du2 = philox_uniform_double_u64(out2[0], out2[1]);
        double dn1, dn2;
        philox_box_muller_double(du1, du2, dn1, dn2);
        *p0 = static_cast<scalar_t>(mean + dn1 * stddev);
    } else {
        float u1 = philox_uniform_float_u32(out[0]);
        float u2 = philox_uniform_float_u32(out2[0]);
        float nn1, nn2;
        philox_box_muller(u1, u2, nn1, nn2);
        *p0 = static_cast<scalar_t>(static_cast<float>(mean) +
                                    nn1 * static_cast<float>(stddev));
    }
}

}  // namespace

// ── Rand (uniform [from, to)) ────────────────────────────────────────────

void RandCUDA(const Tensor& dst, double from, double to, const Generator& gen) {
    CUDAScopedDevice scoped(dst.device());
    ZT_CHECK(zt::isFloatingType(dst.scalar_type()),
             "RandCUDA: expected floating-point dtype, got {}",
             zt::toString(dst.scalar_type()));

    const uint64_t seed = gen.seed();
    const uint64_t base_ctr = gen.counter_offset();

    ZT_DISPATCH_SCALARTYPE_FLOAT_ONLY(dst.scalar_type(), [&] {
        if (dst.is_contiguous()) {
            auto* base =
                static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));
            const int64_t n = dst.numel();
            core::ParallelFor(dst.device(), n, [=] __device__(int64_t i) {
                PhiloxEngine eng(seed, base_ctr + static_cast<uint64_t>(i));
                PhiloxEngine::Output out;
                eng(out);
                base[i] = UniformSample<scalar_t>(out, from, to);
            });
        } else {
            core::Indexer indexer({dst}, dst, core::DtypePolicy::NONE);
            core::ParallelFor(
                dst.device(),
                indexer.NumWorkloads(),
                [=] __device__(int64_t i) {
                    PhiloxEngine eng(seed, base_ctr + static_cast<uint64_t>(i));
                    PhiloxEngine::Output out;
                    eng(out);
                    *indexer.GetOutputPtr<scalar_t>(i) =
                        UniformSample<scalar_t>(out, from, to);
                });
        }
    });
}

// ── RandN (normal) ───────────────────────────────────────────────────────

void RandNCUDA(const Tensor& dst,
               double mean,
               double stddev,
               const Generator& gen) {
    CUDAScopedDevice scoped(dst.device());
    ZT_CHECK(zt::isFloatingType(dst.scalar_type()),
             "RandNCUDA: expected floating-point dtype, got {}",
             zt::toString(dst.scalar_type()));

    const uint64_t seed = gen.seed();
    const uint64_t base_ctr = gen.counter_offset();
    const int64_t n = dst.numel();

    ZT_DISPATCH_SCALARTYPE_FLOAT_ONLY(dst.scalar_type(), [&] {
        if (dst.is_contiguous()) {
            auto* base =
                static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));
            core::ParallelFor(
                dst.device(), (n + 1) / 2, [=] __device__(int64_t pair) {
                    int64_t i0 = pair * 2;
                    uint64_t ctr = base_ctr + static_cast<uint64_t>(i0);

                    if (i0 + 1 < n) {
                        PhiloxEngine eng0(seed, ctr);
                        PhiloxEngine eng1(seed, ctr + 1);
                        PhiloxEngine::Output out0, out1;
                        eng0(out0);
                        eng1(out1);

                        BoxMullerWritePair(
                            &base[i0], &base[i0 + 1], out0, out1, mean, stddev);
                    } else {
                        // Odd last element: single sample.
                        PhiloxEngine eng(seed, ctr);
                        PhiloxEngine eng2(seed, ctr + 1);
                        PhiloxEngine::Output out, out2;
                        eng(out);
                        eng2(out2);
                        BoxMullerWriteSingle(
                            &base[i0], out, out2, mean, stddev);
                    }
                });
        } else {
            core::Indexer indexer({dst}, dst, core::DtypePolicy::NONE);
            core::ParallelFor(
                dst.device(),
                indexer.NumWorkloads(),
                [=] __device__(int64_t i) {
                    if (i % 2 != 0) return;
                    uint64_t ctr = base_ctr + static_cast<uint64_t>(i);
                    auto* p0 = indexer.GetOutputPtr<scalar_t>(i);
                    if (i + 1 < n) {
                        auto* p1 = indexer.GetOutputPtr<scalar_t>(i + 1);
                        PhiloxEngine eng0(seed, ctr);
                        PhiloxEngine eng1(seed, ctr + 1);
                        PhiloxEngine::Output out0, out1;
                        eng0(out0);
                        eng1(out1);
                        BoxMullerWritePair(p0, p1, out0, out1, mean, stddev);
                    } else {
                        PhiloxEngine eng(seed, ctr);
                        PhiloxEngine eng2(seed, ctr + 1);
                        PhiloxEngine::Output out, out2;
                        eng(out);
                        eng2(out2);
                        BoxMullerWriteSingle(p0, out, out2, mean, stddev);
                    }
                });
        }
    });
}

// ── RandInt ──────────────────────────────────────────────────────────────

void RandIntCUDA(const Tensor& dst,
                 int64_t low,
                 int64_t high,
                 const Generator& gen) {
    CUDAScopedDevice scoped(dst.device());
    ZT_CHECK(
        high > low, "RandIntCUDA: high ({}) must be > low ({})", high, low);
    ZT_CHECK(zt::isIntegralType(dst.scalar_type()) ||
                 zt::isBooleanType(dst.scalar_type()),
             "RandIntCUDA: expected integer or bool dtype, got {}",
             zt::toString(dst.scalar_type()));

    const uint64_t seed = gen.seed();
    const uint64_t base_ctr = gen.counter_offset();
    const uint64_t range = static_cast<uint64_t>(high - low);

    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(dst.scalar_type(), [&] {
        if (dst.is_contiguous()) {
            auto* base =
                static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));
            const int64_t n = dst.numel();
            core::ParallelFor(dst.device(), n, [=] __device__(int64_t i) {
                PhiloxEngine eng(seed, base_ctr + static_cast<uint64_t>(i));
                PhiloxEngine::Output out;
                eng(out);
                base[i] = IntSample<scalar_t>(out, low, range);
            });
        } else {
            core::Indexer indexer({dst}, dst, core::DtypePolicy::NONE);
            core::ParallelFor(
                dst.device(),
                indexer.NumWorkloads(),
                [=] __device__(int64_t i) {
                    PhiloxEngine eng(seed, base_ctr + static_cast<uint64_t>(i));
                    PhiloxEngine::Output out;
                    eng(out);
                    *indexer.GetOutputPtr<scalar_t>(i) =
                        IntSample<scalar_t>(out, low, range);
                });
        }
    });
}

}  // namespace kernel
}  // namespace zt
