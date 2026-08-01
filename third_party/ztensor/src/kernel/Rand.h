// ztensor/kernel/Rand.h
//
// Random-number fill kernels. Dispatches on device and dtype to fill a
// (possibly strided) tensor with samples from uniform [0,1), standard normal,
// or uniform integer distributions.
//
// The Philox engine is the RNG backbone; each element gets a deterministic
// counter-derived value so CPU and CUDA produce identical sequences for the
// same seed, and the fill is lock-free and parallel.

#pragma once

#include <cstdint>

#include "ztensor/zt/Generator.h"
#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

/// Fill `dst` with uniform [from, to) samples. Floating-point dtypes only.
void Rand(const Tensor& dst,
          double from,
          double to,
          const Generator& gen = Generator());

/// Fill `dst` with standard-normal samples. Floating-point dtypes only.
void RandN(const Tensor& dst,
           double mean,
           double stddev,
           const Generator& gen = Generator());

/// Fill `dst` with uniform integers in [low, high). Integer/Bool dtypes only.
void RandInt(const Tensor& dst,
             int64_t low,
             int64_t high,
             const Generator& gen = Generator());

// ── In-place convenience aliases ─────────────────────────────────────────
// These are the same as Rand/RandN/RandInt with explicit output.

inline void Uniform_(const Tensor& dst,
                     double from,
                     double to,
                     const Generator& gen = Generator()) {
    Rand(dst, from, to, gen);
}

inline void Normal_(const Tensor& dst,
                    double mean,
                    double stddev,
                    const Generator& gen = Generator()) {
    RandN(dst, mean, stddev, gen);
}

inline void Random_(const Tensor& dst,
                    int64_t from,
                    int64_t to,
                    const Generator& gen = Generator()) {
    RandInt(dst, from, to, gen);
}

// ── CPU / CUDA back-ends (exposed for testing and the device dispatcher) ─

void RandCPU(const Tensor& dst, double from, double to, const Generator& gen);
void RandNCPU(const Tensor& dst,
              double mean,
              double stddev,
              const Generator& gen);
void RandIntCPU(const Tensor& dst,
                int64_t low,
                int64_t high,
                const Generator& gen);

#ifdef BUILD_CUDA_MODULE
void RandCUDA(const Tensor& dst, double from, double to, const Generator& gen);
void RandNCUDA(const Tensor& dst,
               double mean,
               double stddev,
               const Generator& gen);
void RandIntCUDA(const Tensor& dst,
                 int64_t low,
                 int64_t high,
                 const Generator& gen);
#endif

}  // namespace zt::kernel
