// ztensor/zt/Generator.h
//
// Philox 4×32-10 counter-based random number engine (PyTorch/Open3D RNG
// backbone). Counter-based RNGs are deterministic, stateless, and parallel:
// given a seed (the "key") and a position (the "counter"), the output is
// fully determined — no mutable state per thread, no locks, and identical
// sequences on CPU and CUDA.
//
// The Generator class wraps the global state (seed + counter offset) so that
// tensors of any size can be filled without re-generating the same value.
// Each parallel-for workload advances the counter linearly from a per-
// workload base, producing a deterministic sub-sequence.
//
// References:
//   Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
//   PyTorch aten/src/ATen/core/PhiloxRNGEngine.h

#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "ztensor/zt/Macros.h"

namespace zt {

// ── Helper: 32×32 → 64-bit widening multiply ───────────────────────────
ZT_HOST_DEVICE inline uint64_t philox_mul32(uint32_t a, uint32_t b) noexcept {
    return static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
}

// ── Philox 4×32-10 engine ─────────────────────────────────────────────────
//
// A single instance produces one std::array<uint32_t, 4> per invocation.
// The engine is a POD of std::array<uint32_t, 2> (key) + std::array<uint32_t,
// 4> (counter) and can be copied onto device stacks trivially.
class PhiloxEngine {
public:
    // Output type: four 32-bit random words.
    using Output = std::array<uint32_t, 4>;

    ZT_HOST_DEVICE explicit PhiloxEngine(uint64_t seed,
                                         uint64_t counter_lo = 0,
                                         uint64_t counter_hi = 0) noexcept {
        key_[0] = static_cast<uint32_t>(seed);
        key_[1] = static_cast<uint32_t>(seed >> 32);
        counter_[0] = static_cast<uint32_t>(counter_lo);
        counter_[1] = static_cast<uint32_t>(counter_lo >> 32);
        counter_[2] = static_cast<uint32_t>(counter_hi);
        counter_[3] = static_cast<uint32_t>(counter_hi >> 32);
    }

    ZT_HOST_DEVICE void operator()(Output& out) noexcept {
        compute(counter_.data(), key_.data(), out.data());
        advance_counter();
    }

    ZT_HOST_DEVICE void compute_at(Output& out) const noexcept {
        compute(counter_.data(), key_.data(), out.data());
    }

    ZT_HOST_DEVICE const uint32_t* counter_data() const noexcept {
        return counter_.data();
    }
    ZT_HOST_DEVICE const uint32_t* key_data() const noexcept {
        return key_.data();
    }

private:
    std::array<uint32_t, 2> key_ = {};
    std::array<uint32_t, 4> counter_ = {};

    ZT_HOST_DEVICE void advance_counter() noexcept {
        for (int i = 0; i < 4; ++i) {
            counter_[i]++;
            if (counter_[i] != 0U) {
                break;
            }
        }
    }

    static constexpr uint64_t kPhiloxMul = 0xD2B74407B1CE6E93ULL;
    static constexpr uint32_t kPhiloxWeylA = 0x9E3779B9U;
    static constexpr uint32_t kPhiloxWeylB = 0xBB67AE85U;

    ZT_HOST_DEVICE static void compute(const uint32_t* ctr,
                                       const uint32_t* key,
                                       uint32_t* out) noexcept {
        uint32_t k0 = key[0];
        uint32_t k1 = key[1];

        uint32_t s0 = ctr[0];
        uint32_t s1 = ctr[1];
        uint32_t s2 = ctr[2];
        uint32_t s3 = ctr[3];

        for (int i = 0; i < 9; ++i) {
            single_round(s0, s1, s2, s3, k0, k1);
            k0 += kPhiloxWeylA;
            k1 += kPhiloxWeylB;
        }
        // Tenth round (no Weyl addition, per the spec).
        {
            uint64_t p0 = philox_mul32(
                s0, static_cast<uint32_t>(kPhiloxMul));
            uint64_t p2 = philox_mul32(
                s2, static_cast<uint32_t>(kPhiloxMul));
            uint32_t ns0 = static_cast<uint32_t>(p2 >> 32) ^ k1 ^ s3;
            uint32_t ns1 = static_cast<uint32_t>(p2) ^ k0 ^ s2;
            uint32_t ns2 = static_cast<uint32_t>(p0 >> 32) ^ k1 ^ s1;
            uint32_t ns3 = static_cast<uint32_t>(p0) ^ k0 ^ s0;
            s0 = ns0;
            s1 = ns1;
            s2 = ns2;
            s3 = ns3;
        }

        out[0] = s0;
        out[1] = s1;
        out[2] = s2;
        out[3] = s3;
    }

    ZT_HOST_DEVICE static void single_round(uint32_t& s0,
                                            uint32_t& s1,
                                            uint32_t& s2,
                                            uint32_t& s3,
                                            uint32_t k0,
                                            uint32_t k1) noexcept {
        uint64_t p0 = philox_mul32(
            s0, static_cast<uint32_t>(kPhiloxMul));
        uint64_t p2 = philox_mul32(
            s2, static_cast<uint32_t>(kPhiloxMul));

        uint32_t ns0 = static_cast<uint32_t>(p2 >> 32) ^ k1 ^ s3;
        uint32_t ns1 = static_cast<uint32_t>(p2) ^ k0;
        uint32_t ns2 = static_cast<uint32_t>(p0 >> 32) ^ k1 ^ s1;
        uint32_t ns3 = static_cast<uint32_t>(p0) ^ k0 ^ s0;

        s0 = ns0;
        s1 = ns1;
        s2 = ns2;
        s3 = ns3;
    }
};

// ── Uniform / Normal distribution transforms ──────────────────────────────
//
// These are ZT_HOST_DEVICE free functions that consume Philox output.

/// Map one uint32_t to a `float` in [0, 1).
ZT_HOST_DEVICE inline float philox_uniform_float_u32(uint32_t x) noexcept {
    constexpr float kScale = 0x1.0p-24F;
    return static_cast<float>(x >> 8) * kScale;
}

/// Map two uint32_t to a `double` in [0, 1).
ZT_HOST_DEVICE inline double philox_uniform_double_u64(uint32_t hi,
                                                       uint32_t lo) noexcept {
    constexpr double kScale = 0x1.0p-52;
    uint64_t combined =
        (static_cast<uint64_t>(hi & 0x03FFFFFFU) << 26) |
        static_cast<uint64_t>(lo >> 6);
    return static_cast<double>(combined) * kScale;
}

/// Box-Muller: two independent uniform [0,1) floats → two standard-normal
/// samples. Uses the single-precision math functions to avoid double
/// promotion on CUDA devices.
ZT_HOST_DEVICE inline void philox_box_muller(float u1,
                                             float u2,
                                             float& n1,
                                             float& n2) noexcept {
    constexpr float kEps = 1e-10F;
    u1 = u1 > kEps ? u1 : kEps;
    float r = sqrtf(-2.0F * logf(u1));
    float theta = 2.0F * 3.14159265358979323846F * u2;
    n1 = r * cosf(theta);
    n2 = r * sinf(theta);
}

/// Box-Muller for double precision.
ZT_HOST_DEVICE inline void philox_box_muller_double(double u1,
                                                    double u2,
                                                    double& n1,
                                                    double& n2) noexcept {
    constexpr double kEps = 1e-15;
    u1 = u1 > kEps ? u1 : kEps;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    n1 = r * cos(theta);
    n2 = r * sin(theta);
}

// ── Generator ─────────────────────────────────────────────────────────────
//
// A handle to the global (or an explicit) random-number state.

class Generator {
public:
    Generator() = default;

    explicit Generator(uint64_t seed, uint64_t counter_offset = 0)
        : seed_(seed), counter_offset_(counter_offset) {}

    uint64_t seed() const noexcept {
        return seed_ > 0 ? seed_ : global_seed();
    }
    uint64_t counter_offset() const noexcept { return counter_offset_; }

    Generator offset(uint64_t n) const {
        return Generator(seed(), counter_offset_ + n);
    }

    static uint64_t global_seed() noexcept;
    static void set_global_seed(uint64_t seed) noexcept;

private:
    uint64_t seed_ = 0;
    uint64_t counter_offset_ = 0;
};

/// Set the global random seed.
void manual_seed(uint64_t seed);

}  // namespace zt
