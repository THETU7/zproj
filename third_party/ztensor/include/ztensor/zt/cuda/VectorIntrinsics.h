// ztensor/zt/cuda/VectorIntrinsics.h
//
// Portable stand-ins for NVIDIA's packed (SIMD-within-register) integer
// intrinsics `__v{cmpgt,min,max}u{2,4}` that HIP/ROCm does not provide. They
// operate on one uint32_t holding 2x uint16 or 4x uint8 lanes, and are used by
// consumers that vectorise small-element kernels (e.g. a 3x3 median filter).
//
// On CUDA the intrinsics come from <cuda_runtime.h> (via Vendor.h), so this
// header is a no-op. On HIP it provides per-lane fallbacks with identical
// semantics; the device compiler vectorises the short unrolled loops, so the
// fallback is not the hot path but is auditable.

#pragma once

#include <cstdint>

#include "ztensor/zt/cuda/Vendor.h"

#if defined(__HIPCC__)

// 2x uint16 lanes.
static __device__ inline uint32_t __vcmpgtu2(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    if ((a & 0xFFFFu) > (b & 0xFFFFu)) r |= 0x0000FFFFu;
    if ((a >> 16) > (b >> 16)) r |= 0xFFFF0000u;
    return r;
}
static __device__ inline uint32_t __vminu2(uint32_t a, uint32_t b) {
    const uint32_t a0 = a & 0xFFFFu, a1 = a >> 16;
    const uint32_t b0 = b & 0xFFFFu, b1 = b >> 16;
    return (a0 < b0 ? a0 : b0) | ((a1 < b1 ? a1 : b1) << 16);
}
static __device__ inline uint32_t __vmaxu2(uint32_t a, uint32_t b) {
    const uint32_t a0 = a & 0xFFFFu, a1 = a >> 16;
    const uint32_t b0 = b & 0xFFFFu, b1 = b >> 16;
    return (a0 > b0 ? a0 : b0) | ((a1 > b1 ? a1 : b1) << 16);
}

// 4x uint8 lanes.
static __device__ inline uint32_t __vcmpgtu4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t al = (a >> (8 * i)) & 0xFFu;
        const uint32_t bl = (b >> (8 * i)) & 0xFFu;
        if (al > bl) r |= 0xFFu << (8 * i);
    }
    return r;
}
static __device__ inline uint32_t __vminu4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t al = (a >> (8 * i)) & 0xFFu;
        const uint32_t bl = (b >> (8 * i)) & 0xFFu;
        r |= (al < bl ? al : bl) << (8 * i);
    }
    return r;
}
static __device__ inline uint32_t __vmaxu4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t al = (a >> (8 * i)) & 0xFFu;
        const uint32_t bl = (b >> (8 * i)) & 0xFFu;
        r |= (al > bl ? al : bl) << (8 * i);
    }
    return r;
}

#elif defined(__CUDACC__)
// NVIDIA CUDA: __vcmpgtu2/4 etc. are provided by <cuda_runtime.h>.
#else
#error \
    "ztensor/zt/cuda/VectorIntrinsics.h must be included only from a CUDA or HIP translation unit"
#endif
