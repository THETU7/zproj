// ztensor/zt/cuda/Warp.h
//
// Portable warp/wavefront primitives for GPU kernel (.cu) translation units.
// Self-contained: this header includes ztensor/zt/cuda/Vendor.h itself, but the
// conventional order keeps Vendor.h first. Valid only inside a CUDA (.cu,
// nvcc: __CUDACC__) or HIP (hip-clang: __HIPCC__) translation unit, because it
// wraps the vendor shuffle/sync builtins.
//
// Warp width differs by vendor: NVIDIA warps are 32 lanes; AMD wavefronts are
// 64 lanes (RDNA wave32 mode reports 32). The libSGM-style kernels this
// library serves keep a *logical* warp and rely on each shuffle's `width`
// argument to scope the shuffle to its sub-warp, so the same source compiles
// unchanged on both backends.
//
// Mask semantics (why the HIP branch drops the mask):
//   * CUDA `__shfl_*_sync` take a 32-bit mask and accept a *partial* mask, so
//     independent sub-warp shuffles (each sub-warp passes its own mask, e.g.
//     libSGM's `generate_mask<SUBGROUP>() << (group * SUBGROUP)`) are legal.
//     The CUDA branch forwards them verbatim.
//   * HIP `__shfl_*_sync` (ROCm >= 6.2) exist and are enabled by default, but
//     they take a *64-bit* mask and require it to equal the full active mask
//     (`mask == __ballot(true)`); they cannot express CUDA's independent
//     sub-warp masks. Since the `width` argument already scopes each shuffle,
//     and HIP's `_sync` merely validates the mask before delegating to the
//     unmasked `__shfl_*`, the HIP branch calls the unmasked `__shfl_*`
//     (dropping the mask, keeping width). This is semantically identical, not a
//     weaker fallback.

#pragma once

#include <cstdint>

#include "ztensor/zt/cuda/Vendor.h"

namespace zt {
namespace cuda {

#if defined(__HIPCC__)
// AMD wavefront: 64 lanes (RDNA wave32 mode ignores the upper half).
using WarpMask = uint64_t;
constexpr WarpMask kWarpFullMask = ~WarpMask{0};
#elif defined(__CUDACC__)
// NVIDIA warp: 32 lanes.
using WarpMask = uint32_t;
constexpr WarpMask kWarpFullMask = ~WarpMask{0};
#else
#error \
    "ztensor/zt/cuda/Warp.h must be included only from a CUDA or HIP translation unit"
#endif

#if defined(__HIPCC__)
// ----------------------------------------------------------------------------
// AMD HIP / ROCm -- unmasked shuffles (mask is redundant with `width`; see the
// header comment for why HIP's `__shfl_*_sync` cannot express sub-warp masks).
// ----------------------------------------------------------------------------
template<typename T>
__device__ inline T shfl_sync(WarpMask /*mask*/,
                              T var,
                              int src_lane,
                              int width) {
    return __shfl(var, src_lane, width);
}
template<typename T>
__device__ inline T shfl_up_sync(WarpMask /*mask*/,
                                 T var,
                                 unsigned int delta,
                                 int width) {
    return __shfl_up(var, delta, width);
}
template<typename T>
__device__ inline T shfl_down_sync(WarpMask /*mask*/,
                                   T var,
                                   unsigned int delta,
                                   int width) {
    return __shfl_down(var, delta, width);
}
template<typename T>
__device__ inline T shfl_xor_sync(WarpMask /*mask*/,
                                  T var,
                                  int lane_mask,
                                  int width) {
    return __shfl_xor(var, lane_mask, width);
}
// HIP's `__shfl_xor` has no bool overload; route bool through unsigned char
// (bit-identical, and the result round-trips to the same bool).
__device__ inline bool shfl_xor_sync(WarpMask /*mask*/,
                                     bool var,
                                     int lane_mask,
                                     int width) {
    return static_cast<bool>(
        __shfl_xor(static_cast<unsigned char>(var), lane_mask, width));
}
// __syncwarp() is a wavefront execution+memory barrier (HIP's unmasked form).
__device__ inline void syncwarp() { __syncwarp(); }

#else  // defined(__CUDACC__)
// ----------------------------------------------------------------------------
// NVIDIA CUDA -- forward to the native masked builtins.
// ----------------------------------------------------------------------------
template<typename T>
__device__ inline T shfl_sync(WarpMask mask, T var, int src_lane, int width) {
    return __shfl_sync(mask, var, src_lane, width);
}
template<typename T>
__device__ inline T shfl_up_sync(WarpMask mask,
                                 T var,
                                 unsigned int delta,
                                 int width) {
    return __shfl_up_sync(mask, var, delta, width);
}
template<typename T>
__device__ inline T shfl_down_sync(WarpMask mask,
                                   T var,
                                   unsigned int delta,
                                   int width) {
    return __shfl_down_sync(mask, var, delta, width);
}
template<typename T>
__device__ inline T shfl_xor_sync(WarpMask mask,
                                  T var,
                                  int lane_mask,
                                  int width) {
    return __shfl_xor_sync(mask, var, lane_mask, width);
}
__device__ inline void syncwarp() { __syncwarp(); }

#endif  // __HIPCC__

}  // namespace cuda
}  // namespace zt
