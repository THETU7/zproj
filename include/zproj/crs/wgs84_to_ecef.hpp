// Public host API for the CUDA-accelerated WGS84 geodetic -> ECEF transform.
//
// The signatures intentionally avoid CUDA types, so plain C++ consumers
// (examples, tests, downstream apps) can call these without pulling in the
// CUDA headers or being compiled by nvcc.
#pragma once

#include "ztensor/zt/Tensor.h"

namespace zproj::crs {

/**
 * @brief transform points form geodetic to ecef
 * @param in points in geodetic, required [N, 3]
 * @param dst out points in ecef, can be empty or [N, 3]
 */
void wgs84_to_ecef(const zt::Tensor& in, zt::Tensor& dst);

}  // namespace zproj::crs
