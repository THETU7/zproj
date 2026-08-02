// Public host API for the CUDA-accelerated WGS84 ECEF -> geodetic transform.
//
// The signatures intentionally avoid CUDA types, so plain C++ consumers
// (examples, tests, downstream apps) can call these without pulling in the
// CUDA headers or being compiled by nvcc.
#pragma once

#include "ztensor/zt/Tensor.h"

namespace zproj::crs {

/**
 * @brief transform points from ecef to geodetic (wgs84)
 * @param in points in ecef, required [N, 3]
 * @param dst out points in geodetic, can be empty or [N, 3]
 */
void ecef_to_wgs84(const zt::Tensor& in, zt::Tensor& dst);

}  // namespace zproj::crs
