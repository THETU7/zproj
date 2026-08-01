#pragma once

#include "zproj/crs/wgs84_to_ecef.hpp"

namespace zproj::crs {

void wgs84_to_ecef_cpu(const zt::Tensor& in, zt::Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void wgs84_to_ecef_cuda(const zt::Tensor& in, zt::Tensor& dst);
#endif  // BUILD_CUDA_MODULE

}  // namespace zproj::crs
