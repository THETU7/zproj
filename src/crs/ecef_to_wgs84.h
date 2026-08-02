#pragma once

#include "zproj/crs/ecef_to_wgs84.hpp"

namespace zproj::crs {

void ecef_to_wgs84_cpu(const zt::Tensor& in, zt::Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void ecef_to_wgs84_cuda(const zt::Tensor& in, zt::Tensor& dst);
#endif  // BUILD_CUDA_MODULE

}  // namespace zproj::crs
