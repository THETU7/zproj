#include "zproj/crs/wgs84.hpp"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/cuda/Exception.h"
#include "ztensor/zt/cuda/Stream.h"

namespace {

__global__ void wgs84_to_ecef_kernel(const zproj::crs::Geodetic* geo,
                                     zproj::crs::Ecef* wgs84,
                                     unsigned int num) {
    unsigned int idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (idx < num) {
        wgs84[idx] = zproj::crs::to_ecef(geo[idx]);
    }
}

}  // namespace

namespace zproj::crs {

void wgs84_to_ecef_cuda(const zt::Tensor& in, zt::Tensor& dst) {
    int64_t num = in.size(0);

    int64_t grid_size = (num + 255) / 256;

    const auto* in_ptr = in.data_ptr<zproj::crs::Geodetic>();

    Ecef* dst_ptr = dst.data_ptr<zproj::crs::Ecef>();

    // Launch on the caller's current (thread-local) stream so this kernel
    // stays ordered with other ztensor CUDA work, and check the launch.
    wgs84_to_ecef_kernel<<<static_cast<unsigned int>(grid_size), 256, 0,
                           zt::cuda::GetStream()>>>(
        in_ptr, dst_ptr, static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("wgs84_to_ecef_kernel launch failed");
}
}  // namespace zproj::crs
