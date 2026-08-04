#include <cmath>
#include <limits>

#include "zproj/crs/rpc_ray.hpp"
#include "ztensor/zt/cuda/Exception.h"
#include "ztensor/zt/cuda/Stream.h"
#include "ztensor/zt/utility/Log.h"

#include "crs/triangulation.h"

namespace {

__global__ void triangulation_kernel(
    const zproj::crs::RpcInfo left,
    const zproj::crs::RpcInverseInit left_init,
    const zproj::crs::RpcInfo right,
    const zproj::crs::RpcInverseInit right_init,
    const double* left_colrow,
    const double* right_colrow,
    double* lonlath,
    double* rms,
    double h_low,
    double h_high,
    unsigned int num) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num) {
        zproj::crs::RpcRay ray_left;
        zproj::crs::RpcRay ray_right;
        const bool rays_ok = zproj::crs::rpc_ray(left,
                                                 left_init,
                                                 left_colrow[2 * idx],
                                                 left_colrow[2 * idx + 1],
                                                 h_low,
                                                 h_high,
                                                 ray_left) &&
                             zproj::crs::rpc_ray(right,
                                                 right_init,
                                                 right_colrow[2 * idx],
                                                 right_colrow[2 * idx + 1],
                                                 h_low,
                                                 h_high,
                                                 ray_right);

        zproj::crs::Ecef p;
        double err = 0.0;
        const bool ok = rays_ok && zproj::crs::triangulate_pair(
                                       ray_left, ray_right, p, err);
        if (ok) {
            const zproj::crs::Geodetic g = zproj::crs::from_ecef(p);
            lonlath[3 * idx] = g.x() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 1] = g.y() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 2] = g.z();
            rms[idx] = err;
        } else {
            lonlath[3 * idx] = HUGE_VAL;
            lonlath[3 * idx + 1] = HUGE_VAL;
            lonlath[3 * idx + 2] = HUGE_VAL;
            rms[idx] = HUGE_VAL;
        }
    }
}

}  // namespace

namespace zproj::crs {

void triangulation_cuda(const RpcInfo& left,
                        const RpcInverseInit& left_init,
                        const RpcInfo& right,
                        const RpcInverseInit& right_init,
                        double h_low,
                        double h_high,
                        const zt::Tensor& left_colrow,
                        const zt::Tensor& right_colrow,
                        zt::Tensor& lonlath,
                        zt::Tensor& rms) {
    const int64_t num = left_colrow.size(0);

    // The kernel indexes points with unsigned int; reject counts that would
    // truncate on the static_cast below.
    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "triangulation_cuda: point count {} exceeds the kernel's unsigned-int "
        "index range",
        num);

    const int64_t grid_size = (num + 255) / 256;

    // Launch on the caller's current (thread-local) stream so this kernel
    // stays ordered with other ztensor CUDA work, and check the launch.
    triangulation_kernel<<<static_cast<unsigned int>(grid_size),
                           256,
                           0,
                           zt::cuda::GetStream()>>>(
        left,
        left_init,
        right,
        right_init,
        left_colrow.data_ptr<double>(),
        right_colrow.data_ptr<double>(),
        lonlath.data_ptr<double>(),
        rms.data_ptr<double>(),
        h_low,
        h_high,
        static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("triangulation_kernel launch failed");
}

}  // namespace zproj::crs
