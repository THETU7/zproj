#include <cmath>

#include "zproj/crs/rpc_ray.hpp"
#include "ztensor/zt/Tensor.h"

#include "crs/triangulation.h"

namespace zproj::crs {

void triangulation_cpu(const RpcInfo& left,
                       const RpcInverseInit& left_init,
                       const RpcInfo& right,
                       const RpcInverseInit& right_init,
                       double h_low,
                       double h_high,
                       const zt::Tensor& left_colrow,
                       const zt::Tensor& right_colrow,
                       zt::Tensor& lonlath,
                       zt::Tensor& rms) {
    const auto* left_ptr = left_colrow.data_ptr<double>();
    const auto* right_ptr = right_colrow.data_ptr<double>();
    auto* lonlath_ptr = lonlath.data_ptr<double>();
    auto* rms_ptr = rms.data_ptr<double>();
    const int64_t num = left_colrow.size(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int64_t i = 0; i < num; ++i) {
        RpcRay ray_left;
        RpcRay ray_right;
        const bool rays_ok = rpc_ray(left,
                                     left_init,
                                     left_ptr[(2 * i) + 0],
                                     left_ptr[(2 * i) + 1],
                                     h_low,
                                     h_high,
                                     ray_left) &&
                             rpc_ray(right,
                                     right_init,
                                     right_ptr[(2 * i) + 0],
                                     right_ptr[(2 * i) + 1],
                                     h_low,
                                     h_high,
                                     ray_right);

        Ecef p;
        double err = 0.0;
        const bool ok =
            rays_ok && triangulate_pair(ray_left, ray_right, p, err);
        if (ok) {
            const Geodetic g = from_ecef(p);
            lonlath_ptr[(3 * i) + 0] = g.x() * kRadToDeg;
            lonlath_ptr[(3 * i) + 1] = g.y() * kRadToDeg;
            lonlath_ptr[(3 * i) + 2] = g.z();
            rms_ptr[i] = err;
        } else {
            // GDAL's failure convention: HUGE_VAL everywhere.
            lonlath_ptr[(3 * i) + 0] = HUGE_VAL;
            lonlath_ptr[(3 * i) + 1] = HUGE_VAL;
            lonlath_ptr[(3 * i) + 2] = HUGE_VAL;
            rms_ptr[i] = HUGE_VAL;
        }
    }
}

}  // namespace zproj::crs
