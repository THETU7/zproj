#include <cmath>

#include "ztensor/zt/Tensor.h"

#include "crs/rpc.h"

namespace zproj::crs {

void rpc_forward_cpu(const RpcInfo& info,
                     const zt::Tensor& in,
                     zt::Tensor& dst) {
    const auto* in_ptr = in.data_ptr<double>();
    auto* dst_ptr = dst.data_ptr<double>();
    const int64_t num = in.size(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int64_t i = 0; i < num; ++i) {
        double col = 0.0;
        double row = 0.0;
        rpc_forward_point(info,
                          in_ptr[(3 * i) + 0],
                          in_ptr[(3 * i) + 1],
                          in_ptr[(3 * i) + 2],
                          col,
                          row);
        dst_ptr[(2 * i) + 0] = col;
        dst_ptr[(2 * i) + 1] = row;
    }
}

void rpc_inverse_cpu(const RpcInfo& info,
                     const RpcInverseInit& init,
                     const RpcOptions& options,
                     const zt::Tensor& in,
                     zt::Tensor& dst) {
    const auto* in_ptr = in.data_ptr<double>();
    auto* dst_ptr = dst.data_ptr<double>();
    const int64_t num = in.size(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int64_t i = 0; i < num; ++i) {
        double lon = 0.0;
        double lat = 0.0;
        const bool ok =
            (options.inverse_method == InverseMethod::Analytic)
                ? rpc_inverse_point_analytic(info,
                                             init,
                                             in_ptr[(3 * i) + 0],
                                             in_ptr[(3 * i) + 1],
                                             in_ptr[(3 * i) + 2],
                                             lon,
                                             lat,
                                             options.pixel_error_threshold,
                                             options.max_iterations)
                : rpc_inverse_point(info,
                                    init,
                                    in_ptr[(3 * i) + 0],
                                    in_ptr[(3 * i) + 1],
                                    in_ptr[(3 * i) + 2],
                                    lon,
                                    lat,
                                    options.pixel_error_threshold,
                                    options.max_iterations);
        if (ok) {
            dst_ptr[(2 * i) + 0] = lon;
            dst_ptr[(2 * i) + 1] = lat;
        } else {
            // GDAL's failure convention: HUGE_VAL in both columns.
            dst_ptr[(2 * i) + 0] = HUGE_VAL;
            dst_ptr[(2 * i) + 1] = HUGE_VAL;
        }
    }
}

}  // namespace zproj::crs
