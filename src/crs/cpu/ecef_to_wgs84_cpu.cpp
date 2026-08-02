#include "zproj/crs/wgs84.hpp"
#include "ztensor/zt/Tensor.h"

namespace zproj::crs {

void ecef_to_wgs84_cpu(const zt::Tensor& in, zt::Tensor& dst) {
    const auto* in_ptr = in.data_ptr<zproj::crs::Ecef>();

    Geodetic* dst_ptr = dst.data_ptr<zproj::crs::Geodetic>();

    int64_t num = in.size(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int64_t i = 0; i < num; ++i) {
        dst_ptr[i] = from_ecef(*(in_ptr + i));
    }
}

}  // namespace zproj::crs
