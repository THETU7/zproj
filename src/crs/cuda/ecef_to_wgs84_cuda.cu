#include <limits>

#include "zproj/crs/wgs84.hpp"
#include "ztensor/zt/cuda/Exception.h"
#include "ztensor/zt/cuda/Stream.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/utility/Log.h"

namespace {

__global__ void ecef_to_wgs84_kernel(const zproj::crs::Ecef* ecef,
                                     zproj::crs::Geodetic* wgs84,
                                     unsigned int num) {
    unsigned int idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (idx < num) {
        wgs84[idx] = zproj::crs::from_ecef(ecef[idx]);
    }
}

}  // namespace

namespace zproj::crs {

void ecef_to_wgs84_cuda(const zt::Tensor& in, zt::Tensor& dst) {
    int64_t num = in.size(0);

    // The kernel indexes points with unsigned int; reject counts that would
    // truncate on the static_cast below (~4B points is unreachable in
    // practice, but make the limit explicit rather than silent).
    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "ecef_to_wgs84_cuda: point count {} exceeds the kernel's "
        "unsigned-int index range",
        num);

    int64_t grid_size = (num + 255) / 256;

    const auto* in_ptr = in.data_ptr<zproj::crs::Ecef>();

    Geodetic* dst_ptr = dst.data_ptr<zproj::crs::Geodetic>();

    // Launch on the caller's current (thread-local) stream so this kernel
    // stays ordered with other ztensor CUDA work, and check the launch.
    ecef_to_wgs84_kernel<<<static_cast<unsigned int>(grid_size),
                           256,
                           0,
                           zt::cuda::GetStream()>>>(
        in_ptr, dst_ptr, static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("ecef_to_wgs84_kernel launch failed");
}
}  // namespace zproj::crs
