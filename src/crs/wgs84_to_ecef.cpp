#include "wgs84_to_ecef.h"

#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace zproj::crs {

void wgs84_to_ecef(const zt::Tensor& in, zt::Tensor& dst) {
    // [N, 3] check
    ZT_CHECK_EQ(in.dim(), 2);
    ZT_CHECK_EQ(in.size(1), 3);

    if (!dst.defined()) {
        dst = zt::empty_like(in);
    }

    ZT_CHECK_EQ(dst.dim(), 2);
    ZT_CHECK_EQ(dst.size(1), 3);
    ZT_CHECK_LE(in.size(0), dst.size(0));

    ZT_CHECK(in.is_contiguous() && dst.is_contiguous(),
             "required contiguous in wgs84_to_ecef, in: {}, dst: {}",
             in.is_contiguous(),
             dst.is_contiguous());

    ZT_CHECK(in.scalar_type() == zt::ScalarType::Double,
             "wgs84_to_ecef input tensor must use double type.");

    ZT_CHECK(dst.scalar_type() == zt::ScalarType::Double,
             "wgs84_to_ecef dst tensor must use double type.");

    if (in.is_cpu() && dst.is_cpu()) {
        wgs84_to_ecef_cpu(in, dst);
        return;
    }

#ifdef BUILD_CUDA_MODULE
    if (in.is_cuda() && dst.is_cuda()) {
        wgs84_to_ecef_cuda(in, dst);
        return;
    }
#endif  // BUILD_CUDA_MODULE

    ZT_LOG_ERROR("wgs84_to_ecef: unsupported device (in {}, dst {})",
                 in.device().string(),
                 dst.device().string());
}

}  // namespace zproj::crs
