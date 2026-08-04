#include "triangulation.h"

#include <stdexcept>

#include "zproj/crs/triangulation.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace zproj::crs {

namespace {

// Force the analytic inverse with a tight threshold: rays feeding
// triangulation need machine precision, not the affine's ~0.1 px.
RpcOptions AnalyticOptions(const RpcOptions& options) {
    RpcOptions analytic = options;
    analytic.inverse_method = InverseMethod::Analytic;
    analytic.pixel_error_threshold = 1e-9;
    analytic.max_iterations = 20;
    return analytic;
}

// A [N, width] double tensor used as transform input/output.
void CheckPointTensor(const zt::Tensor& t, int64_t width, const char* name) {
    ZT_CHECK_EQ(t.dim(), 2);
    ZT_CHECK_EQ(t.size(1), width);
    ZT_CHECK(t.is_contiguous(), "{} must be contiguous", name);
    ZT_CHECK(t.scalar_type() == zt::ScalarType::Double,
             "{} must use double type",
             name);
}

}  // namespace

RpcStereo::RpcStereo(RpcInfo left,
                     RpcInfo right,
                     double h_low,
                     double h_high,
                     RpcOptions options)
    : left_(left, AnalyticOptions(options)),
      right_(right, AnalyticOptions(options)),
      h_low_(h_low),
      h_high_(h_high) {}

void RpcStereo::triangulate(const zt::Tensor& left_colrow,
                            const zt::Tensor& right_colrow,
                            zt::Tensor& lonlath,
                            zt::Tensor& rms) const {
    CheckPointTensor(left_colrow, 2, "left_colrow");
    CheckPointTensor(right_colrow, 2, "right_colrow");
    ZT_CHECK_EQ(left_colrow.size(0), right_colrow.size(0));
    const int64_t num = left_colrow.size(0);

    if (!lonlath.defined()) {
        lonlath = zt::empty(
            {num, 3}, zt::dtype(zt::kDouble).device(left_colrow.device()));
    }
    CheckPointTensor(lonlath, 3, "lonlath");
    ZT_CHECK_LE(num, lonlath.size(0));

    if (!rms.defined()) {
        rms = zt::empty({num},
                        zt::dtype(zt::kDouble).device(left_colrow.device()));
    }
    ZT_CHECK_EQ(rms.dim(), 1);
    ZT_CHECK_EQ(rms.size(0), num);
    ZT_CHECK(rms.is_contiguous(), "rms must be contiguous");
    ZT_CHECK(rms.scalar_type() == zt::ScalarType::Double,
             "rms must use double type");
    ZT_CHECK(rms.is_cpu() == left_colrow.is_cpu(),
             "rms must live on the same device as left_colrow");

    if (left_colrow.is_cpu() && right_colrow.is_cpu() && lonlath.is_cpu() &&
        rms.is_cpu()) {
        triangulation_cpu(left_.info(),
                          left_.inverse_init(),
                          right_.info(),
                          right_.inverse_init(),
                          h_low_,
                          h_high_,
                          left_colrow,
                          right_colrow,
                          lonlath,
                          rms);
        return;
    }

#ifdef BUILD_CUDA_MODULE
    if (left_colrow.is_cuda() && right_colrow.is_cuda() && lonlath.is_cuda() &&
        rms.is_cuda()) {
        triangulation_cuda(left_.info(),
                           left_.inverse_init(),
                           right_.info(),
                           right_.inverse_init(),
                           h_low_,
                           h_high_,
                           left_colrow,
                           right_colrow,
                           lonlath,
                           rms);
        return;
    }
#endif  // BUILD_CUDA_MODULE

    ZT_LOG_ERROR("rpc_stereo: unsupported device (in {}, dst {})",
                 left_colrow.device().string(),
                 lonlath.device().string());
}

}  // namespace zproj::crs
