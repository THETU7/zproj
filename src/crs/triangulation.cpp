#include "triangulation.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "zproj/crs/triangulation.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace zproj::crs {

namespace {

// A [N, width] double tensor used as transform input/output.
void CheckPointTensor(const zt::Tensor& t, int64_t width, const char* name) {
    ZT_CHECK_EQ(t.dim(), 2);
    ZT_CHECK_EQ(t.size(1), width);
    ZT_CHECK(t.is_contiguous(), "{} must be contiguous", name);
    ZT_CHECK(t.scalar_type() == zt::ScalarType::Double,
             "{} must use double type",
             name);
}

// A [V, N, 2] double tensor of per-view (col, row) observations.
void CheckViewPointTensor(const zt::Tensor& t,
                          int64_t views,
                          const char* name) {
    ZT_CHECK_EQ(t.dim(), 3);
    ZT_CHECK_EQ(t.size(0), views);
    ZT_CHECK_EQ(t.size(2), 2);
    ZT_CHECK(t.is_contiguous(), "{} must be contiguous", name);
    ZT_CHECK(t.scalar_type() == zt::ScalarType::Double,
             "{} must use double type",
             name);
}

#ifdef BUILD_CUDA_MODULE
// Upload one trivially-copyable parameter array (RpcInfo / RpcInverseInit /
// float mirrors) to the target device as a kByte blob: a single coarse H2D
// memcpy of O(num_views) bytes, independent of the point count.
zt::Tensor ToDeviceBlob(const void* data,
                        std::size_t bytes,
                        const zt::Device& device) {
    const zt::Tensor host = zt::from_blob(const_cast<void*>(data),
                                          {static_cast<int64_t>(bytes)},
                                          zt::dtype(zt::kByte));
    return host.to(device);
}
#endif  // BUILD_CUDA_MODULE

}  // namespace

// The analytic inverse (threshold 1e-9 px, 20 iterations) is forced inside
// rpc_ray, so RpcStereo holds RpcModels only to carry each image's RpcInfo and
// precomputed affine seed (RpcInverseInit) -- their options_ are unused. The
// float-path precomputations are built alongside for StereoPrecision::FloatEnu.
RpcStereo::RpcStereo(RpcInfo left,
                     RpcInfo right,
                     double h_low,
                     double h_high,
                     StereoPrecision precision)
    : left_(left),
      right_(right),
      h_low_(h_low),
      h_high_(h_high),
      precision_(precision),
      left_float_(MakeRpcInfoFloat(left)),
      right_float_(MakeRpcInfoFloat(right)),
      left_init_float_(MakeRpcInverseInitFloat(left, left_.inverse_init())),
      right_init_float_(MakeRpcInverseInitFloat(right, right_.inverse_init())),
      enu_frame_(MakeEnuFrame(left, right)) {}

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
        if (precision_ == StereoPrecision::FloatEnu) {
            const StereoFloatParams params{left_float_,
                                           right_float_,
                                           left_init_float_,
                                           right_init_float_,
                                           enu_frame_,
                                           h_low_,
                                           h_high_};
            triangulation_cpu_float(
                params, left_colrow, right_colrow, lonlath, rms);
        } else {
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
        }
        return;
    }

#ifdef BUILD_CUDA_MODULE
    if (left_colrow.is_cuda() && right_colrow.is_cuda() && lonlath.is_cuda() &&
        rms.is_cuda()) {
        if (precision_ == StereoPrecision::FloatEnu) {
            const StereoFloatParams params{left_float_,
                                           right_float_,
                                           left_init_float_,
                                           right_init_float_,
                                           enu_frame_,
                                           h_low_,
                                           h_high_};
            triangulation_cuda_float(
                params, left_colrow, right_colrow, lonlath, rms);
        } else {
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
        }
        return;
    }
#endif  // BUILD_CUDA_MODULE

    ZT_LOG_ERROR("rpc_stereo: unsupported device (in {}, dst {})",
                 left_colrow.device().string(),
                 lonlath.device().string());
}

// Same arrangement as RpcStereo: RpcModel is constructed per view only to
// precompute the affine inverse seed (RpcInverseInit); the analytic inverse
// forced by rpc_ray / rpc_ray_enu does the rest.
RpcMultiStereo::RpcMultiStereo(std::vector<RpcInfo> infos,
                               double h_low,
                               double h_high,
                               StereoPrecision precision)
    : infos_(std::move(infos)),
      num_views_(static_cast<int>(infos_.size())),
      h_low_(h_low),
      h_high_(h_high),
      precision_(precision) {
    ZT_CHECK(infos_.size() >= 2,
             "RpcMultiStereo needs at least 2 RPC models, got {}",
             infos_.size());
    inits_.reserve(infos_.size());
    infos_float_.reserve(infos_.size());
    inits_float_.reserve(infos_.size());
    for (const RpcInfo& info : infos_) {
        const RpcModel model(info);
        inits_.push_back(model.inverse_init());
        infos_float_.push_back(MakeRpcInfoFloat(info));
        inits_float_.push_back(
            MakeRpcInverseInitFloat(info, model.inverse_init()));
    }
    enu_frame_ = MakeEnuFrame(infos_.data(), num_views_);
}

void RpcMultiStereo::triangulate(const zt::Tensor& colrow,
                                 zt::Tensor& lonlath,
                                 zt::Tensor& rms) const {
    CheckViewPointTensor(colrow, num_views_, "colrow");
    const int64_t num = colrow.size(1);

    if (!lonlath.defined()) {
        lonlath =
            zt::empty({num, 3}, zt::dtype(zt::kDouble).device(colrow.device()));
    }
    CheckPointTensor(lonlath, 3, "lonlath");
    ZT_CHECK_LE(num, lonlath.size(0));

    if (!rms.defined()) {
        rms = zt::empty({num}, zt::dtype(zt::kDouble).device(colrow.device()));
    }
    ZT_CHECK_EQ(rms.dim(), 1);
    ZT_CHECK_EQ(rms.size(0), num);
    ZT_CHECK(rms.is_contiguous(), "rms must be contiguous");
    ZT_CHECK(rms.scalar_type() == zt::ScalarType::Double,
             "rms must use double type");
    ZT_CHECK(rms.is_cpu() == colrow.is_cpu(),
             "rms must live on the same device as colrow");

    if (colrow.is_cpu() && lonlath.is_cpu() && rms.is_cpu()) {
        if (precision_ == StereoPrecision::FloatEnu) {
            triangulation_nview_cpu_float(infos_float_.data(),
                                          inits_float_.data(),
                                          enu_frame_,
                                          num_views_,
                                          h_low_,
                                          h_high_,
                                          colrow,
                                          lonlath,
                                          rms);
        } else {
            triangulation_nview_cpu(infos_.data(),
                                    inits_.data(),
                                    num_views_,
                                    h_low_,
                                    h_high_,
                                    colrow,
                                    lonlath,
                                    rms);
        }
        return;
    }

#ifdef BUILD_CUDA_MODULE
    if (colrow.is_cuda() && lonlath.is_cuda() && rms.is_cuda()) {
        if (precision_ == StereoPrecision::FloatEnu) {
            if (!infos_float_dev_.defined() ||
                infos_float_dev_.device() != colrow.device()) {
                infos_float_dev_ =
                    ToDeviceBlob(infos_float_.data(),
                                 infos_float_.size() * sizeof(RpcInfoFloat),
                                 colrow.device());
                inits_float_dev_ = ToDeviceBlob(
                    inits_float_.data(),
                    inits_float_.size() * sizeof(RpcInverseInitFloat),
                    colrow.device());
            }
            triangulation_cuda_nview_float(
                static_cast<const RpcInfoFloat*>(infos_float_dev_.data_ptr()),
                static_cast<const RpcInverseInitFloat*>(
                    inits_float_dev_.data_ptr()),
                enu_frame_,
                num_views_,
                h_low_,
                h_high_,
                colrow,
                lonlath,
                rms);
        } else {
            if (!infos_dev_.defined() ||
                infos_dev_.device() != colrow.device()) {
                infos_dev_ = ToDeviceBlob(infos_.data(),
                                          infos_.size() * sizeof(RpcInfo),
                                          colrow.device());
                inits_dev_ =
                    ToDeviceBlob(inits_.data(),
                                 inits_.size() * sizeof(RpcInverseInit),
                                 colrow.device());
            }
            triangulation_cuda_nview(
                static_cast<const RpcInfo*>(infos_dev_.data_ptr()),
                static_cast<const RpcInverseInit*>(inits_dev_.data_ptr()),
                num_views_,
                h_low_,
                h_high_,
                colrow,
                lonlath,
                rms);
        }
        return;
    }
#endif  // BUILD_CUDA_MODULE

    ZT_LOG_ERROR("rpc_multi_stereo: unsupported device (in {}, dst {})",
                 colrow.device().string(),
                 lonlath.device().string());
}

}  // namespace zproj::crs
