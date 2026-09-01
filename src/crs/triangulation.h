#pragma once

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_ray_float.hpp"
#include "ztensor/zt/Tensor.h"

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
                       zt::Tensor& rms);

// Float-path parameters: everything the ENU kernels/loops need beyond the
// point tensors, precomputed once per RpcStereo.
struct StereoFloatParams {
    RpcInfoFloat left;
    RpcInfoFloat right;
    RpcInverseInitFloat left_init;
    RpcInverseInitFloat right_init;
    EnuFrame frame;
    double h_low = 0.0;
    double h_high = 0.0;
};

void triangulation_cpu_float(const StereoFloatParams& params,
                             const zt::Tensor& left_colrow,
                             const zt::Tensor& right_colrow,
                             zt::Tensor& lonlath,
                             zt::Tensor& rms);

#ifdef BUILD_CUDA_MODULE
void triangulation_cuda(const RpcInfo& left,
                        const RpcInverseInit& left_init,
                        const RpcInfo& right,
                        const RpcInverseInit& right_init,
                        double h_low,
                        double h_high,
                        const zt::Tensor& left_colrow,
                        const zt::Tensor& right_colrow,
                        zt::Tensor& lonlath,
                        zt::Tensor& rms);

void triangulation_cuda_float(const StereoFloatParams& params,
                              const zt::Tensor& left_colrow,
                              const zt::Tensor& right_colrow,
                              zt::Tensor& lonlath,
                              zt::Tensor& rms);
#endif  // BUILD_CUDA_MODULE

}  // namespace zproj::crs
