#pragma once

#include "zproj/crs/rpc.hpp"
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
#endif  // BUILD_CUDA_MODULE

}  // namespace zproj::crs
