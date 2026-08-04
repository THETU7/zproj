#pragma once

#include "zproj/crs/rpc.hpp"

namespace zproj::crs {

void rpc_forward_cpu(const RpcInfo& info,
                     const zt::Tensor& in,
                     zt::Tensor& dst);
void rpc_inverse_cpu(const RpcInfo& info,
                     const RpcInverseInit& init,
                     const RpcOptions& options,
                     const zt::Tensor& in,
                     zt::Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void rpc_forward_cuda(const RpcInfo& info,
                      const zt::Tensor& in,
                      zt::Tensor& dst);
void rpc_inverse_cuda(const RpcInfo& info,
                      const RpcInverseInit& init,
                      const RpcOptions& options,
                      const zt::Tensor& in,
                      zt::Tensor& dst);
#endif  // BUILD_CUDA_MODULE

}  // namespace zproj::crs
