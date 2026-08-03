#pragma once

#include "zproj/rpc/rpc.hpp"

namespace zproj::rpc {

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

}  // namespace zproj::rpc
