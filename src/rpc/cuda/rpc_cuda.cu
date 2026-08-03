#include <cmath>
#include <limits>

#include "ztensor/zt/cuda/Exception.h"
#include "ztensor/zt/cuda/Stream.h"
#include "ztensor/zt/utility/Log.h"

#include "rpc/rpc.h"

namespace {

__global__ void rpc_forward_kernel(const zproj::rpc::RpcInfo info,
                                   const double* in,
                                   double* out,
                                   unsigned int num) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num) {
        double col = 0.0;
        double row = 0.0;
        zproj::rpc::rpc_forward_point(
            info, in[3 * idx], in[3 * idx + 1], in[3 * idx + 2], col, row);
        out[2 * idx] = col;
        out[2 * idx + 1] = row;
    }
}

__global__ void rpc_inverse_kernel(const zproj::rpc::RpcInfo info,
                                   const zproj::rpc::RpcInverseInit init,
                                   const double* in,
                                   double* out,
                                   double pixel_error_threshold,
                                   int max_iterations,
                                   unsigned int num) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num) {
        double lon = 0.0;
        double lat = 0.0;
        const bool ok = zproj::rpc::rpc_inverse_point(info,
                                                      init,
                                                      in[3 * idx],
                                                      in[3 * idx + 1],
                                                      in[3 * idx + 2],
                                                      lon,
                                                      lat,
                                                      pixel_error_threshold,
                                                      max_iterations);
        if (ok) {
            out[2 * idx] = lon;
            out[2 * idx + 1] = lat;
        } else {
            out[2 * idx] = HUGE_VAL;
            out[2 * idx + 1] = HUGE_VAL;
        }
    }
}

}  // namespace

namespace zproj::rpc {

void rpc_forward_cuda(const RpcInfo& info,
                      const zt::Tensor& in,
                      zt::Tensor& dst) {
    const int64_t num = in.size(0);

    // The kernel indexes points with unsigned int; reject counts that would
    // truncate on the static_cast below.
    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "rpc_forward_cuda: point count {} exceeds the kernel's unsigned-int "
        "index range",
        num);

    const int64_t grid_size = (num + 255) / 256;

    // Launch on the caller's current (thread-local) stream so this kernel
    // stays ordered with other ztensor CUDA work, and check the launch.
    rpc_forward_kernel<<<static_cast<unsigned int>(grid_size),
                         256,
                         0,
                         zt::cuda::GetStream()>>>(
        info,
        in.data_ptr<double>(),
        dst.data_ptr<double>(),
        static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("rpc_forward_kernel launch failed");
}

void rpc_inverse_cuda(const RpcInfo& info,
                      const RpcInverseInit& init,
                      const RpcOptions& options,
                      const zt::Tensor& in,
                      zt::Tensor& dst) {
    const int64_t num = in.size(0);

    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "rpc_inverse_cuda: point count {} exceeds the kernel's unsigned-int "
        "index range",
        num);

    const int64_t grid_size = (num + 255) / 256;

    rpc_inverse_kernel<<<static_cast<unsigned int>(grid_size),
                         256,
                         0,
                         zt::cuda::GetStream()>>>(
        info,
        init,
        in.data_ptr<double>(),
        dst.data_ptr<double>(),
        options.pixel_error_threshold,
        options.max_iterations,
        static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("rpc_inverse_kernel launch failed");
}

}  // namespace zproj::rpc
