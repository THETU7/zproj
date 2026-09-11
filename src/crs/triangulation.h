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

// ---- N-view (multi-stereo) entry points ---------------------------------
//
// infos/inits: num_views parameter arrays (host memory for the CPU entry
// points; DEVICE memory for the CUDA ones -- the caller uploads them once
// and keeps the buffer alive across launches).
//
// colrow: [num_views, num, 2] double. A non-finite pixel marks "this view
// does not observe this point" and is skipped; >= 2 valid views are
// required, else the point fails (HUGE_VAL, GDAL convention).

// The CUDA kernels keep the per-point ray bundle in thread-local memory;
// beyond this many views fall back to an error (the CPU path is unbounded).
inline constexpr int kMaxCudaNviewViews = 32;

void triangulation_nview_cpu(const RpcInfo* infos,
                             const RpcInverseInit* inits,
                             int num_views,
                             double h_low,
                             double h_high,
                             const zt::Tensor& colrow,
                             zt::Tensor& lonlath,
                             zt::Tensor& rms);

void triangulation_nview_cpu_float(const RpcInfoFloat* infos,
                                   const RpcInverseInitFloat* inits,
                                   const EnuFrame& frame,
                                   int num_views,
                                   double h_low,
                                   double h_high,
                                   const zt::Tensor& colrow,
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

// infos/inits point to DEVICE memory (uploaded by the caller, kept alive
// across launches); num_views <= kMaxCudaNviewViews is enforced here.
void triangulation_cuda_nview(const RpcInfo* infos,
                              const RpcInverseInit* inits,
                              int num_views,
                              double h_low,
                              double h_high,
                              const zt::Tensor& colrow,
                              zt::Tensor& lonlath,
                              zt::Tensor& rms);

void triangulation_cuda_nview_float(const RpcInfoFloat* infos,
                                    const RpcInverseInitFloat* inits,
                                    const EnuFrame& frame,
                                    int num_views,
                                    double h_low,
                                    double h_high,
                                    const zt::Tensor& colrow,
                                    zt::Tensor& lonlath,
                                    zt::Tensor& rms);
#endif  // BUILD_CUDA_MODULE

}  // namespace zproj::crs
