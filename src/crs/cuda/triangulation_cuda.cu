#include <cmath>
#include <cstdint>
#include <limits>

#include "zproj/crs/rpc_ray.hpp"
#include "ztensor/zt/cuda/Exception.h"
#include "ztensor/zt/cuda/Stream.h"
#include "ztensor/zt/utility/Log.h"

#include "crs/triangulation.h"

namespace {

constexpr unsigned int kBlock = 256;

__global__ void triangulation_kernel(
    const zproj::crs::RpcInfo left,
    const zproj::crs::RpcInverseInit left_init,
    const zproj::crs::RpcInfo right,
    const zproj::crs::RpcInverseInit right_init,
    const double* left_colrow,
    const double* right_colrow,
    double* lonlath,
    double* rms,
    double h_low,
    double h_high,
    unsigned int num) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num) {
        zproj::crs::RpcRay ray_left;
        zproj::crs::RpcRay ray_right;
        const bool rays_ok = zproj::crs::rpc_ray(left,
                                                 left_init,
                                                 left_colrow[2 * idx],
                                                 left_colrow[2 * idx + 1],
                                                 h_low,
                                                 h_high,
                                                 ray_left) &&
                             zproj::crs::rpc_ray(right,
                                                 right_init,
                                                 right_colrow[2 * idx],
                                                 right_colrow[2 * idx + 1],
                                                 h_low,
                                                 h_high,
                                                 ray_right);

        zproj::crs::Ecef p;
        double err = 0.0;
        const bool ok = rays_ok && zproj::crs::triangulate_pair(
                                       ray_left, ray_right, p, err);
        if (ok) {
            const zproj::crs::Geodetic g = zproj::crs::from_ecef(p);
            lonlath[3 * idx] = g.x() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 1] = g.y() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 2] = g.z();
            rms[idx] = err;
        } else {
            lonlath[3 * idx] = HUGE_VAL;
            lonlath[3 * idx + 1] = HUGE_VAL;
            lonlath[3 * idx + 2] = HUGE_VAL;
            rms[idx] = HUGE_VAL;
        }
    }
}

// Float path: float Newton inverse + ENU-local float rays and intersection;
// only the geodetic<->cartesian endpoint conversions stay double.
__global__ void triangulation_float_kernel(
    const zproj::crs::RpcInfoFloat left,
    const zproj::crs::RpcInverseInitFloat left_init,
    const zproj::crs::RpcInfoFloat right,
    const zproj::crs::RpcInverseInitFloat right_init,
    const zproj::crs::EnuFrame frame,
    const double* left_colrow,
    const double* right_colrow,
    double* lonlath,
    double* rms,
    double h_low,
    double h_high,
    unsigned int num) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num) {
        zproj::crs::RpcRayEnu ray_left;
        zproj::crs::RpcRayEnu ray_right;
        const bool rays_ok = zproj::crs::rpc_ray_enu(left,
                                                     left_init,
                                                     frame,
                                                     left_colrow[2 * idx],
                                                     left_colrow[2 * idx + 1],
                                                     h_low,
                                                     h_high,
                                                     ray_left) &&
                             zproj::crs::rpc_ray_enu(right,
                                                     right_init,
                                                     frame,
                                                     right_colrow[2 * idx],
                                                     right_colrow[2 * idx + 1],
                                                     h_low,
                                                     h_high,
                                                     ray_right);

        zproj::crs::Enu p_enu;
        float err = 0.0f;
        const bool ok = rays_ok && zproj::crs::triangulate_pair(
                                       ray_left, ray_right, p_enu, err);
        if (ok) {
            const zproj::crs::Geodetic g =
                zproj::crs::from_ecef(zproj::crs::FromEnu(frame, p_enu));
            lonlath[3 * idx] = g.x() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 1] = g.y() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 2] = g.z();
            rms[idx] = err;
        } else {
            lonlath[3 * idx] = HUGE_VAL;
            lonlath[3 * idx + 1] = HUGE_VAL;
            lonlath[3 * idx + 2] = HUGE_VAL;
            rms[idx] = HUGE_VAL;
        }
    }
}

// N-view kernels: per point, build the valid ray bundle across all views (a
// non-finite pixel marks an unobserved view and is skipped, like the CPU
// loop and VW StereoModel's NaN filtering; so is a non-convergent inverse),
// then triangulate_nview picks the closed form (2 valid rays) or the Slabaugh
// normal equations (3+). The ray bundle lives in thread-local memory, so the
// view count is capped at kMaxCudaNviewViews (enforced host-side).
__global__ void triangulation_nview_kernel(
    const zproj::crs::RpcInfo* infos,
    const zproj::crs::RpcInverseInit* inits,
    const double* colrow,  // [V * num * 2], view-major
    double* lonlath,
    double* rms,
    double h_low,
    double h_high,
    int num_views,
    unsigned int num) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num) {
        zproj::crs::RpcRay rays[zproj::crs::kMaxCudaNviewViews];
        int valid = 0;
        for (int v = 0; v < num_views; ++v) {
            const int64_t off =
                (static_cast<int64_t>(v) * static_cast<int64_t>(num)) + idx;
            const double col = colrow[2 * off];
            const double row = colrow[2 * off + 1];
            if (!std::isfinite(col) || !std::isfinite(row)) {
                continue;
            }
            if (zproj::crs::rpc_ray(
                    infos[v], inits[v], col, row, h_low, h_high, rays[valid])) {
                ++valid;
            }
        }

        zproj::crs::Ecef p;
        double err = 0.0;
        const bool ok =
            valid >= 2 && zproj::crs::triangulate_nview(rays, valid, p, err);
        if (ok) {
            const zproj::crs::Geodetic g = zproj::crs::from_ecef(p);
            lonlath[3 * idx] = g.x() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 1] = g.y() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 2] = g.z();
            rms[idx] = err;
        } else {
            lonlath[3 * idx] = HUGE_VAL;
            lonlath[3 * idx + 1] = HUGE_VAL;
            lonlath[3 * idx + 2] = HUGE_VAL;
            rms[idx] = HUGE_VAL;
        }
    }
}

// Float N-view path: float Newton inverse + ENU-local float rays and the
// float Slabaugh normal equations; only the geodetic<->cartesian endpoint
// conversions stay double.
__global__ void triangulation_nview_float_kernel(
    const zproj::crs::RpcInfoFloat* infos,
    const zproj::crs::RpcInverseInitFloat* inits,
    const zproj::crs::EnuFrame frame,
    const double* colrow,  // [V * num * 2], view-major
    double* lonlath,
    double* rms,
    double h_low,
    double h_high,
    int num_views,
    unsigned int num) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num) {
        zproj::crs::RpcRayEnu rays[zproj::crs::kMaxCudaNviewViews];
        int valid = 0;
        for (int v = 0; v < num_views; ++v) {
            const int64_t off =
                (static_cast<int64_t>(v) * static_cast<int64_t>(num)) + idx;
            const double col = colrow[2 * off];
            const double row = colrow[2 * off + 1];
            if (!std::isfinite(col) || !std::isfinite(row)) {
                continue;
            }
            if (zproj::crs::rpc_ray_enu(infos[v],
                                        inits[v],
                                        frame,
                                        col,
                                        row,
                                        h_low,
                                        h_high,
                                        rays[valid])) {
                ++valid;
            }
        }

        zproj::crs::Enu p_enu;
        float err = 0.0f;
        const bool ok = valid >= 2 &&
                        zproj::crs::triangulate_nview(rays, valid, p_enu, err);
        if (ok) {
            const zproj::crs::Geodetic g =
                zproj::crs::from_ecef(zproj::crs::FromEnu(frame, p_enu));
            lonlath[3 * idx] = g.x() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 1] = g.y() * zproj::crs::kRadToDeg;
            lonlath[3 * idx + 2] = g.z();
            rms[idx] = err;
        } else {
            lonlath[3 * idx] = HUGE_VAL;
            lonlath[3 * idx + 1] = HUGE_VAL;
            lonlath[3 * idx + 2] = HUGE_VAL;
            rms[idx] = HUGE_VAL;
        }
    }
}

}  // namespace

namespace zproj::crs {

void triangulation_cuda(const RpcInfo& left,
                        const RpcInverseInit& left_init,
                        const RpcInfo& right,
                        const RpcInverseInit& right_init,
                        double h_low,
                        double h_high,
                        const zt::Tensor& left_colrow,
                        const zt::Tensor& right_colrow,
                        zt::Tensor& lonlath,
                        zt::Tensor& rms) {
    const int64_t num = left_colrow.size(0);

    // The kernel indexes points with unsigned int; reject counts that would
    // truncate on the static_cast below.
    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "triangulation_cuda: point count {} exceeds the kernel's unsigned-int "
        "index range",
        num);

    const int64_t grid_size = (num + kBlock - 1) / kBlock;

    // Launch on the caller's current (thread-local) stream so this kernel
    // stays ordered with other ztensor CUDA work, and check the launch.
    triangulation_kernel<<<static_cast<unsigned int>(grid_size),
                           kBlock,
                           0,
                           zt::cuda::GetStream()>>>(
        left,
        left_init,
        right,
        right_init,
        left_colrow.data_ptr<double>(),
        right_colrow.data_ptr<double>(),
        lonlath.data_ptr<double>(),
        rms.data_ptr<double>(),
        h_low,
        h_high,
        static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("triangulation_kernel launch failed");
}

void triangulation_cuda_float(const StereoFloatParams& params,
                              const zt::Tensor& left_colrow,
                              const zt::Tensor& right_colrow,
                              zt::Tensor& lonlath,
                              zt::Tensor& rms) {
    const int64_t num = left_colrow.size(0);

    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "triangulation_cuda_float: point count {} exceeds the kernel's "
        "unsigned-int index range",
        num);

    const int64_t grid_size = (num + kBlock - 1) / kBlock;

    triangulation_float_kernel<<<static_cast<unsigned int>(grid_size),
                                 kBlock,
                                 0,
                                 zt::cuda::GetStream()>>>(
        params.left,
        params.left_init,
        params.right,
        params.right_init,
        params.frame,
        left_colrow.data_ptr<double>(),
        right_colrow.data_ptr<double>(),
        lonlath.data_ptr<double>(),
        rms.data_ptr<double>(),
        params.h_low,
        params.h_high,
        static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("triangulation_float_kernel launch failed");
}

void triangulation_cuda_nview(const RpcInfo* infos,
                              const RpcInverseInit* inits,
                              int num_views,
                              double h_low,
                              double h_high,
                              const zt::Tensor& colrow,
                              zt::Tensor& lonlath,
                              zt::Tensor& rms) {
    ZT_CHECK(num_views >= 2 && num_views <= kMaxCudaNviewViews,
             "triangulation_cuda_nview: view count {} outside the kernel's "
             "[2, {}] thread-local bundle range",
             num_views,
             kMaxCudaNviewViews);
    const int64_t num = colrow.size(1);

    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "triangulation_cuda_nview: point count {} exceeds the kernel's "
        "unsigned-int index range",
        num);

    const int64_t grid_size = (num + kBlock - 1) / kBlock;

    // infos/inits already live in device memory (uploaded once by the
    // caller), so the launch itself moves no model parameters.
    triangulation_nview_kernel<<<static_cast<unsigned int>(grid_size),
                                 kBlock,
                                 0,
                                 zt::cuda::GetStream()>>>(
        infos,
        inits,
        colrow.data_ptr<double>(),
        lonlath.data_ptr<double>(),
        rms.data_ptr<double>(),
        h_low,
        h_high,
        num_views,
        static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("triangulation_nview_kernel launch failed");
}

void triangulation_cuda_nview_float(const RpcInfoFloat* infos,
                                    const RpcInverseInitFloat* inits,
                                    const EnuFrame& frame,
                                    int num_views,
                                    double h_low,
                                    double h_high,
                                    const zt::Tensor& colrow,
                                    zt::Tensor& lonlath,
                                    zt::Tensor& rms) {
    ZT_CHECK(num_views >= 2 && num_views <= kMaxCudaNviewViews,
             "triangulation_cuda_nview_float: view count {} outside the "
             "kernel's [2, {}] thread-local bundle range",
             num_views,
             kMaxCudaNviewViews);
    const int64_t num = colrow.size(1);

    ZT_CHECK(
        num <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "triangulation_cuda_nview_float: point count {} exceeds the kernel's "
        "unsigned-int index range",
        num);

    const int64_t grid_size = (num + kBlock - 1) / kBlock;

    triangulation_nview_float_kernel<<<static_cast<unsigned int>(grid_size),
                                       kBlock,
                                       0,
                                       zt::cuda::GetStream()>>>(
        infos,
        inits,
        frame,
        colrow.data_ptr<double>(),
        lonlath.data_ptr<double>(),
        rms.data_ptr<double>(),
        h_low,
        h_high,
        num_views,
        static_cast<unsigned int>(num));
    ZT_CUDA_GET_LAST_ERROR("triangulation_nview_float_kernel launch failed");
}

}  // namespace zproj::crs
