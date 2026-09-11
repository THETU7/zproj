#include <cmath>
#include <vector>

#include "zproj/crs/rpc_ray.hpp"
#include "ztensor/zt/Tensor.h"

#include "crs/triangulation.h"

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
                       zt::Tensor& rms) {
    const auto* left_ptr = left_colrow.data_ptr<double>();
    const auto* right_ptr = right_colrow.data_ptr<double>();
    auto* lonlath_ptr = lonlath.data_ptr<double>();
    auto* rms_ptr = rms.data_ptr<double>();
    const int64_t num = left_colrow.size(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int64_t i = 0; i < num; ++i) {
        RpcRay ray_left;
        RpcRay ray_right;
        const bool rays_ok = rpc_ray(left,
                                     left_init,
                                     left_ptr[(2 * i) + 0],
                                     left_ptr[(2 * i) + 1],
                                     h_low,
                                     h_high,
                                     ray_left) &&
                             rpc_ray(right,
                                     right_init,
                                     right_ptr[(2 * i) + 0],
                                     right_ptr[(2 * i) + 1],
                                     h_low,
                                     h_high,
                                     ray_right);

        Ecef p;
        double err = 0.0;
        const bool ok =
            rays_ok && triangulate_pair(ray_left, ray_right, p, err);
        if (ok) {
            const Geodetic g = from_ecef(p);
            lonlath_ptr[(3 * i) + 0] = g.x() * kRadToDeg;
            lonlath_ptr[(3 * i) + 1] = g.y() * kRadToDeg;
            lonlath_ptr[(3 * i) + 2] = g.z();
            rms_ptr[i] = err;
        } else {
            // GDAL's failure convention: HUGE_VAL everywhere.
            lonlath_ptr[(3 * i) + 0] = HUGE_VAL;
            lonlath_ptr[(3 * i) + 1] = HUGE_VAL;
            lonlath_ptr[(3 * i) + 2] = HUGE_VAL;
            rms_ptr[i] = HUGE_VAL;
        }
    }
}

void triangulation_cpu_float(const StereoFloatParams& params,
                             const zt::Tensor& left_colrow,
                             const zt::Tensor& right_colrow,
                             zt::Tensor& lonlath,
                             zt::Tensor& rms) {
    const auto* left_ptr = left_colrow.data_ptr<double>();
    const auto* right_ptr = right_colrow.data_ptr<double>();
    auto* lonlath_ptr = lonlath.data_ptr<double>();
    auto* rms_ptr = rms.data_ptr<double>();
    const int64_t num = left_colrow.size(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int64_t i = 0; i < num; ++i) {
        RpcRayEnu ray_left;
        RpcRayEnu ray_right;
        const bool rays_ok = rpc_ray_enu(params.left,
                                         params.left_init,
                                         params.frame,
                                         left_ptr[(2 * i) + 0],
                                         left_ptr[(2 * i) + 1],
                                         params.h_low,
                                         params.h_high,
                                         ray_left) &&
                             rpc_ray_enu(params.right,
                                         params.right_init,
                                         params.frame,
                                         right_ptr[(2 * i) + 0],
                                         right_ptr[(2 * i) + 1],
                                         params.h_low,
                                         params.h_high,
                                         ray_right);

        Enu p_enu;
        float err = 0.0f;
        const bool ok =
            rays_ok && triangulate_pair(ray_left, ray_right, p_enu, err);
        if (ok) {
            const Geodetic g = from_ecef(FromEnu(params.frame, p_enu));
            lonlath_ptr[(3 * i) + 0] = g.x() * kRadToDeg;
            lonlath_ptr[(3 * i) + 1] = g.y() * kRadToDeg;
            lonlath_ptr[(3 * i) + 2] = g.z();
            rms_ptr[i] = err;
        } else {
            lonlath_ptr[(3 * i) + 0] = HUGE_VAL;
            lonlath_ptr[(3 * i) + 1] = HUGE_VAL;
            lonlath_ptr[(3 * i) + 2] = HUGE_VAL;
            rms_ptr[i] = HUGE_VAL;
        }
    }
}

namespace {

// Precision-specific pieces of the N-view loop below, resolved by
// overloading: the ray builder (ECEF double vs scene-local ENU float) and
// the intersection -> ECEF narrowing.
bool BuildRay(const RpcInfo& info,
              const RpcInverseInit& init,
              const EnuFrame* /*frame*/,
              double col,
              double row,
              double h_low,
              double h_high,
              RpcRay& ray) {
    return rpc_ray(info, init, col, row, h_low, h_high, ray);
}

bool BuildRay(const RpcInfoFloat& info,
              const RpcInverseInitFloat& init,
              const EnuFrame* frame,
              double col,
              double row,
              double h_low,
              double h_high,
              RpcRayEnu& ray) {
    return rpc_ray_enu(info, init, *frame, col, row, h_low, h_high, ray);
}

Ecef ToEcef(const EnuFrame* /*frame*/, const Ecef& p) { return p; }

Ecef ToEcef(const EnuFrame* frame, const Enu& p) { return FromEnu(*frame, p); }

// Per-point N-view loop shared by both precision paths: build the valid ray
// bundle (a non-finite pixel marks an unobserved view and is skipped, like
// VW StereoModel's NaN filtering; so is a non-convergent inverse), require
// >= 2 rays, then let triangulate_nview pick the closed form (2 rays) or the
// Slabaugh normal equations (3+). Failed points follow the GDAL HUGE_VAL
// convention.
template<typename RayT,
         typename InfoT,
         typename InitT,
         typename Vec3,
         typename T>
void NviewLoop(const InfoT* infos,
               const InitT* inits,
               const EnuFrame* frame,
               int num_views,
               double h_low,
               double h_high,
               const double* colrow_ptr,
               double* lonlath_ptr,
               double* rms_ptr,
               int64_t num) {
#ifdef _OPENMP
#pragma omp parallel
#endif  // _OPENMP
    {
        // Reused per-thread ray storage: one bundle allocation per thread,
        // not per point.
        std::vector<RayT> rays(num_views);
#ifdef _OPENMP
#pragma omp for
#endif  // _OPENMP
        for (int64_t i = 0; i < num; ++i) {
            int valid = 0;
            for (int v = 0; v < num_views; ++v) {
                // [V, N, 2] layout: view-major, point-minor.
                const double col = colrow_ptr[2 * ((v * num) + i)];
                const double row = colrow_ptr[2 * ((v * num) + i) + 1];
                if (!std::isfinite(col) || !std::isfinite(row)) {
                    continue;
                }
                if (BuildRay(infos[v],
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

            Vec3 p;
            T err = 0;
            const bool ok =
                valid >= 2 && triangulate_nview(rays.data(), valid, p, err);
            if (ok) {
                const Geodetic g = from_ecef(ToEcef(frame, p));
                lonlath_ptr[(3 * i) + 0] = g.x() * kRadToDeg;
                lonlath_ptr[(3 * i) + 1] = g.y() * kRadToDeg;
                lonlath_ptr[(3 * i) + 2] = g.z();
                rms_ptr[i] = err;
            } else {
                lonlath_ptr[(3 * i) + 0] = HUGE_VAL;
                lonlath_ptr[(3 * i) + 1] = HUGE_VAL;
                lonlath_ptr[(3 * i) + 2] = HUGE_VAL;
                rms_ptr[i] = HUGE_VAL;
            }
        }
    }
}

}  // namespace

void triangulation_nview_cpu(const RpcInfo* infos,
                             const RpcInverseInit* inits,
                             int num_views,
                             double h_low,
                             double h_high,
                             const zt::Tensor& colrow,
                             zt::Tensor& lonlath,
                             zt::Tensor& rms) {
    NviewLoop<RpcRay, RpcInfo, RpcInverseInit, Ecef, double>(
        infos,
        inits,
        /*frame=*/nullptr,
        num_views,
        h_low,
        h_high,
        colrow.data_ptr<double>(),
        lonlath.data_ptr<double>(),
        rms.data_ptr<double>(),
        colrow.size(1));
}

void triangulation_nview_cpu_float(const RpcInfoFloat* infos,
                                   const RpcInverseInitFloat* inits,
                                   const EnuFrame& frame,
                                   int num_views,
                                   double h_low,
                                   double h_high,
                                   const zt::Tensor& colrow,
                                   zt::Tensor& lonlath,
                                   zt::Tensor& rms) {
    NviewLoop<RpcRayEnu, RpcInfoFloat, RpcInverseInitFloat, Enu, float>(
        infos,
        inits,
        &frame,
        num_views,
        h_low,
        h_high,
        colrow.data_ptr<double>(),
        lonlath.data_ptr<double>(),
        rms.data_ptr<double>(),
        colrow.size(1));
}

}  // namespace zproj::crs
