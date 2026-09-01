// Float-precision RPC ray construction and triangulation in a scene-local
// ENU frame.
//
// WHY: on consumer GeForce parts FP64 runs at 1/64 of the FP32 rate, and the
// double pipeline (rpc_ray.hpp) is FP64-throughput-bound: the analytic RPC
// inverse's per-iteration polynomial dot products dominate it. The inverse
// already works in normalized lon/lat space (values in [-1, 1]), so it maps
// to float directly. ECEF positions (6.4e6 m, float ulp ~0.5 m) do NOT --
// re-centring the scene into a local ENU frame with a double origin puts
// every ray and intersection quantity back on scene scale (km), where float
// holds sub-millimetre precision.
//
// Accuracy budget (measured, 5 km footprint, GSD 0.1 m): horizontal ~1 mm,
// height ~2 cm, ray rms ~1.5 mm -- the same order as the double path's
// chord-approximation error. The float Newton bottoms out at ~1e-2 px (float
// evaluation noise vs the 0.1 px threshold), which sets the endpoint error;
// it scales with GSD, and the ENU float grid scales with the footprint, so
// continental-size scenes degrade toward metres. The geodetic <-> cartesian
// conversions stay double: float trig on absolute lon/lat angles alone costs
// ~0.7 m of ECEF noise per point.
//
// Everything here is host/device shared (ZT_HOST_DEVICE inline), mirroring
// rpc_ray.hpp, so the CUDA kernel and the CPU loop cannot drift apart.
#pragma once

#include <array>
#include <cmath>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/wgs84.hpp"

namespace zproj::crs {

// A point or vector in the scene-local ENU frame [m], single precision.
using Enu = zt::eigen::Vec3f;

// Float mirror of RpcInfo for the float Newton: the pixel/height
// normalization and the four coefficient sets, all float. The lon/lat
// normalization is kept in double because the float inverse's output must be
// re-centered exactly: reconstructing lon ~100 deg with float scales/offsets
// would quantize it at the float ulp (~7.6e-6 deg ~ 0.7 m of ground).
struct RpcInfoFloat {
    float samp_off = 0.0f;      // col offset [px]
    float line_off = 0.0f;      // row offset [px]
    float height_off = 0.0f;    // height offset [m]
    float samp_scale = 1.0f;    // col scale [px]
    float line_scale = 1.0f;    // row scale [px]
    float height_scale = 1.0f;  // height scale [m]

    std::array<float, kRpcCoeffCount> line_num_coeff{};
    std::array<float, kRpcCoeffCount> line_den_coeff{};
    std::array<float, kRpcCoeffCount> samp_num_coeff{};
    std::array<float, kRpcCoeffCount> samp_den_coeff{};

    // Exact lon/lat normalization (degrees), used only to reconstruct the
    // inverse's output from the converged normalized (L, P).
    double long_off = 0.0;
    double lat_off = 0.0;
    double long_scale = 1.0;
    double lat_scale = 1.0;
};

// Float affine seed, relative to the lon/lat offsets:
//   (lon - long_off) = lon_c0_rel + lon_c1*col + lon_c2*row
// Seeding the OFFSET (not the absolute ~100 deg longitude) keeps the float
// grid where it is tight. Built from a double RpcInverseInit.
struct RpcInverseInitFloat {
    float lon_c0_rel = 0.0f;
    float lon_c1 = 0.0f;
    float lon_c2 = 0.0f;
    float lat_c0_rel = 0.0f;
    float lat_c1 = 0.0f;
    float lat_c2 = 0.0f;
};

// Scene-local ENU frame: a double origin (ECEF) and the east/north/up unit
// basis at that point. Rays and intersections live in this frame as float.
struct EnuFrame {
    Ecef p0{0.0, 0.0, 0.0};
    Ecef east{1.0, 0.0, 0.0};
    Ecef north{0.0, 1.0, 0.0};
    Ecef up{0.0, 0.0, 1.0};
};

// A viewing ray in the scene-local ENU frame: origin [m from the frame
// origin] and its unit direction.
struct RpcRayEnu {
    Enu origin;
    Enu dir;
};

// ---- host-side construction (once per model / stereo pair) ----------------

// Float mirror of `info` for the float Newton. Coefficients are rounded to
// float, which perturbs the RPC by ~1e-7 relative -- a near-rigid shift of
// the back-projection locus that largely cancels in the ray direction.
inline RpcInfoFloat MakeRpcInfoFloat(const RpcInfo& info) {
    RpcInfoFloat f;
    f.samp_off = static_cast<float>(info.samp_off);
    f.line_off = static_cast<float>(info.line_off);
    f.height_off = static_cast<float>(info.height_off);
    f.samp_scale = static_cast<float>(info.samp_scale);
    f.line_scale = static_cast<float>(info.line_scale);
    f.height_scale = static_cast<float>(info.height_scale);
    for (int i = 0; i < kRpcCoeffCount; ++i) {
        f.line_num_coeff[i] = static_cast<float>(info.line_num_coeff[i]);
        f.line_den_coeff[i] = static_cast<float>(info.line_den_coeff[i]);
        f.samp_num_coeff[i] = static_cast<float>(info.samp_num_coeff[i]);
        f.samp_den_coeff[i] = static_cast<float>(info.samp_den_coeff[i]);
    }
    f.long_off = info.long_off;
    f.lat_off = info.lat_off;
    f.long_scale = info.long_scale;
    f.lat_scale = info.lat_scale;
    return f;
}

// Offset-relative float seed from the double affine inverse approximation.
inline RpcInverseInitFloat MakeRpcInverseInitFloat(const RpcInfo& info,
                                                   const RpcInverseInit& init) {
    RpcInverseInitFloat f;
    f.lon_c0_rel = static_cast<float>(init.lon_c0 - info.long_off);
    f.lon_c1 = static_cast<float>(init.lon_c1);
    f.lon_c2 = static_cast<float>(init.lon_c2);
    f.lat_c0_rel = static_cast<float>(init.lat_c0 - info.lat_off);
    f.lat_c1 = static_cast<float>(init.lat_c1);
    f.lat_c2 = static_cast<float>(init.lat_c2);
    return f;
}

// ENU frame centred between the two models' offset points (the overlap
// region of a stereo pair). The origin and basis stay double: they carry the
// 6.4e6 m ECEF magnitudes that float cannot hold.
inline EnuFrame MakeEnuFrame(const RpcInfo& left, const RpcInfo& right) {
    const double lon = 0.5 * (left.long_off + right.long_off) * kDegToRad;
    const double lat = 0.5 * (left.lat_off + right.lat_off) * kDegToRad;
    const double h = 0.5 * (left.height_off + right.height_off);

    EnuFrame frame;
    frame.p0 = to_ecef(Geodetic{lon, lat, h});
    const double sin_lon = sin(lon);
    const double cos_lon = cos(lon);
    const double sin_lat = sin(lat);
    const double cos_lat = cos(lat);
    frame.east = Ecef{-sin_lon, cos_lon, 0.0};
    frame.north = Ecef{-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat};
    frame.up = Ecef{cos_lat * cos_lon, cos_lat * sin_lon, sin_lat};
    return frame;
}

// ECEF -> ENU (float): the subtraction and projection run in double, only
// the scene-scale result is narrowed.
ZT_HOST_DEVICE inline Enu ToEnu(const EnuFrame& frame, const Ecef& p) {
    const double dx = p.x() - frame.p0.x();
    const double dy = p.y() - frame.p0.y();
    const double dz = p.z() - frame.p0.z();
    return Enu{static_cast<float>(frame.east.x() * dx + frame.east.y() * dy +
                                  frame.east.z() * dz),
               static_cast<float>(frame.north.x() * dx + frame.north.y() * dy +
                                  frame.north.z() * dz),
               static_cast<float>(frame.up.x() * dx + frame.up.y() * dy +
                                  frame.up.z() * dz)};
}

// ENU (float) -> ECEF (double).
ZT_HOST_DEVICE inline Ecef FromEnu(const EnuFrame& frame, const Enu& e) {
    const double x = static_cast<double>(e.x());
    const double y = static_cast<double>(e.y());
    const double z = static_cast<double>(e.z());
    return Ecef{frame.p0.x() + frame.east.x() * x + frame.north.x() * y +
                    frame.up.x() * z,
                frame.p0.y() + frame.east.y() * x + frame.north.y() * y +
                    frame.up.y() * z,
                frame.p0.z() + frame.east.z() * x + frame.north.z() * y +
                    frame.up.z() * z};
}

namespace detail {

ZT_HOST_DEVICE inline void rpc_compute_terms_float(float lon_n,
                                                   float lat_n,
                                                   float height_n,
                                                   float* terms) {
    terms[0] = 1.0f;
    terms[1] = lon_n;
    terms[2] = lat_n;
    terms[3] = height_n;
    terms[4] = lon_n * lat_n;
    terms[5] = lon_n * height_n;
    terms[6] = lat_n * height_n;
    terms[7] = lon_n * lon_n;
    terms[8] = lat_n * lat_n;
    terms[9] = height_n * height_n;
    terms[10] = lon_n * lat_n * height_n;
    terms[11] = lon_n * lon_n * lon_n;
    terms[12] = lon_n * lat_n * lat_n;
    terms[13] = lon_n * height_n * height_n;
    terms[14] = lon_n * lon_n * lat_n;
    terms[15] = lat_n * lat_n * lat_n;
    terms[16] = lat_n * height_n * height_n;
    terms[17] = lon_n * lon_n * height_n;
    terms[18] = lat_n * lat_n * height_n;
    terms[19] = height_n * height_n * height_n;
}

}  // namespace detail

// Analytic-Jacobian Newton inverse of the float RPC: the same iteration as
// rpc_inverse_point_analytic (true local Jacobian, quadratic convergence),
// evaluated in float in normalized (L, P) space. The float evaluation noise
// (~1e-7 relative on the residual) floors convergence near 1e-2 px, so
// `pixel_error_threshold` should stay at GDAL's 0.1 px -- tightening it
// below the noise floor just burns iterations. lon/lat are returned in
// degrees, reconstructed in double from the converged (L, P). Identical
// math on host and device.
ZT_HOST_DEVICE inline bool rpc_inverse_point_analytic_float(
    const RpcInfoFloat& info,
    const RpcInverseInitFloat& init,
    double col,
    double row,
    double height,
    double& lon,
    double& lat,
    float pixel_error_threshold,
    int max_iterations) {
    const float col_f = static_cast<float>(col);
    const float row_f = static_cast<float>(row);

    // Normalized target pixel.
    const float tgt_samp = (col_f - info.samp_off - 0.5f) / info.samp_scale;
    const float tgt_line = (row_f - info.line_off - 0.5f) / info.line_scale;

    // Seed (L, P) from the offset-relative affine (float grid is tight
    // there; no dateline handling -- the seed is scene-local by construction).
    const float dlon =
        init.lon_c0_rel + (init.lon_c1 * col_f) + (init.lon_c2 * row_f);
    const float dlat =
        init.lat_c0_rel + (init.lat_c1 * col_f) + (init.lat_c2 * row_f);
    float L = dlon / static_cast<float>(info.long_scale);
    float P = dlat / static_cast<float>(info.lat_scale);
    const float H =
        (static_cast<float>(height) - info.height_off) / info.height_scale;

    bool converged = false;
    for (int i = 0; i < max_iterations; ++i) {
        std::array<float, kRpcCoeffCount> terms;
        detail::rpc_compute_terms_float(L, P, H, terms.data());

        float Ns = 0.0f;
        float Ds = 0.0f;
        float Nl = 0.0f;
        float Dl = 0.0f;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif  // defined(__CUDA_ARCH__)
        for (int k = 0; k < kRpcCoeffCount; ++k) {
            Ns = fmaf(terms[k], info.samp_num_coeff[k], Ns);
            Ds = fmaf(terms[k], info.samp_den_coeff[k], Ds);
            Nl = fmaf(terms[k], info.line_num_coeff[k], Nl);
            Dl = fmaf(terms[k], info.line_den_coeff[k], Dl);
        }
        const float fs = (Ns / Ds) - tgt_samp;
        const float fl = (Nl / Dl) - tgt_line;

        const float err =
            fmaxf(fabsf(fs) * info.samp_scale, fabsf(fl) * info.line_scale);
        if (err < pixel_error_threshold) {
            converged = true;
            break;
        }

        // Analytic 2x2 Jacobian of (samp_n, line_n) w.r.t. (L, P) at fixed
        // H, via the quotient rule Q_i = (D*num_i - N*den_i)/D^2 (the float
        // counterpart of rpc_jac_from_eval).
        const std::array<float, kRpcCoeffCount> dL = {
            0.0f,  1.0f,         0.0f,         0.0f,  P,
            H,     0.0f,         2.0f * L,     0.0f,  0.0f,
            P * H, 3.0f * L * L, P * P,        H * H, 2.0f * L * P,
            0.0f,  0.0f,         2.0f * L * H, 0.0f,  0.0f};
        const std::array<float, kRpcCoeffCount> dP = {
            0.0f,         0.0f,  1.0f,         0.0f,         L,
            0.0f,         H,     0.0f,         2.0f * P,     0.0f,
            L * H,        0.0f,  2.0f * L * P, 0.0f,         L * L,
            3.0f * P * P, H * H, 0.0f,         2.0f * P * H, 0.0f};
        const float inv_Ds2 = 1.0f / (Ds * Ds);
        const float inv_Dl2 = 1.0f / (Dl * Dl);
        float js_ll = 0.0f;
        float js_lp = 0.0f;
        float jl_ll = 0.0f;
        float jl_lp = 0.0f;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif  // defined(__CUDA_ARCH__)
        for (int k = 0; k < kRpcCoeffCount; ++k) {
            const float Qs = ((Ds * info.samp_num_coeff[k]) -
                              (Ns * info.samp_den_coeff[k])) *
                             inv_Ds2;
            const float Ql = ((Dl * info.line_num_coeff[k]) -
                              (Nl * info.line_den_coeff[k])) *
                             inv_Dl2;
            js_ll = fmaf(Qs, dL[k], js_ll);
            js_lp = fmaf(Qs, dP[k], js_lp);
            jl_ll = fmaf(Ql, dL[k], jl_ll);
            jl_lp = fmaf(Ql, dP[k], jl_lp);
        }

        const float det = (js_ll * jl_lp) - (js_lp * jl_ll);
        if (det == 0.0f || !std::isfinite(det)) {
            break;  // singular / degenerate Jacobian
        }
        const float inv_det = 1.0f / det;

        // Newton step in normalized space.
        L -= (jl_lp * fs - js_lp * fl) * inv_det;
        P -= (-(jl_ll * fs) + (js_ll * fl)) * inv_det;
        if (!std::isfinite(L) || !std::isfinite(P)) {
            break;
        }
    }

    // Reconstruct lon/lat in double: L/P carry ~1e-7 relative float error,
    // and the double scales/offsets must not add quantization on top.
    lon = (static_cast<double>(L) * info.long_scale) + info.long_off;
    lat = (static_cast<double>(P) * info.lat_scale) + info.lat_off;
    return converged;
}

// Build the float viewing ray through (col, row) of an RPC image in the
// scene-local ENU frame: the float analytic inverse at two heights, the
// endpoints converted to ECEF in double (to_ecef needs absolute geodetic
// angles), then narrowed into the ENU frame where the ray direction and all
// downstream intersection math run in float. Heights are metres above the
// ellipsoid. Returns false if either inverse fails.
ZT_HOST_DEVICE inline bool rpc_ray_enu(const RpcInfoFloat& info,
                                       const RpcInverseInitFloat& init,
                                       const EnuFrame& frame,
                                       double col,
                                       double row,
                                       double h_low,
                                       double h_high,
                                       RpcRayEnu& ray) noexcept {
    constexpr float kThreshold = 0.1f;  // px; the float noise floor is ~1e-2
    constexpr int kMaxIterations = 20;

    double lon_low = 0.0;
    double lat_low = 0.0;
    if (!rpc_inverse_point_analytic_float(info,
                                          init,
                                          col,
                                          row,
                                          h_low,
                                          lon_low,
                                          lat_low,
                                          kThreshold,
                                          kMaxIterations)) {
        return false;
    }
    double lon_high = 0.0;
    double lat_high = 0.0;
    if (!rpc_inverse_point_analytic_float(info,
                                          init,
                                          col,
                                          row,
                                          h_high,
                                          lon_high,
                                          lat_high,
                                          kThreshold,
                                          kMaxIterations)) {
        return false;
    }

    const Ecef c_low =
        to_ecef(Geodetic{lon_low * kDegToRad, lat_low * kDegToRad, h_low});
    const Ecef c_high =
        to_ecef(Geodetic{lon_high * kDegToRad, lat_high * kDegToRad, h_high});

    const Enu a = ToEnu(frame, c_low);
    const Enu b = ToEnu(frame, c_high);
    const float dx = b.x() - a.x();
    const float dy = b.y() - a.y();
    const float dz = b.z() - a.z();
    const float len = sqrtf((dx * dx) + (dy * dy) + (dz * dz));
    if (!(len > 0.0f) || !std::isfinite(len)) {
        return false;
    }

    ray.origin = a;
    ray.dir = Enu{dx / len, dy / len, dz / len};
    return true;
}

// Two-view intersection in the ENU frame (float): midpoint of the closest
// points on the two rays, the same closed form as the double
// triangulate_pair. Sets `err` to the distance between the closest points
// [m]. Returns false if the rays are parallel.
ZT_HOST_DEVICE inline bool triangulate_pair(const RpcRayEnu& a,
                                            const RpcRayEnu& b,
                                            Enu& p,
                                            float& err) noexcept {
    const float v12x = (a.dir.y() * b.dir.z()) - (a.dir.z() * b.dir.y());
    const float v12y = (a.dir.z() * b.dir.x()) - (a.dir.x() * b.dir.z());
    const float v12z = (a.dir.x() * b.dir.y()) - (a.dir.y() * b.dir.x());
    const float v1x = (v12y * a.dir.z()) - (v12z * a.dir.y());
    const float v1y = (v12z * a.dir.x()) - (v12x * a.dir.z());
    const float v1z = (v12x * a.dir.y()) - (v12y * a.dir.x());
    const float v2x = (v12y * b.dir.z()) - (v12z * b.dir.y());
    const float v2y = (v12z * b.dir.x()) - (v12x * b.dir.z());
    const float v2z = (v12x * b.dir.y()) - (v12y * b.dir.x());

    // dot(v2, dir_a) = -sin^2 of the convergence angle (unit directions);
    // the threshold is the float counterpart of the double 1e-12.
    const float den_a =
        (v2x * a.dir.x()) + (v2y * a.dir.y()) + (v2z * a.dir.z());
    const float den_b =
        (v1x * b.dir.x()) + (v1y * b.dir.y()) + (v1z * b.dir.z());
    if (fabsf(den_a) <= 1e-10f || fabsf(den_b) <= 1e-10f) {
        return false;
    }

    const float wx = b.origin.x() - a.origin.x();
    const float wy = b.origin.y() - a.origin.y();
    const float wz = b.origin.z() - a.origin.z();
    const float ta = ((v2x * wx) + (v2y * wy) + (v2z * wz)) / den_a;
    const float ux = a.origin.x() - b.origin.x();
    const float uy = a.origin.y() - b.origin.y();
    const float uz = a.origin.z() - b.origin.z();
    const float tb = ((v1x * ux) + (v1y * uy) + (v1z * uz)) / den_b;

    const float cax = a.origin.x() + (a.dir.x() * ta);
    const float cay = a.origin.y() + (a.dir.y() * ta);
    const float caz = a.origin.z() + (a.dir.z() * ta);
    const float cbx = b.origin.x() + (b.dir.x() * tb);
    const float cby = b.origin.y() + (b.dir.y() * tb);
    const float cbz = b.origin.z() + (b.dir.z() * tb);

    p = Enu{0.5f * (cax + cbx), 0.5f * (cay + cby), 0.5f * (caz + cbz)};
    const float ex = cax - cbx;
    const float ey = cay - cby;
    const float ez = caz - cbz;
    err = sqrtf((ex * ex) + (ey * ey) + (ez * ez));
    return true;
}

}  // namespace zproj::crs
