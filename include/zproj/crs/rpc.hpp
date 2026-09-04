// Public host API for the RPC (Rational Polynomial Coefficients) sensor
// model -- a CUDA-accelerated reimplementation of GDAL's GDALRPCTransformer.
//
// RPC models (RPC00 / "Rigorous Projection Model", see
// http://geotiff.maptools.org/rpc_prop.html) describe the mapping between
// image coordinates (col, row) and ground coordinates (lon, lat, height)
// with four cubic rational polynomials over normalized coordinates:
//
//   * lonlatalt_to_colrow() applies the forward rational-polynomial
//     evaluation (lon/lat/alt -> col/row), the same math as GDAL's
//     RPCTransformPoint().
//   * colrowalt_to_lonlat() inverts it with the same fixed-point iteration
//     GDAL uses: an affine (col,row) -> (lon,lat) approximation computed once
//     at model construction seeds a Newton-style loop that forward-transforms
//     each guess and corrects it by the affine's (constant) inverse Jacobian.
//
// No DEM is involved: the caller supplies the height for every point, both
// for the forward and the inverse transform (GDAL's RPC_DEM machinery is
// intentionally not implemented). Lon/lat are in degrees, height in the RPC
// model's units (typically metres). The math is shared host/device via
// ZT_HOST_DEVICE, so the CPU reference and the GPU kernel cannot drift apart.
//
// The signatures intentionally avoid CUDA types: plain C++ consumers can call
// these without pulling in CUDA headers or being compiled by nvcc.
#pragma once

#include <array>
#include <cmath>

#include "ztensor/zt/Macros.h"
#include "ztensor/zt/Tensor.h"

namespace zproj::crs {

// Number of coefficients in each RPC rational polynomial: the 20 cubic
// monomials over (lon, lat, height) -- 1, L, P, H, LP, LH, PH, L2, P2, H2,
// LPH, L3, LP2, LH2, L2P, P3, PH2, L2H, P2H, H3.
inline constexpr int kRpcCoeffCount = 20;

// RPC model parameters. Field-for-field equivalent of GDALRPCInfoV2
// (gcore/gdal.h), kept GDAL-free so this header carries no GDAL dependency.
struct RpcInfo {
    double line_off = 0.0;  // row offset
    double samp_off = 0.0;  // col offset
    double lat_off = 0.0;
    double long_off = 0.0;
    double height_off = 0.0;

    double line_scale = 1.0;
    double samp_scale = 1.0;
    double lat_scale = 1.0;
    double long_scale = 1.0;
    double height_scale = 1.0;

    std::array<double, kRpcCoeffCount> line_num_coeff{};  // row numerator
    std::array<double, kRpcCoeffCount> line_den_coeff{};  // row denominator
    std::array<double, kRpcCoeffCount> samp_num_coeff{};  // col numerator
    std::array<double, kRpcCoeffCount> samp_den_coeff{};  // col denominator

    // Suggested validity bounds, used to pick the inverse's reference point
    // (GDAL semantics: the default -180..180 means "unknown").
    double min_lon = -180.0;
    double min_lat = -90.0;
    double max_lon = 180.0;
    double max_lat = 90.0;
};

// Which solver colrowalt_to_lonlat() uses to invert the RPC.
enum class InverseMethod {
    // GDAL's no-DEM inverse: an affine (col,row)->(lon,lat) approximation
    // seeds a Newton-style loop whose correction matrix is that same constant
    // affine (a frozen approximate inverse Jacobian) on every iteration.
    // Linearly convergent; matches GDAL's RPCInverseTransformPoint() exactly.
    AffineGdal,
    // Newton with the true local analytic Jacobian of the RPC, recomputed each
    // iteration (ASP's image_to_ground approach). Quadratically convergent and
    // reaches machine precision in a few iterations -- the right choice when
    // the ground points later define rays for triangulation.
    Analytic,
};

// Inverse solver tuning. Defaults match GDAL's no-DEM behavior.
struct RpcOptions {
    // Convergence threshold in pixels for the iterative inverse
    // (GDAL's dfPixErrThreshold / RPC_PIXEL_ERROR_THRESHOLD, default 0.1).
    double pixel_error_threshold = 0.1;
    // Iteration cap (GDAL's default is 10 when no DEM is used). Values <= 0
    // fall back to the default.
    int max_iterations = 10;
    // Solver selection (see InverseMethod). Default keeps the GDAL-equivalent
    // behavior so existing callers are unaffected.
    InverseMethod inverse_method = InverseMethod::AffineGdal;
};

// Affine (col, row) -> (lon, lat) initial approximation that seeds the
// iterative inverse. Computed once per model at construction time, the same
// way GDAL builds adfPLToLatLongGeoTransform in GDALCreateRPCTransformerV2.
struct RpcInverseInit {
    double lon_c0 = 0.0;  // lon = lon_c0 + lon_c1*col + lon_c2*row
    double lon_c1 = 1.0;
    double lon_c2 = 0.0;
    double lat_c0 = 0.0;  // lat = lat_c0 + lat_c1*col + lat_c2*row
    double lat_c1 = 0.0;
    double lat_c2 = 1.0;
};

// Fill `terms` with the 20 cubic monomials over the normalized coordinates
// (GDAL's RPCComputeTerms). Templated on the scalar type so the double
// reference path and the float path (rpc_ray_float.hpp) share ONE
// implementation of the polynomial. Identical math on host and device.
template<typename T>
ZT_HOST_DEVICE inline void rpc_compute_terms(T lon_n,
                                             T lat_n,
                                             T height_n,
                                             T* terms) {
    terms[0] = T(1);
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

// Forward RPC evaluation: (lon, lat, height) in degrees / model units ->
// (col, row) in GDAL's "top-left corner of pixel (0,0)" convention (RPCs are
// defined on pixel centres, hence the +0.5). Dateline wrapping matches GDAL's
// RPCTransformPoint(). Identical math on host and device.
ZT_HOST_DEVICE inline void rpc_forward_point(const RpcInfo& info,
                                             double lon,
                                             double lat,
                                             double height,
                                             double& col,
                                             double& row) {
    // Avoid dateline issues (GDAL RPCTransformPoint).
    double diff_long = lon - info.long_off;
    if (diff_long < -270.0) {
        diff_long += 360.0;
    } else if (diff_long > 270.0) {
        diff_long -= 360.0;
    }

    const double lon_n = diff_long / info.long_scale;
    const double lat_n = (lat - info.lat_off) / info.lat_scale;
    const double height_n = (height - info.height_off) / info.height_scale;

    std::array<double, kRpcCoeffCount> terms;
    rpc_compute_terms(lon_n, lat_n, height_n, terms.data());

    double line_num = 0.0;
    double line_den = 0.0;
    double samp_num = 0.0;
    double samp_den = 0.0;
    for (int i = 0; i < kRpcCoeffCount; ++i) {
        line_num += terms[i] * info.line_num_coeff[i];
        line_den += terms[i] * info.line_den_coeff[i];
        samp_num += terms[i] * info.samp_num_coeff[i];
        samp_den += terms[i] * info.samp_den_coeff[i];
    }

    // RPCs use the centre of the upper-left pixel as (0, 0); convert to
    // GDAL's top-left-corner convention by adding half a pixel.
    col = ((samp_num / samp_den) * info.samp_scale) + info.samp_off + 0.5;
    row = ((line_num / line_den) * info.line_scale) + info.line_off + 0.5;
}

// Iterative inverse: (col, row, height) -> (lon, lat). Seeds the Newton loop
// with the affine approximation and forward-transforms each guess at the
// caller-supplied height (no DEM). Returns false when the loop exhausts
// `max_iterations` without reaching `pixel_error_threshold`, mirroring
// GDAL's RPCInverseTransformPoint() no-DEM path (no oscillation/boost
// heuristics -- those only apply with a DEM). Identical math on host/device.
ZT_HOST_DEVICE inline bool rpc_inverse_point(const RpcInfo& info,
                                             const RpcInverseInit& init,
                                             double col,
                                             double row,
                                             double height,
                                             double& lon,
                                             double& lat,
                                             double pixel_error_threshold,
                                             int max_iterations) {
    double result_lon = init.lon_c0 + (init.lon_c1 * col) + (init.lon_c2 * row);
    double result_lat = init.lat_c0 + (init.lat_c1 * col) + (init.lat_c2 * row);

    for (int i = 0; i < max_iterations; ++i) {
        double back_col = 0.0;
        double back_row = 0.0;
        rpc_forward_point(
            info, result_lon, result_lat, height, back_col, back_row);

        const double col_delta = back_col - col;
        const double row_delta = back_row - row;
        const double error = fmax(fabs(col_delta), fabs(row_delta));
        if (error < pixel_error_threshold) {
            lon = result_lon;
            lat = result_lat;
            return true;
        }

        // Newton-style correction using the affine as the constant inverse
        // Jacobian (GDAL RPCInverseTransformPoint).
        const double new_lon =
            result_lon - (col_delta * init.lon_c1) - (row_delta * init.lon_c2);
        const double new_lat =
            result_lat - (col_delta * init.lat_c1) - (row_delta * init.lat_c2);
        result_lon = new_lon;
        result_lat = new_lat;
    }
    return false;
}

namespace detail {

// Quotient-rule Jacobian of the two rationals w.r.t. the normalized (L, P)
// at fixed H, straight from the four coefficient sets. Templated on the
// scalar type so the double and float inverses share ONE implementation.
// Internal helper: shared by rpc_jac_from_eval (below),
// rpc_pixel_jacobian, and the shared Newton core.
template<typename T>
ZT_HOST_DEVICE inline void rpc_jac_coeffs(
    const std::array<T, kRpcCoeffCount>& samp_num,
    const std::array<T, kRpcCoeffCount>& samp_den,
    const std::array<T, kRpcCoeffCount>& line_num,
    const std::array<T, kRpcCoeffCount>& line_den,
    T L,
    T P,
    T H,
    T Ns,
    T Ds,
    T Nl,
    T Dl,
    T& js_ll,
    T& js_lp,
    T& jl_ll,
    T& jl_lp) {
    // Derivatives of the 20 cubic monomials (see rpc_compute_terms) w.r.t.
    // L and P, with H held constant. Order matches the term index.
    const std::array<T, kRpcCoeffCount> dL = {
        T(0),  T(1),         T(0),         T(0),  P,
        H,     T(0),         T(2) * L,     T(0),  T(0),
        P * H, T(3) * L * L, P * P,        H * H, T(2) * L * P,
        T(0),  T(0),         T(2) * L * H, T(0),  T(0)};
    const std::array<T, kRpcCoeffCount> dP = {
        T(0),         T(0),  T(1),         T(0),         L,
        T(0),         H,     T(0),         T(2) * P,     T(0),
        L * H,        T(0),  T(2) * L * P, T(0),         L * L,
        T(3) * P * P, H * H, T(0),         T(2) * P * H, T(0)};

    const T inv_Ds2 = T(1) / (Ds * Ds);
    const T inv_Dl2 = T(1) / (Dl * Dl);
    js_ll = T(0);
    js_lp = T(0);
    jl_ll = T(0);
    jl_lp = T(0);
    for (int i = 0; i < kRpcCoeffCount; ++i) {
        const T Qs = ((Ds * samp_num[i]) - (Ns * samp_den[i])) * inv_Ds2;
        const T Ql = ((Dl * line_num[i]) - (Nl * line_den[i])) * inv_Dl2;
        js_ll += Qs * dL[i];
        js_lp += Qs * dP[i];
        jl_ll += Ql * dL[i];
        jl_lp += Ql * dP[i];
    }
}

}  // namespace detail

// Build the 2x2 Jacobian of the normalized pixel (S_norm, Line_norm) with
// respect to the normalized lon/lat (L, P) at a FIXED normalized height H,
// reusing the forward evaluation's terms and the four rational dot products.
// This is the ASP normalizedLlhToPixJac equivalent (RPCModel.cc:453). Internal
// helper: shared by rpc_pixel_jacobian (below) and the shared Newton core.
//
// For the rational R = (terms.num)/(terms.den), the quotient-jacobian of R
// w.r.t. term i is Q_i = (D*num_i - N*den_i)/D^2; then dR/dL = sum_i Q_i *
// d(term_i)/dL, with the monomial derivatives taken w.r.t. L and P only (H is
// fixed). Double wrapper over detail::rpc_jac_coeffs. Identical math on host
// and device.
ZT_HOST_DEVICE inline void rpc_jac_from_eval(const RpcInfo& info,
                                             double L,
                                             double P,
                                             double H,
                                             double Ns,
                                             double Ds,
                                             double Nl,
                                             double Dl,
                                             double& js_ll,
                                             double& js_lp,
                                             double& jl_ll,
                                             double& jl_lp) {
    detail::rpc_jac_coeffs(info.samp_num_coeff,
                           info.samp_den_coeff,
                           info.line_num_coeff,
                           info.line_den_coeff,
                           L,
                           P,
                           H,
                           Ns,
                           Ds,
                           Nl,
                           Dl,
                           js_ll,
                           js_lp,
                           jl_ll,
                           jl_lp);
}

namespace detail {

// The analytic-Jacobian Newton iteration over the normalized (L, P), shared
// by the double inverse (rpc_inverse_point_analytic) and the float inverse
// (rpc_inverse_point_analytic_float in rpc_ray_float.hpp): ONE implementation
// of the solver, instantiated per precision. Callers prepare the normalized
// target pixel, the seed (with dateline wrap applied), and the normalized
// height; the core converges L/P and reports success. Identical math on
// host/device.
template<typename T>
ZT_HOST_DEVICE inline bool rpc_newton_inverse_core(
    const std::array<T, kRpcCoeffCount>& samp_num,
    const std::array<T, kRpcCoeffCount>& samp_den,
    const std::array<T, kRpcCoeffCount>& line_num,
    const std::array<T, kRpcCoeffCount>& line_den,
    T tgt_samp,
    T tgt_line,
    T seed_l,
    T seed_p,
    T height_n,
    T samp_scale,
    T line_scale,
    T pixel_error_threshold,
    int max_iterations,
    T& l_out,
    T& p_out) noexcept {
    T L = seed_l;
    T P = seed_p;
    const T H = height_n;

    bool converged = false;
    for (int i = 0; i < max_iterations; ++i) {
        std::array<T, kRpcCoeffCount> terms;
        rpc_compute_terms(L, P, H, terms.data());

        T Ns = T(0);
        T Ds = T(0);
        T Nl = T(0);
        T Dl = T(0);
        for (int k = 0; k < kRpcCoeffCount; ++k) {
            Ns += terms[k] * samp_num[k];
            Ds += terms[k] * samp_den[k];
            Nl += terms[k] * line_num[k];
            Dl += terms[k] * line_den[k];
        }
        const T fs = (Ns / Ds) - tgt_samp;
        const T fl = (Nl / Dl) - tgt_line;

        // Convergence check in PIXEL space: a normalized residual f maps to
        // |f| * scale pixels.
        const T err = fmax(fabs(fs) * samp_scale, fabs(fl) * line_scale);
        if (err < pixel_error_threshold) {
            converged = true;
            break;
        }

        T js_ll = T(0);
        T js_lp = T(0);
        T jl_ll = T(0);
        T jl_lp = T(0);
        rpc_jac_coeffs(samp_num,
                       samp_den,
                       line_num,
                       line_den,
                       L,
                       P,
                       H,
                       Ns,
                       Ds,
                       Nl,
                       Dl,
                       js_ll,
                       js_lp,
                       jl_ll,
                       jl_lp);

        const T det = (js_ll * jl_lp) - (js_lp * jl_ll);
        if (det == T(0) || !std::isfinite(det)) {
            break;  // singular / degenerate Jacobian
        }
        const T inv_det = T(1) / det;

        // Newton step in normalized space: [L; P] -= Jn^{-1} [fs; fl].
        L -= (jl_lp * fs - js_lp * fl) * inv_det;
        P -= (-(jl_ll * fs) + (js_ll * fl)) * inv_det;
        if (!std::isfinite(L) || !std::isfinite(P)) {
            break;
        }
    }

    l_out = L;
    p_out = P;
    return converged;
}

}  // namespace detail

// Analytic Jacobian d(col,row)/d(lon,lat) at (lon, lat, height): the
// pixel-vs-geodetic Jacobian, chained from the normalized one through the
// scale/offset normalization. Exposed so tests can check it against finite
// differences of rpc_forward_point. Identical math on host and device.
ZT_HOST_DEVICE inline void rpc_pixel_jacobian(const RpcInfo& info,
                                              double lon,
                                              double lat,
                                              double height,
                                              double& dcol_dlon,
                                              double& dcol_dlat,
                                              double& drow_dlon,
                                              double& drow_dlat) {
    double diff_long = lon - info.long_off;
    if (diff_long < -270.0) {
        diff_long += 360.0;
    } else if (diff_long > 270.0) {
        diff_long -= 360.0;
    }
    const double L = diff_long / info.long_scale;
    const double P = (lat - info.lat_off) / info.lat_scale;
    const double H = (height - info.height_off) / info.height_scale;

    std::array<double, kRpcCoeffCount> terms;
    rpc_compute_terms(L, P, H, terms.data());

    double Ns = 0.0;
    double Ds = 0.0;
    double Nl = 0.0;
    double Dl = 0.0;
    for (int i = 0; i < kRpcCoeffCount; ++i) {
        Ns += terms[i] * info.samp_num_coeff[i];
        Ds += terms[i] * info.samp_den_coeff[i];
        Nl += terms[i] * info.line_num_coeff[i];
        Dl += terms[i] * info.line_den_coeff[i];
    }

    double js_ll = 0.0;
    double js_lp = 0.0;
    double jl_ll = 0.0;
    double jl_lp = 0.0;
    rpc_jac_from_eval(
        info, L, P, H, Ns, Ds, Nl, Dl, js_ll, js_lp, jl_ll, jl_lp);

    // col = S_norm * samp_scale + samp_off + 0.5;  L = (lon -
    // long_off)/long_scale
    dcol_dlon = info.samp_scale * js_ll / info.long_scale;
    dcol_dlat = info.samp_scale * js_lp / info.lat_scale;
    drow_dlon = info.line_scale * jl_ll / info.long_scale;
    drow_dlat = info.line_scale * jl_lp / info.lat_scale;
}

// Analytic-Jacobian inverse: (col, row, height) -> (lon, lat). Same
// fixed-height Newton solve as rpc_inverse_point, but the per-iteration
// correction uses the TRUE local Jacobian of the RPC (recomputed each step)
// instead of the constant affine, giving quadratic convergence. ASP's
// image_to_ground approach (RPCModel.cc:533). The convergence check stays in
// pixel space, so pixel_error_threshold means the same thing as for the affine
// solver; tighten it (e.g. 1e-9) to reach machine precision. Returns false on
// non-convergence or a singular Jacobian (caller writes HUGE_VAL, same
// convention). Thin wrapper over the precision-templated
// detail::rpc_newton_inverse_core, which is shared with the float path.
// Identical math on host and device.
ZT_HOST_DEVICE inline bool rpc_inverse_point_analytic(
    const RpcInfo& info,
    const RpcInverseInit& init,
    double col,
    double row,
    double height,
    double& lon,
    double& lat,
    double pixel_error_threshold,
    int max_iterations) {
    // Normalized target pixel (invert the pixel->normalized-pixel map).
    const double tgt_samp = (col - info.samp_off - 0.5) / info.samp_scale;
    const double tgt_line = (row - info.line_off - 0.5) / info.line_scale;

    // Seed (L, P) from the affine inverse -- same starting point as the affine
    // solver, so the only variable being changed is the correction matrix.
    // The seed longitude is wrapped to the model's hemisphere (GDAL's
    // dateline handling), and the float seed mirrors this exactly.
    const double seed_lon =
        init.lon_c0 + (init.lon_c1 * col) + (init.lon_c2 * row);
    const double seed_lat =
        init.lat_c0 + (init.lat_c1 * col) + (init.lat_c2 * row);
    double diff_long = seed_lon - info.long_off;
    if (diff_long < -270.0) {
        diff_long += 360.0;
    } else if (diff_long > 270.0) {
        diff_long -= 360.0;
    }
    const double seed_l = diff_long / info.long_scale;
    const double seed_p = (seed_lat - info.lat_off) / info.lat_scale;
    const double height_n = (height - info.height_off) / info.height_scale;

    double L = 0.0;
    double P = 0.0;
    const bool converged =
        detail::rpc_newton_inverse_core(info.samp_num_coeff,
                                        info.samp_den_coeff,
                                        info.line_num_coeff,
                                        info.line_den_coeff,
                                        tgt_samp,
                                        tgt_line,
                                        seed_l,
                                        seed_p,
                                        height_n,
                                        info.samp_scale,
                                        info.line_scale,
                                        pixel_error_threshold,
                                        max_iterations,
                                        L,
                                        P);

    lon = (L * info.long_scale) + info.long_off;
    lat = (P * info.lat_scale) + info.lat_off;
    return converged;
}

// RPC model: holds the coefficients plus the precomputed inverse initial
// approximation. Transform methods operate on [N, 3] double tensors; the
// output tensors are [N, 2].
class RpcModel {
public:
    explicit RpcModel(RpcInfo info, RpcOptions options = {});

    // lon/lat/alt [N, 3] (degrees, degrees, model units) -> col/row [N, 2].
    void lonlatalt_to_colrow(const zt::Tensor& in, zt::Tensor& dst) const;

    // col/row/alt [N, 3] -> lon/lat [N, 2] (degrees). Points that fail to
    // converge are set to HUGE_VAL in both columns (GDAL's failure
    // convention).
    void colrowalt_to_lonlat(const zt::Tensor& in, zt::Tensor& dst) const;

    const RpcInfo& info() const noexcept { return info_; }
    const RpcOptions& options() const noexcept { return options_; }
    const RpcInverseInit& inverse_init() const noexcept {
        return inverse_init_;
    }

private:
    RpcInfo info_;
    RpcOptions options_;
    RpcInverseInit inverse_init_;
};

}  // namespace zproj::crs
