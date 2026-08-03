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

namespace zproj::rpc {

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

// Inverse solver tuning. Defaults match GDAL's no-DEM behavior.
struct RpcOptions {
    // Convergence threshold in pixels for the iterative inverse
    // (GDAL's dfPixErrThreshold / RPC_PIXEL_ERROR_THRESHOLD, default 0.1).
    double pixel_error_threshold = 0.1;
    // Iteration cap (GDAL's default is 10 when no DEM is used). Values <= 0
    // fall back to the default.
    int max_iterations = 10;
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
// (GDAL's RPCComputeTerms). Identical math on host and device.
ZT_HOST_DEVICE inline void rpc_compute_terms(double lon_n,
                                             double lat_n,
                                             double height_n,
                                             double* terms) {
    terms[0] = 1.0;
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

}  // namespace zproj::rpc
