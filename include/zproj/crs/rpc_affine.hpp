// Image-space affine bias correction for RPC sensor models, and its
// least-squares solver (the Ceres-based zproj_refine library).
//
// Satellite RPC products carry systematic errors (attitude drift,
// star-tracker misalignment, timing bias, ...). The standard remedy --
// Grodecki & Dial, "Block Adjustment of High-Resolution Satellite Images
// Described by Rational Polynomials" (PE&RS 69(1), 2003); GDAL's
// RPC_AFFINE_* transformer options; ASP's rpc solve -- overlays a small
// affine on the RPC-projected pixels, one per image:
//
//     col' = p[0] + p[1] * col + p[2] * row
//     row' = p[3] + p[4] * col + p[5] * row
//
// The affine CANNOT be folded into the 80 RPC coefficients (a linear
// combination of two rational polynomials is not a rational polynomial), so
// it lives beside the RpcInfo as a separate value -- the same design as
// GDAL's RPC_AFFINE fields and ASP's .adjust files.
//
// solve_rpc_affine() is a miniature bundle adjustment in the style of ASP's
// BaReprojErr cost: every stereo match gets a free ground block
// (lon deg, lat deg, h m) initialized by ray triangulation with the current
// affines, and the residuals run through the FORWARD RPC only
//
//     r = affine(rpc_forward(ground)) - observed_pixel            [pixels]
//
// -- never the iterative inverse, so the whole residual graph is a smooth
// rational polynomial that autodiff handles exactly. GCP observations pin
// their ground block constant.
//
// Identifiability caveat (classical block adjustment, Grodecki & Dial 2003):
// with matches only, the affines trade off against the free ground blocks
// along directions the composite ground->pixel map leaves unconstrained.
// The exact degenerate cases: a height-LINEAR pair (constant height
// coefficients) admits a uniform height shift that both images' affine
// translations absorb, and a partner with NO height dependence admits an
// infinite-dimensional family (per-point motion along its ray fibers). Real
// RPC products couple height into both images through the position-dependent
// cross terms (LH, PH, ...), which breaks these directions; GCPs or a small
// affine_prior_weight anchor the solution regardless of geometry.
//
// The solver lives in the zproj_refine library (Ceres, CPU): a per-scene
// problem of 2 x <=6 affine parameters plus 3 per match, solved once. The
// per-point math that is hot at pixel scale stays in rpc.hpp / rpc_ray.hpp
// (host/device shared).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_ray.hpp"

namespace zproj::crs {

// Index constants into RpcAffine::p (the Ceres parameter-block layout).
inline constexpr int kRpcAffineE0 = 0;
inline constexpr int kRpcAffineE1 = 1;
inline constexpr int kRpcAffineE2 = 2;
inline constexpr int kRpcAffineF0 = 3;
inline constexpr int kRpcAffineF1 = 4;
inline constexpr int kRpcAffineF2 = 5;

// The 6-parameter image-space affine correction. Identity by default.
struct RpcAffine {
    // col' = p[kRpcAffineE0] + p[kRpcAffineE1] * col + p[kRpcAffineE2] * row
    // row' = p[kRpcAffineF0] + p[kRpcAffineF1] * col + p[kRpcAffineF2] * row
    std::array<double, 6> p{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    static constexpr RpcAffine Identity() { return RpcAffine{}; }

    // Determinant of the 2x2 linear part; zero means Unapply is impossible.
    double det() const {
        return (p[kRpcAffineE1] * p[kRpcAffineF2]) -
               (p[kRpcAffineE2] * p[kRpcAffineF1]);
    }

    // RPC pixel -> corrected pixel.
    ZT_HOST_DEVICE void Apply(double col,
                              double row,
                              double& out_col,
                              double& out_row) const {
        out_col =
            p[kRpcAffineE0] + (p[kRpcAffineE1] * col) + (p[kRpcAffineE2] * row);
        out_row =
            p[kRpcAffineF0] + (p[kRpcAffineF1] * col) + (p[kRpcAffineF2] * row);
    }

    // Corrected pixel -> RPC pixel (the inverse affine). Returns false when
    // the linear part is singular.
    ZT_HOST_DEVICE bool Unapply(double col,
                                double row,
                                double& out_col,
                                double& out_row) const {
        const double d = det();
        if (d == 0.0) {
            return false;
        }
        const double c = col - p[kRpcAffineE0];
        const double r = row - p[kRpcAffineF0];
        out_col = (p[kRpcAffineF2] * c - p[kRpcAffineE2] * r) / d;
        out_row = (p[kRpcAffineE1] * r - p[kRpcAffineF1] * c) / d;
        return true;
    }
};

// How many affine parameters per image the solver floats; the enum value is
// the parameter count.
enum class RpcAffineDoF {
    Translation = 2,       // e0, f0
    TranslationScale = 4,  // + e1, f1
    Full = 6,              // + e2, f2 (shear/cross terms)
};

// One stereo tie point: the same ground feature measured in both images.
struct RpcMatch {
    double left_col = 0.0;
    double left_row = 0.0;
    double right_col = 0.0;
    double right_row = 0.0;
};

// One ground control point: known ground position plus its measured pixel.
struct RpcGcp {
    double lon = 0.0;     // degrees
    double lat = 0.0;     // degrees
    double height = 0.0;  // metres above the ellipsoid (RPC model units)
    double col = 0.0;
    double row = 0.0;
};

struct RpcAffineOptions {
    // Parameterization per image.
    RpcAffineDoF dof = RpcAffineDoF::Full;
    // Residual normalization in pixels (ASP's pixel_sigma): residuals are
    // divided by this before the loss function, so robust_threshold_px keeps
    // its pixel meaning for any sigma.
    double pixel_sigma = 1.0;
    // Huber threshold in pixels; <= 0 disables the robust loss.
    double robust_threshold_px = 0.0;
    // Tikhonov weight pulling each affine towards identity. 0 disables.
    // Recommended for noisy matches-only solves: without an anchor, the
    // weakly observable height-vs-translation gauge lets Levenberg-Marquardt
    // drift far while barely changing the cost (values ~1e-2..1 work well;
    // GCPs anchor the same direction exactly).
    double affine_prior_weight = 0.0;
    // Levenberg-Marquardt iteration cap.
    int max_iterations = 100;
    // Worker threads for the solve (Ceres parallelizes the Jacobian build
    // and the Schur elimination across residual blocks). 0 = one thread per
    // hardware thread.
    int num_threads = 0;
    // Ceres minimizer progress to stdout.
    bool verbose = false;
};

struct RpcAffineReport {
    // True when the problem was assembled and Ceres returned a usable
    // solution (the affines were updated). False leaves them untouched.
    bool ok = false;
    int num_matches = 0;          // matches used (after skipping failures)
    int num_matches_skipped = 0;  // matches dropped (triangulation failed)
    int num_gcps = 0;             // GCP observations across both images
    // RMS reprojection error per pixel component, over every observation
    // (matches in both images + GCPs), before/after the solve. Computed
    // without the robust loss, so outliers always show up here.
    double rms_before_px = 0.0;
    double rms_after_px = 0.0;
    // Failure reason or the Ceres brief report.
    std::string message;
};

// Two-view solve: float both images' affines against stereo matches and/or
// per-image GCPs. The incoming affines are the initial guess (identity for a
// fresh solve); they are overwritten only on success.
RpcAffineReport solve_rpc_affine(const RpcInfo& left,
                                 const RpcInfo& right,
                                 const std::vector<RpcMatch>& matches,
                                 const std::vector<RpcGcp>& left_gcps,
                                 const std::vector<RpcGcp>& right_gcps,
                                 RpcAffine& left_affine,
                                 RpcAffine& right_affine,
                                 const RpcAffineOptions& options = {});

// Single-image solve: float one affine against GCPs only.
RpcAffineReport solve_rpc_affine(const RpcInfo& info,
                                 const std::vector<RpcGcp>& gcps,
                                 RpcAffine& affine,
                                 const RpcAffineOptions& options = {});

// Forward RPC evaluation followed by the affine correction (degrees / metres
// -> corrected pixels). Identical math on host and device.
ZT_HOST_DEVICE inline void rpc_forward_point_affine(const RpcInfo& info,
                                                    const RpcAffine& aff,
                                                    double lon,
                                                    double lat,
                                                    double height,
                                                    double& col,
                                                    double& row) {
    double raw_col = 0.0;
    double raw_row = 0.0;
    rpc_forward_point(info, lon, lat, height, raw_col, raw_row);
    aff.Apply(raw_col, raw_row, col, row);
}

// Viewing ray through an affine-corrected pixel: un-correct the pixel, then
// the plain rpc_ray back-projection. Same conventions as rpc_ray; also
// returns false for a singular affine.
ZT_HOST_DEVICE inline bool rpc_ray_affine(const RpcInfo& info,
                                          const RpcInverseInit& init,
                                          const RpcAffine& aff,
                                          double col,
                                          double row,
                                          double h_low,
                                          double h_high,
                                          RpcRay& ray) noexcept {
    double raw_col = 0.0;
    double raw_row = 0.0;
    if (!aff.Unapply(col, row, raw_col, raw_row)) {
        return false;
    }
    return rpc_ray(info, init, raw_col, raw_row, h_low, h_high, ray);
}

}  // namespace zproj::crs
