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
// affines, and the residuals run through the FORWARD RPC only//
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
// affine_prior_weight anchor the solution regardless of geometry. For
// matches-only solves, zero_mean_affines removes the common-mode gauge
// exactly (the classical relative adjustment), and per-match DEM heights
// (RpcMatch::height) pin the height direction -- together they make a
// no-GCP solve drift-free.
//
// The solver lives in the zproj_refine library (Ceres, CPU): a per-scene
// problem of 2 x <=6 affine parameters plus 3 per match, solved once. The
// per-point math that is hot at pixel scale stays in rpc.hpp / rpc_ray.hpp
// (host/device shared). solve_rpc_bundle_adjust() (rpc_bundle_adjust.hpp)
// generalizes the same formulation to N views over a control network.
#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
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

// How a banded affine blends its band shifts across rows (the correction
// basis shared by the banded solver and RpcAffineBanded below).
enum class RpcAffineBandBasis {
    // One-hot: the band whose [row_lo, row_hi) contains the row takes the
    // full shift (nearest band outside the covered range). Piecewise-
    // constant correction; jumps at band boundaries.
    Constant,
    // Tent: linear interpolation between the two nearest band centers
    // (constant extension outside them). Piecewise-LINEAR, C0-continuous
    // -- no seams. The usual choice for smooth row-dependent error.
    Linear,
};

// The consumer-side composite correction of solve_rpc_bundle_adjust_banded
// (rpc_bundle_adjust.hpp): one RpcAffine plus per-band translation shifts,
// ready to apply to pixels -- downstream code consumes THIS instead of
// juggling the scene affine, the band table and the shift table and
// re-implementing the blending.
//
//     col' = (e0 + dx_band(row)) + e1*col + e2*row
//     row' = (f0 + dy_band(row)) + f1*col + f2*row
//
// Fixed-size storage (kMaxBands, the same cap convention as the CUDA
// N-view kernels) keeps the struct a plain-data value that device kernels
// can take by value, like RpcAffine. Bands are stored sorted by row
// center at Make() time (host); the blending methods are host/device
// shared arithmetic. Compose with the existing machinery through
// EffectiveAffine:
//
//     corrected pixel:            baff.Apply(col, row, c2, r2);
//     forward through correction: rpc_forward_point_affine(
//                                     info, baff.EffectiveAffine(row), ...);
//     back-projection / rays:     rpc_ray_affine(
//                                     info, init, baff.EffectiveAffine(row),
//                                     col, row, h_lo, h_hi, ray);
//
// The EffectiveAffine row for BACK-projection is the observed pixel's row:
// the shift varies slowly with row, so the slope-times-shift error of
// un-applying at the observed row rather than the exact pre-correction row
// is milli-pixel scale.
//
// Uniqueness note: on the band-center span the composite correction is
// uniquely determined by the solve, but in the constant-extension zones
// beyond the outermost centers an affine tilt trades against per-point
// ground gradients -- there the composite is only as good as the local
// data anchor. Keep the outer band centers near the scene's row edges
// (uniform tiling already does: the extension zones are half a band wide).
class RpcAffineBanded {
public:
    // One band's correction: rows [row_lo, row_hi) shift by (dx, dy) px.
    struct Band {
        double row_lo = 0.0;
        double row_hi = 0.0;
        double dx = 0.0;
        double dy = 0.0;
    };

    // Storage cap (bands per scene); raise if a scene ever needs finer
    // banding. 32 uniform bands over a 50000-line scene is ~1560 lines per
    // band -- far finer than any practical wave.
    static constexpr int kMaxBands = 32;

    // Identity affine with no bands.
    RpcAffineBanded() = default;

    // Host-side assembly: takes the bands in any order (sorted by row
    // center internally), validates the count against kMaxBands. Nullopt
    // on overflow.
    static std::optional<RpcAffineBanded> Make(const RpcAffine& affine,
                                               std::vector<Band> bands,
                                               RpcAffineBandBasis basis);

    ZT_HOST_DEVICE int num_bands() const noexcept { return num_bands_; }
    ZT_HOST_DEVICE const RpcAffine& affine() const noexcept { return affine_; }
    ZT_HOST_DEVICE RpcAffineBandBasis basis() const noexcept { return basis_; }
    ZT_HOST_DEVICE const Band& band(int i) const noexcept {
        return bands_[static_cast<std::size_t>(i)];
    }

    // The blended shift at one row, per the basis: the containing (else
    // nearest) band under Constant, the tent interpolation between the two
    // bracketing centers (constant extension outside them) under Linear.
    // Zero when there are no bands.
    ZT_HOST_DEVICE void ShiftAt(double row, double& dx, double& dy) const {
        dx = 0.0;
        dy = 0.0;
        if (num_bands_ == 0) {
            return;
        }
        if (basis_ == RpcAffineBandBasis::Constant) {
            int pick = -1;
            for (int i = 0; i < num_bands_; ++i) {
                const Band& b = bands_[static_cast<std::size_t>(i)];
                if (row >= b.row_lo && row < b.row_hi) {
                    pick = i;
                    break;
                }
            }
            if (pick < 0) {
                // Outside every band: nearest center.
                double best = HUGE_VAL;
                for (int i = 0; i < num_bands_; ++i) {
                    const double d = fabs(Center(i) - row);
                    if (d < best) {
                        best = d;
                        pick = i;
                    }
                }
            }
            dx = bands_[static_cast<std::size_t>(pick)].dx;
            dy = bands_[static_cast<std::size_t>(pick)].dy;
            return;
        }
        // Linear (tent) basis over the center-sorted bands.
        int j = 0;
        while (j < num_bands_ && Center(j) < row) {
            ++j;
        }
        if (j == 0 || j == num_bands_) {
            // Constant extension outside the center span.
            const Band& b =
                bands_[static_cast<std::size_t>(j == 0 ? 0 : num_bands_ - 1)];
            dx = b.dx;
            dy = b.dy;
            return;
        }
        const int i = j - 1;
        const double span = Center(j) - Center(i);
        if (!(span > 0.0)) {
            const Band& b = bands_[static_cast<std::size_t>(j)];
            dx = b.dx;
            dy = b.dy;
            return;
        }
        const double w = (Center(j) - row) / span;
        const Band& lo = bands_[static_cast<std::size_t>(i)];
        const Band& hi = bands_[static_cast<std::size_t>(j)];
        dx = (w * lo.dx) + ((1.0 - w) * hi.dx);
        dy = (w * lo.dy) + ((1.0 - w) * hi.dy);
    }

    // The affine with the shift at `row` folded into the translations --
    // the composition point with rpc_forward_point_affine / rpc_ray_affine.
    ZT_HOST_DEVICE RpcAffine EffectiveAffine(double row) const {
        RpcAffine eff = affine_;
        double dx = 0.0;
        double dy = 0.0;
        ShiftAt(row, dx, dy);
        eff.p[0] += dx;
        eff.p[3] += dy;
        return eff;
    }

    // RPC pixel -> corrected pixel: the affine, then the blended shift.
    ZT_HOST_DEVICE void Apply(double col,
                              double row,
                              double& out_col,
                              double& out_row) const {
        affine_.Apply(col, row, out_col, out_row);
        double dx = 0.0;
        double dy = 0.0;
        ShiftAt(row, dx, dy);
        out_col += dx;
        out_row += dy;
    }

private:
    ZT_HOST_DEVICE double Center(int i) const {
        const Band& b = bands_[static_cast<std::size_t>(i)];
        return 0.5 * (b.row_lo + b.row_hi);
    }

    RpcAffine affine_;
    RpcAffineBandBasis basis_ = RpcAffineBandBasis::Linear;
    int num_bands_ = 0;
    std::array<Band, static_cast<std::size_t>(kMaxBands)> bands_{};
};

inline std::optional<RpcAffineBanded> RpcAffineBanded::Make(
    const RpcAffine& affine,
    std::vector<Band> bands,
    RpcAffineBandBasis basis) {
    if (bands.size() > static_cast<std::size_t>(kMaxBands)) {
        return std::nullopt;
    }
    std::sort(bands.begin(), bands.end(), [](const Band& a, const Band& b) {
        return (a.row_lo + a.row_hi) < (b.row_lo + b.row_hi);
    });
    RpcAffineBanded out;
    out.affine_ = affine;
    out.basis_ = basis;
    out.num_bands_ = static_cast<int>(bands.size());
    for (std::size_t i = 0; i < bands.size(); ++i) {
        out.bands_[i] = bands[i];
    }
    return out;
}

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
    // Optional height above the ellipsoid [m]. NaN (the default) leaves the
    // match's ground height free; a finite value pins it -- e.g. sampled
    // from a coarse DEM -- which removes the height-vs-translation gauge
    // without any GCP (see RpcAffineOptions::zero_mean_affines for the
    // companion constraint on the common mode).
    double height = std::numeric_limits<double>::quiet_NaN();
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
    // GCPs anchor the same direction exactly). With zero_mean_affines the
    // prior acts on the differential parameters only.
    double affine_prior_weight = 0.0;
    // Two-view only: constrain the two affines' MEAN to the identity (an
    // exact constraint, implemented by parameterizing the right image's
    // affine as the mirror of the left's). This is the classical relative
    // adjustment: without ground reference, matches determine only the
    // DIFFERENTIAL correction between the images, while the common mode
    // (uniform ground shift absorbed by both images' translations) is an
    // exact gauge that otherwise soaks up noise as drift. Pinning the mean
    // to "zero correction" removes it; the returned affines are then a
    // convention (differential split symmetrically), not absolute
    // corrections. Combine with per-match DEM heights (RpcMatch::height)
    // to also pin the weakly observable height direction, or with GCPs --
    // which anchor both exactly and make this flag unnecessary.
    bool zero_mean_affines = false;
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
