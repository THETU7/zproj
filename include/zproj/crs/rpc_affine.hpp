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

// How a gridded affine blends its cell shifts across the (col, row) grid
// (the correction basis shared by the gridded solver and RpcAffineGridded
// below).
enum class RpcAffineGridBasis {
    // One-hot: the cell whose [col_lo, col_hi) x [row_lo, row_hi) contains
    // the pixel takes the full shift (nearest center outside the covered
    // range). Piecewise-constant correction; jumps at cell boundaries.
    // The shape of detector-array stitching errors (CCD segment offsets).
    Constant,
    // Tensor-product tent: bilinear interpolation between the four
    // bracketing cell centers (constant extension outside the center
    // spans, per axis). Piecewise-bilinear, C0-continuous -- no seams.
    // The usual choice for smooth 2D-dependent error.
    Linear,
};

// The consumer-side composite correction of solve_rpc_bundle_adjust_
// gridded (rpc_bundle_adjust.hpp): one RpcAffine plus per-cell
// translation shifts over a 2D column x row grid, ready to apply to
// pixels -- downstream code consumes THIS instead of juggling the scene
// affine, the cell table and the shift table and re-implementing the
// blending.
//
//     col' = (e0 + dx_cell(col, row)) + e1*col + e2*row
//     row' = (f0 + dy_cell(col, row)) + f1*col + f2*row
//
// A one-row grid (MakeUniform) degenerates to pure column banding: the
// row axis carries no structure.
//
// Fixed-size storage (kMaxCells, the same cap convention as the CUDA
// N-view kernels) keeps the struct a plain-data value that device kernels
// can take by value, like RpcAffine. Cells are stored in canonical
// row-major order (sorted by row center, then col center) at Make() time
// (host); the blending methods are host/device shared arithmetic. The
// Linear basis additionally requires a REGULAR tensor grid (every row
// carrying the same column centers) -- Make validates and rejects
// anything else, and MakeUniform2D always produces one. Compose with the
// existing machinery through EffectiveAffine:
//
//     corrected pixel:            grid.Apply(col, row, c2, r2);
//     forward through correction: rpc_forward_point_affine(
//                                     info, grid.EffectiveAffine(col, row),
//                                     ...);
//     back-projection / rays:     rpc_ray_affine(
//                                     info, init,
//                                     grid.EffectiveAffine(col, row),
//                                     col, row, h_lo, h_hi, ray);
//
// The EffectiveAffine (col, row) for BACK-projection are the observed
// pixel's: the shift varies slowly, so the slope-times-shift error of
// un-applying at the observed pixel rather than the exact pre-correction
// one is milli-pixel scale.
//
// Uniqueness note: on the cell-center spans the composite correction is
// uniquely determined by the solve, but in the constant-extension zones
// beyond the outermost centers an affine tilt trades against per-point
// ground gradients -- there the composite is only as good as the local
// data anchor. Keep the outer cell centers near the scene's edges
// (uniform tiling already does: the extension zones are half a cell
// wide).
class RpcAffineGridded {
public:
    // One cell's correction: pixels in [col_lo, col_hi) x [row_lo, row_hi)
    // shift by (dx, dy) px.
    struct Cell {
        double col_lo = 0.0;
        double col_hi = 0.0;
        double row_lo = 0.0;
        double row_hi = 0.0;
        double dx = 0.0;
        double dy = 0.0;
    };

    // Storage cap (cells per scene); raise if a scene ever needs a finer
    // grid. 64 cells covers an 8x8 grid -- over a ~30000 x 50000 px
    // scene that is finer than any practical 2D systematic structure.
    static constexpr int kMaxCells = 64;

    // Identity affine with no cells.
    RpcAffineGridded() = default;

    // Host-side assembly: takes the cells in any order (sorted into
    // canonical row-major order internally), validates the count against
    // kMaxCells, every cell's ranges, and (under Linear) the regular
    // tensor structure. Nullopt on violation.
    static std::optional<RpcAffineGridded> Make(const RpcAffine& affine,
                                                std::vector<Cell> cells,
                                                RpcAffineGridBasis basis);

    // Uniform column banding: `num_col` equal-width zero-shift cells
    // tiling [col_lo, col_hi) around `affine`, the row axis degenerate
    // (one row band; the shift does not depend on row) -- the
    // cross-track-only model and the fresh-solve starting point for
    // solve_rpc_bundle_adjust_gridded. Nullopt on the same violations.
    static std::optional<RpcAffineGridded> MakeUniform(
        const RpcAffine& affine,
        double col_lo,
        double col_hi,
        int num_col,
        RpcAffineGridBasis basis);

    // Uniform 2D convenience: a `num_col` x `num_row` grid of equal
    // zero-shift cells tiling [col_lo, col_hi) x [row_lo, row_hi) around
    // `affine` -- the fresh-solve starting point for the 2D model.
    static std::optional<RpcAffineGridded> MakeUniform2D(
        const RpcAffine& affine,
        double col_lo,
        double col_hi,
        double row_lo,
        double row_hi,
        int num_col,
        int num_row,
        RpcAffineGridBasis basis);

    ZT_HOST_DEVICE int num_cells() const noexcept { return num_cells_; }
    ZT_HOST_DEVICE const RpcAffine& affine() const noexcept { return affine_; }
    ZT_HOST_DEVICE RpcAffineGridBasis basis() const noexcept { return basis_; }
    ZT_HOST_DEVICE const Cell& cell(int i) const noexcept {
        return cells_[static_cast<std::size_t>(i)];
    }
    // Cells per grid row of the canonical (regular, row-major) layout.
    ZT_HOST_DEVICE int num_cols() const noexcept { return num_cols_; }

    // Host-side mutators, the in/out path of solve_rpc_bundle_adjust_
    // gridded (grid structure stays fixed; only the values move).
    // set_cell_shift indexes cells in the stored canonical order.
    void set_affine(const RpcAffine& affine) noexcept { affine_ = affine; }
    void set_cell_shift(int i, double dx, double dy) noexcept {
        Cell& c = cells_[static_cast<std::size_t>(i)];
        c.dx = dx;
        c.dy = dy;
    }

    // The blended shift at one pixel, per the basis: the containing (else
    // nearest-center) cell under Constant, the bilinear interpolation
    // between the four bracketing centers (constant extension outside
    // the center spans, per axis) under Linear. Zero when there are no
    // cells.
    ZT_HOST_DEVICE void ShiftAt(double col,
                                double row,
                                double& dx,
                                double& dy) const {
        dx = 0.0;
        dy = 0.0;
        if (num_cells_ == 0) {
            return;
        }
        if (basis_ == RpcAffineGridBasis::Constant) {
            int pick = -1;
            for (int i = 0; i < num_cells_; ++i) {
                const Cell& c = cells_[static_cast<std::size_t>(i)];
                if (col >= c.col_lo && col < c.col_hi && row >= c.row_lo &&
                    row < c.row_hi) {
                    pick = i;
                    break;
                }
            }
            if (pick < 0) {
                // Outside every cell: nearest center (pixel-space 2D
                // distance).
                double best = HUGE_VAL;
                for (int i = 0; i < num_cells_; ++i) {
                    const double dc = ColCenter(i) - col;
                    const double dr = RowCenter(i) - row;
                    const double d = (dc * dc) + (dr * dr);
                    if (d < best) {
                        best = d;
                        pick = i;
                    }
                }
            }
            dx = cells_[static_cast<std::size_t>(pick)].dx;
            dy = cells_[static_cast<std::size_t>(pick)].dy;
            return;
        }
        // Linear (tensor tent) basis over the regular row-major grid:
        // bracket each axis by its centers (constant extension outside a
        // center span collapses to the boundary cell), then blend the up
        // to four bracketing cells with the tent weights.
        int ic = 0;
        int jc = 0;
        double wci = 1.0;
        double wcj = 0.0;
        if (num_cols_ > 1) {
            while (jc < num_cols_ && ColCenter(jc) < col) {
                ++jc;
            }
            if (jc == 0 || jc == num_cols_) {
                ic = jc = (jc == 0) ? 0 : num_cols_ - 1;
                wci = 1.0;
                wcj = 0.0;
            } else {
                ic = jc - 1;
                const double span = ColCenter(jc) - ColCenter(ic);
                if (!(span > 0.0)) {
                    ic = jc;
                    wci = 1.0;
                    wcj = 0.0;
                } else {
                    wci = (ColCenter(jc) - col) / span;
                    wcj = 1.0 - wci;
                }
            }
        }
        const int rows = num_cells_ / num_cols_;
        int ir = 0;
        int jr = 0;
        double wri = 1.0;
        double wrj = 0.0;
        if (rows > 1) {
            while (jr < rows && RowCenter(jr * num_cols_) < row) {
                ++jr;
            }
            if (jr == 0 || jr == rows) {
                ir = jr = (jr == 0) ? 0 : rows - 1;
                wri = 1.0;
                wrj = 0.0;
            } else {
                ir = jr - 1;
                const double span =
                    RowCenter(jr * num_cols_) - RowCenter(ir * num_cols_);
                if (!(span > 0.0)) {
                    ir = jr;
                    wri = 1.0;
                    wrj = 0.0;
                } else {
                    wri = (RowCenter(jr * num_cols_) - row) / span;
                    wrj = 1.0 - wri;
                }
            }
        }
        const int idx[2][2] = {{ir * num_cols_ + ic, ir * num_cols_ + jc},
                               {jr * num_cols_ + ic, jr * num_cols_ + jc}};
        const double w[2][2] = {{wri * wci, wri * wcj}, {wrj * wci, wrj * wcj}};
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                const Cell& c = cells_[static_cast<std::size_t>(idx[a][b])];
                dx += w[a][b] * c.dx;
                dy += w[a][b] * c.dy;
            }
        }
    }

    // The affine with the shift at (col, row) folded into the
    // translations -- the composition point with rpc_forward_point_affine
    // / rpc_ray_affine.
    ZT_HOST_DEVICE RpcAffine EffectiveAffine(double col, double row) const {
        RpcAffine eff = affine_;
        double dx = 0.0;
        double dy = 0.0;
        ShiftAt(col, row, dx, dy);
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
        ShiftAt(col, row, dx, dy);
        out_col += dx;
        out_row += dy;
    }

private:
    ZT_HOST_DEVICE double ColCenter(int i) const {
        const Cell& c = cells_[static_cast<std::size_t>(i)];
        return 0.5 * (c.col_lo + c.col_hi);
    }
    ZT_HOST_DEVICE double RowCenter(int i) const {
        const Cell& c = cells_[static_cast<std::size_t>(i)];
        return 0.5 * (c.row_lo + c.row_hi);
    }

    RpcAffine affine_;
    RpcAffineGridBasis basis_ = RpcAffineGridBasis::Linear;
    int num_cells_ = 0;
    int num_cols_ = 1;  // cells per grid row (canonical tensor layout)
    std::array<Cell, static_cast<std::size_t>(kMaxCells)> cells_{};
};

inline std::optional<RpcAffineGridded> RpcAffineGridded::Make(
    const RpcAffine& affine,
    std::vector<Cell> cells,
    RpcAffineGridBasis basis) {
    if (cells.size() > static_cast<std::size_t>(kMaxCells)) {
        return std::nullopt;
    }
    for (const Cell& c : cells) {
        if (!(c.col_lo < c.col_hi) || !(c.row_lo < c.row_hi)) {
            return std::nullopt;
        }
    }
    const auto RowKey = [](const Cell& c) { return c.row_lo + c.row_hi; };
    const auto ColKey = [](const Cell& c) { return c.col_lo + c.col_hi; };
    std::sort(cells.begin(), cells.end(), [&](const Cell& a, const Cell& b) {
        if (RowKey(a) != RowKey(b)) {
            return RowKey(a) < RowKey(b);
        }
        return ColKey(a) < ColKey(b);
    });
    // Detect the tensor layout: cells per grid row = the leading run
    // sharing a row key. Linear requires every row to carry the same
    // column keys (exact doubles -- MakeUniform2D's arithmetic is
    // row-invariant); Constant accepts any layout and never reads
    // num_cols_.
    std::size_t num_col = 0;
    while (num_col < cells.size() &&
           RowKey(cells[num_col]) == RowKey(cells[0])) {
        ++num_col;
    }
    bool regular = (cells.size() % num_col) == 0;
    if (regular && basis == RpcAffineGridBasis::Linear) {
        const std::size_t rows = cells.size() / num_col;
        for (std::size_t r = 0; r < rows && regular; ++r) {
            for (std::size_t c = 0; c < num_col; ++c) {
                if (RowKey(cells[r * num_col + c]) !=
                        RowKey(cells[r * num_col]) ||
                    ColKey(cells[r * num_col + c]) != ColKey(cells[c])) {
                    regular = false;
                    break;
                }
            }
        }
    }
    if (!regular && basis == RpcAffineGridBasis::Linear) {
        return std::nullopt;
    }
    RpcAffineGridded out;
    out.affine_ = affine;
    out.basis_ = basis;
    out.num_cells_ = static_cast<int>(cells.size());
    out.num_cols_ = static_cast<int>(num_col);
    for (std::size_t i = 0; i < cells.size(); ++i) {
        out.cells_[i] = cells[i];
    }
    return out;
}

inline std::optional<RpcAffineGridded> RpcAffineGridded::MakeUniform(
    const RpcAffine& affine,
    double col_lo,
    double col_hi,
    int num_col,
    RpcAffineGridBasis basis) {
    // Degenerate row axis: one row band [0, 1) -- the row coordinate
    // never changes the shift (it clamps to the single row center under
    // Linear, and Constant's containment/nearest gives the same cell).
    return MakeUniform2D(affine, col_lo, col_hi, 0.0, 1.0, num_col, 1, basis);
}

inline std::optional<RpcAffineGridded> RpcAffineGridded::MakeUniform2D(
    const RpcAffine& affine,
    double col_lo,
    double col_hi,
    double row_lo,
    double row_hi,
    int num_col,
    int num_row,
    RpcAffineGridBasis basis) {
    if (num_col < 1 || num_row < 1 || num_col > kMaxCells ||
        num_row > kMaxCells || num_col * num_row > kMaxCells ||
        !(col_lo < col_hi) || !(row_lo < row_hi)) {
        return std::nullopt;
    }
    std::vector<Cell> cells(static_cast<std::size_t>(num_col * num_row));
    for (int r = 0; r < num_row; ++r) {
        for (int c = 0; c < num_col; ++c) {
            cells[static_cast<std::size_t>(r * num_col + c)] =
                Cell{col_lo + (col_hi - col_lo) * static_cast<double>(c) /
                                  static_cast<double>(num_col),
                     col_lo + (col_hi - col_lo) * static_cast<double>(c + 1) /
                                  static_cast<double>(num_col),
                     row_lo + (row_hi - row_lo) * static_cast<double>(r) /
                                  static_cast<double>(num_row),
                     row_lo + (row_hi - row_lo) * static_cast<double>(r + 1) /
                                  static_cast<double>(num_row),
                     0.0,
                     0.0};
        }
    }
    return Make(affine, std::move(cells), basis);
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

// The robust loss shape applied to the pixel residuals when
// robust_threshold_px > 0. All thresholds are in pixels (residuals are
// normalized by pixel_sigma first, the threshold with them). Huber is the
// classical photogrammetric default (ASP's --robust-threshold, COLMAP's
// reprojection loss); the redescending Cauchy/Tukey shapes down-weight
// large residuals progressively harder -- Tukey's influence reaches zero,
// which suits heavy mismatch noise at the cost of slightly biasing the
// fit towards its current basin (start from a loose-threshold pass, or
// the median warm starts, so the basin is the right one).
enum class RpcLossKind {
    None,    // Plain least squares (also the effect of threshold <= 0).
    Huber,   // Linear beyond the threshold; bounded, never-zero influence.
    Cauchy,  // Lorentzian; ~1/r influence decay.
    Tukey,   // Biweight; influence exactly zero beyond ~threshold*2.34.
};

struct RpcAffineOptions {
    // Parameterization per image.
    RpcAffineDoF dof = RpcAffineDoF::Full;
    // Residual normalization in pixels (ASP's pixel_sigma): residuals are
    // divided by this before the loss function, so robust_threshold_px keeps
    // its pixel meaning for any sigma.
    double pixel_sigma = 1.0;
    // Robust-loss threshold in pixels; <= 0 disables the robust loss.
    double robust_threshold_px = 0.0;
    // The loss SHAPE robust_threshold_px arms (see RpcLossKind).
    RpcLossKind loss_kind = RpcLossKind::Huber;
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
