// Multi-view (N >= 2) satellite bundle adjustment over a control network of
// RPC + image-space affine cameras -- the N-view generalization of the
// two-view solve_rpc_affine() (rpc_affine.hpp), structured after ASP's
// bundle_adjust and VisionWorkbench's ControlNetwork.
//
// Camera model: the same 6-parameter image-space affine on the RPC-projected
// pixels as the two-view solver (Grodecki & Dial 2003; GDAL's RPC_AFFINE
// fields; IPOL's "Generic Bundle Adjustment for indirect RPC" formulation).
// ASP's current bundle_adjust instead floats a 6-parameter ECEF rigid
// correction (translation + axis-angle rotation, AdjustedCameraModel) in
// GROUND space for RPC cameras -- the image-space affine is the classical
// block-adjustment formulation, composes with the RpcAffine already used
// throughout zproj (triangulation through rpc_ray_affine, the two-view
// solver), and is what GDAL's transformer actually accepts.
//
// Control network (VW's ControlMeasure / ControlPoint): one RpcBaPoint is a
// ground feature measured in >= 2 images (a tie point), or a point with
// KNOWN ground plus its pixel measures (a GCP, ground_fixed). ASP builds the
// same structure from interest-point tracks (build_control_network); any
// multi-view match pipeline can fill it in.
//
// Formulation: identical miniature bundle adjustment to the two-view case.
// Every tie point gets a free ground block (lon deg, lat deg, h m)
// initialized by N-view ray triangulation with the CURRENT affines
// (triangulate_nview, the Slabaugh normal equations -- VW
// StereoModel::triangulate_point), and every measure residuals through the
// FORWARD RPC only (never the iterative inverse), so the whole residual
// graph stays a smooth rational polynomial that autodiff handles exactly:
//
//     r = affine_v(rpc_forward_v(ground)) - observed_pixel_v    [pixels]
//
// A GCP's ground block is constant (ASP's --fix-gcp_xyz mode); its pixel
// measures then pin the cameras' affines ABSOLUTELY against known ground.
// A tie point with a finite height (e.g. sampled from a coarse DEM) holds
// its ground height constant, which pins the weakly observable
// height-vs-translation gauge without any GCP (same as RpcMatch::height).
//
// Gauge (the N-view generalization of the two-view zero_mean_affines): with
// matches only, the network's common mode -- the average affine, the
// direction every camera can drift together while the free ground blocks
// absorb it -- is (near-)unobservable. zero_mean_affines removes it exactly
// by construction: views 0..V-2 float free, and the last view's affine is
// DERIVED as
//
//     p_{V-1} = V * identity - sum_{i < V-1} p_i      (sum p_i = V*identity)
//
// -- the two-view mirror (2*identity - p) generalized to N. The returned
// affines are then a symmetric convention (differential split with zero
// mean), not absolute corrections, exactly as in the two-view case. GCPs
// anchor the common mode absolutely and make the flag unnecessary. Unlike
// ASP (which anchors no-GCP solves with a soft GSD-scaled prior on the
// triangulated points, --tri-weight), the gauge is removed exactly, and the
// affine_prior_weight acts as the soft anchor.
//
// Solver: Ceres Levenberg-Marquardt with DENSE_SCHUR -- the same
// cameras-vs-points sparsity ASP exploits (it uses DENSE_SCHUR below ~100
// cameras) -- plus an optional Huber robust loss on the pixel residuals.
// Each residual couples one ground block to one affine (the derived-view
// residuals couple it to V-1), so the Schur complement eliminates the 3
// million-point blocks and solves the <= 6V x 6V camera system.
#pragma once

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_affine.hpp"

namespace zproj::crs {

// One measurement of a control point in one image -- VW's ControlMeasure
// (image id + pixel position). `view` indexes into solve_rpc_bundle_adjust's
// views vector; the measures of one point must come from distinct views.
struct RpcBaMeasure {
    int view = 0;
    double col = 0.0;
    double row = 0.0;
};

// One control point -- VW's ControlPoint. A tie point carries >= 2 measures
// and a free ground block; a GCP (ground_fixed) additionally carries its
// KNOWN ground position, held constant through the solve.
struct RpcBaPoint {
    std::vector<RpcBaMeasure> measures;

    // True for a ground control point: lon/lat/height below are survey
    // truth, the ground block is constant, and the measures pin the viewing
    // affines absolutely. False for a tie point.
    bool ground_fixed = false;

    // Ground position. For a GCP all three are required (a non-finite value
    // invalidates the point). For a tie point only `height` is meaningful:
    // NaN (the default) leaves the ground height free, while a finite value
    // (e.g. sampled from a coarse DEM) holds it constant -- the height-gauge
    // pin of the two-view RpcMatch::height, generalized to N views.
    double lon = 0.0;                                          // degrees
    double lat = 0.0;                                          // degrees
    double height = std::numeric_limits<double>::quiet_NaN();  // metres
};

struct RpcBaOptions {
    // Parameterization per view (same meanings as RpcAffineOptions).
    RpcAffineDoF dof = RpcAffineDoF::Full;
    // Residual normalization in pixels (ASP's pixel_sigma).
    double pixel_sigma = 1.0;
    // Robust-loss threshold in pixels; <= 0 disables the robust loss.
    double robust_threshold_px = 0.0;
    // The loss shape robust_threshold_px arms (see RpcLossKind).
    RpcLossKind loss_kind = RpcLossKind::Huber;
    // Tikhonov weight pulling each floated affine towards identity. 0
    // disables. Under zero_mean_affines it regularizes the differential
    // parameters only.
    double affine_prior_weight = 0.0;
    // Pixel-unit identity prior (a sigma in px); 0 disables. Unlike
    // affine_prior_weight -- which scales the RAW parameters and so
    // barely touches the ~1-scale linear terms -- this maps every affine
    // deviation to the pixel shift it causes at the RPC validity domain's
    // edge (translations directly, linear terms times samp_scale /
    // line_scale) before weighting. That is the deviation that matters
    // physically: a 1e-4 e1 drift is invisible to a unit-weight parameter
    // prior yet tilts a 100000 px scene edge by 10 px. Use this whenever
    // the RPC absolute georeferencing is trusted to a few pixels
    // (typical for satellite products, unlike aerial blocks that need a
    // fully free adjustment): e.g. 2.0 px keeps the solve near identity
    // while still absorbing genuine few-pixel biases. Recommended for
    // stage 1 of the two-stage gridded solve (see below).
    double identity_prior_px = 0.0;
    // Constrain the V affines' MEAN to the identity exactly by deriving the
    // last view's affine from the others (see the file comment). The
    // caller's initial affine for the last view is overruled by the
    // constraint. GCPs anchor the common mode absolutely and make this
    // unnecessary.
    bool zero_mean_affines = false;
    // Levenberg-Marquardt iteration cap.
    int max_iterations = 100;
    // Worker threads for the solve. 0 = one per hardware thread.
    int num_threads = 0;
    // Ceres minimizer progress to stdout.
    bool verbose = false;
};

struct RpcBaReport {
    // True when the problem was assembled and Ceres returned a usable
    // solution (the affines were updated). False leaves them untouched.
    bool ok = false;
    int num_points = 0;          // tie points used (after skipping failures)
    int num_points_skipped = 0;  // tie points dropped (init/validation)
    int num_measures = 0;        // pixel observations used (incl. GCP ones)
    int num_gcps = 0;            // ground-fixed points used
    // RMS reprojection error per pixel component over every used
    // observation, before/after the solve, without the robust loss.
    double rms_before_px = 0.0;
    double rms_after_px = 0.0;
    // Failure reason or the Ceres brief report.
    std::string message;
};

// N-view solve: float every view's affine against the control network of
// tie points and GCPs. `affines` sizes to views (and initializes each view;
// identity for a fresh solve) and is overwritten only on success. The
// optimized ground products are recovered downstream by re-triangulating
// the matches through the corrected affines (rpc_ray_affine +
// triangulate_nview), the same way ASP re-triangulates its control network.
RpcBaReport solve_rpc_bundle_adjust(const std::vector<RpcInfo>& views,
                                    const std::vector<RpcBaPoint>& points,
                                    std::vector<RpcAffine>& affines,
                                    const RpcBaOptions& options = {});

// ========================= gridded variant ================================
//
// solve_rpc_bundle_adjust_gridded(): one GLOBAL affine per scene plus a
// per-cell TRANSLATION shift over a 2D column x row grid -- the camera
// model for 2D-structured systematic error (cross-track detector-array
// structure -- CCD-segment stitching offsets, array-internal distortion
// -- plus any along-track variation; a per-scene affine has already
// absorbed the constant and linear parts of both). The cell shifts absorb
// the remaining structure as a piecewise-constant or piecewise-bilinear
// function of the pixel (col, row):
//
//     col' = (e0 + dx_cell) + e1*col + e2*row
//     row' = (f0 + dy_cell) + f1*col + f2*row
//
// The corrections travel as RpcAffineGridded values (rpc_affine.hpp) --
// one object per scene, in and out: pass identity affines with zero-shift
// grids for a fresh solve (RpcAffineGridded::MakeUniform2D, or
// MakeUniform for column-only banding -- a one-row grid), or previous
// solutions to warm-start. The returned objects are the complete
// downstream correction (Apply / EffectiveAffine); nothing else needs
// assembling. Each scene blends its cells per ITS object's basis, so
// Linear and Constant scenes may be mixed; a scene with NO cells is a
// plain per-view affine scene and may be mixed with gridded ones.
//
// Parameterization internals (exact, no soft constraints):
//   * The cell shifts are ZERO-MEAN within each scene: the canonically
//     last cell (bottom-right) derives its shift from the others. This
//     removes the exact degeneracy between the scene affine's
//     translation and the mean of its cell shifts. A scene with a single
//     cell has its shift pinned to zero (no structure to resolve).
//   * zero_mean_affines, when set, generalizes to "the MEAN of the scene
//     affines is the identity" (the last scene's affine is derived from
//     the others; the caller's initial affine for it is overruled).
//     Ignored under fix_scene_affines (the affines are constants).
//
// Observability caveat (stronger than the plain solver's): with matches
// only, every 2D cell region carries its own horizontal common mode --
// the cells covering that region in ALL scenes can drift together, the
// ground blocks absorb it, and zero-mean does not touch it; a region
// left without an anchor drifts by whole pixels (demonstrated in the
// gridded example). Anchor with at least one GCP per cell region, or
// with cell_shift_prior_weight (a
// Tikhonov prior that pins the drift to "no correction relative to the
// scene affine"); per-point DEM heights pin the height direction as
// usual. The cell count multiplies this exposure: prefer the coarsest
// grid that absorbs the error (the two-stage flow below usually needs
// far fewer cells than a joint solve suggests).
//
// Measures reference SCENES (RpcBaMeasure::view indexes `scenes`, not
// cells): a measure's cell follows from its pixel (col, row), so ordinary
// multi-scene match networks plug in unchanged. Pixels outside a scene's
// grid clamp to the nearest cell/center (constant extension under
// RpcAffineGridBasis::Linear).
//
// Approximation power (Linear basis): fitting a full-period wave of
// amplitude A with N cells along its axis leaves ~A*pi^2/(2N^2) px --
// N=12, A=5 px gives ~0.17 px. The corrected composite (RPC + scene
// affine + cell shift) should stay beside the RPC as separate correction
// layers; folding it back into refitted RPC coefficients re-smears the
// structure (a cubic rational polynomial cannot carry it).
//
// ======================= two-stage variant =================================
//
// solve_rpc_bundle_adjust_two_stage(): the recommended driver for
// satellite RPC refinement, where the absolute georeferencing is near
// correct and only a few pixels of local structure need absorbing --
// unlike aerial blocks, where a single fully free affine adjustment is
// the classical model. One joint affine+grid solve lets the scene
// affine drift/tilt arbitrarily (the gauge directions above are only
// zero-mean-constrained; weak anchoring lets the affine soak up local
// structure and swing whole scenes), so the decomposition is staged:
//
//   stage 1  global affine only, held NEAR IDENTITY by identity_prior_px
//            (pixel-unit deviations; set it to the trusted absolute
//            accuracy of the RPC products, e.g. 2-3 px),
//   stage 2  the affine FROZEN at stage 1's solution; only the cells'
//            translation shifts float (zero-mean per scene), absorbing
//            the remaining few-pixel local structure.
//
// Because a >=2x2 tent grid spans every affine function on the center
// span, the frozen-affine composite reaches the same fit a joint solve
// would -- the staging pins the DECOMPOSITION, not the correction. Each
// stage is also usable standalone (solve_rpc_bundle_adjust with
// identity_prior_px; solve_rpc_bundle_adjust_gridded with
// fix_scene_affines) for warm-started or custom pipelines.

// Gridded options: the plain RpcBaOptions knobs plus the grid-specific
// ones. `dof` restricts the scene affines only (the cell shifts are
// always 2-parameter translations); the blending basis lives on each
// scene's RpcAffineGridded object, not here.
struct RpcBaGridOptions : RpcBaOptions {
    // Tikhonov weight pulling each free cell shift towards zero (i.e.
    // towards the scene affine alone). 0 disables. Recommended for
    // matches-only gridded solves (see the observability caveat above)
    // and for cells weakly covered by matches or GCPs.
    double cell_shift_prior_weight = 0.0;
    // Hold every scene's affine CONSTANT at the incoming value and float
    // only the cell shifts -- stage 2 of the two-stage flow. The affine
    // prior knobs and zero_mean_affines are then moot (ignored).
    bool fix_scene_affines = false;
    // ---- robustness knobs (see solve_rpc_bundle_adjust_robust) --------
    //
    // Stage-1 (global affine) overrides for the two-stage driver:
    // nullopt falls back to robust_threshold_px / loss_kind. The staging
    // wants a LOOSE stage-1 threshold (~ the largest local structure the
    // grid is expected to absorb): at stage 1 that structure is still
    // unmodeled and shows up as large-but-genuine residuals that a tight
    // loss would suppress.
    std::optional<double> affine_robust_threshold_px;
    std::optional<RpcLossKind> affine_loss_kind;
    // Warm-start every free cell's shift at the per-cell MEDIAN of its
    // measures' offsets against the (scene affine only) projection of the
    // triangulated ground -- a 50%-breakdown estimate for a model whose
    // per-cell part is a pure translation. The incoming cell shifts are
    // DISCARDED (zeroed first, so the ground triangulation runs
    // shift-free); a converged warm start's medians reproduce its own
    // shifts, so re-running a solved network is near-idempotent. Cheap
    // (one forward projection per measure) and makes the Ceres polish
    // start inside the right basin under heavy mismatch noise.
    bool median_cell_init = false;
    // First-difference prior between the shifts of adjacent cells
    // (horizontal + vertical neighbors on the regular tensor grid,
    // residual = w * (d_a - d_b), Linear basis scenes only): the physical
    // error fields the grid models (distortion, stitching) are smooth, so
    // this suppresses isolated-cell spikes from noise while leaving
    // genuine structure. Like cell_shift_prior_weight, the weight is
    // relative to the pixel_sigma-normalized residuals: a w-weighted
    // prior charges (w * delta_px / pixel_sigma)^2 for an adjacent-cell
    // difference of delta_px. Pairs touching the derived (canonically
    // last) cell are omitted -- its shift is minus the sum of the others.
    // 0 disables.
    double cell_shift_smoothness_weight = 0.0;
    // Minimum MEASURES per cell (containing-cell accounting, ties + GCPs)
    // for a cell's shift to float; cells below the floor (and cells with
    // no measures at all) fall back to the scene affine: their shift is
    // zeroed and held constant through the solve. The statistical
    // robustifiers (loss, median init, MAD trimming) cannot save a cell
    // whose majority is noise -- a support floor can, by refusing to fit
    // it. 0 disables.
    int min_measures_per_cell = 0;
};

// One measure's reprojection residual, evaluated with the gridded solve's
// final model (scene affine + cell shifts + optimized ground) -- the
// out-record for caller-side robust filtering between multi-pass solves
// (see solve_rpc_bundle_adjust_robust). res_col/res_row are
// model - observed, in pixels.
struct RpcBaMeasureResidual {
    int point = 0;    // index into the `points` vector passed to the solve
    int measure = 0;  // index into points[point].measures
    int scene = 0;    // the measure's scene (RpcBaMeasure::view)
    int cell = -1;    // containing cell, -1 for a scene without cells
    double res_col = 0.0;
    double res_row = 0.0;
};

// Gridded solve: float every scene's global affine plus its cells'
// translation shifts against the control network. `corrected` carries
// one RpcAffineGridded per scene (parallel to `scenes`): its affine and
// cell shifts are the initial guess and are overwritten only on success
// (the grid structure -- cell ranges, count, basis -- is taken as
// given). `residuals`, when non-null, receives one entry per used
// measure on success (empty on failure).
RpcBaReport solve_rpc_bundle_adjust_gridded(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffineGridded>& corrected,
    const RpcBaGridOptions& options = {},
    std::vector<RpcBaMeasureResidual>* residuals = nullptr);

// Two-stage report: each stage's plain report, in order. The final
// corrections live in `corrected` (stage 1's affine + stage 2's shifts).
struct RpcBaTwoStageReport {
    RpcBaReport affine_stage;
    RpcBaReport grid_stage;
};

// Two-stage solve (see the notes above): stage 1 floats the global
// affines under the pixel-unit identity prior, stage 2 freezes them and
// floats the cell shifts. `corrected` carries one RpcAffineGridded per
// scene; its affine is stage 1's in/out value and its cells are stage
// 2's. The stage reports land in the two-stage report; `ok` per stage.
// `residuals`, when non-null, receives stage 2's per-measure residuals
// (the final model's). Stage 1's loss settings come from
// affine_robust_threshold_px / affine_loss_kind when set, else the base
// robust_threshold_px / loss_kind; stage 2 always uses the base pair.
RpcBaTwoStageReport solve_rpc_bundle_adjust_two_stage(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffineGridded>& corrected,
    const RpcBaGridOptions& options = {},
    std::vector<RpcBaMeasureResidual>* residuals = nullptr);

// ========================= robust variant ==================================
//
// solve_rpc_bundle_adjust_robust(): the multi-pass driver for noisy
// control networks -- tie-point match pipelines deliver a few percent of
// gross mismatches (repetitive texture, clouds, water), and the gridded
// model amplifies them: each cell's shift is a 2-parameter L2 estimate
// whose breakdown point is zero, so a single outlier drags a weakly
// covered cell and (through the per-scene zero-mean) nudges every other
// cell. Absolute pixel thresholds cannot separate "genuine local
// structure" from "noise" -- a cell with a real 8 px offset and a clean
// cell with an 8 px outlier look identical -- so the driver combines the
// defenses, loose-to-tight:
//
//   1. per-pass stage-wise LOSS settings (soft, inside Ceres' LM
//      iterations): a wide stage-1 threshold (~ the largest structure
//      the grid should absorb, so genuine local structure is not
//      suppressed where it is still unmodeled) tightening to a small
//      stage-2 one (the residual vs the full model is pure noise +
//      outliers by then);
//   2. between passes, per-measure TRIMMING by (scene, cell) group:
//      a measure is a CANDIDATE when its residual deviates from the
//      group's median by more than trim_mad_k robust sigmas (1.4826 *
//      MAD) -- the threshold FOLLOWS each cell's own offset, which is
//      exactly the per-cell-offset inconsistency that defeats absolute
//      thresholds -- floored at trim_floor_px so a clean (MAD ~ 0) group
//      trims nothing. Groups smaller than 3 measures keep everything
//      (the median of 1-2 measures carries no scale). Candidates are
//      then applied PER POINT, worst-first, one per pass: a bad measure
//      contaminates its whole point's ground block (the triangulation
//      init is plain least squares) and misfits the point's CLEAN
//      measures too, so independent per-measure drops would cull the
//      clean majority alongside the outlier. Dropping only the worst
//      candidate keeps the majority, and the next pass's fresh
//      triangulation heals the spared measures; a point already at its
//      measure floor (a 2-measure tie point with one bad measure has no
//      majority to heal from) goes entirely;
//   3. optional per-pass median_cell_init / cell_shift_smoothness_weight
//      / min_measures_per_cell (the grid knobs above) travel with every
//      pass through the base options.
//
// Each pass warm-starts from the previous (`corrected` in/out) and the
// final pass should carry trim_mad_k <= 0 -- trimming after the last
// solve is wasted. The graduated ladder (e.g. thresholds 3 pass widths
// apart, k 4 -> 3 -> 2, echoing the classic 32/8/2 px schedule) exists
// because a single tight pass risks culling the good measures of a
// genuinely-offset cell BEFORE the model has absorbed the offset;
// loose-first keeps every basin reachable.

// One pass of the ladder.
struct RpcBaRobustPass {
    // This pass's stage-1 / stage-2 loss thresholds in px (0 disables the
    // respective loss). A non-empty ladder sets BOTH explicitly on every
    // pass; the base options' robust_threshold_px /
    // affine_robust_threshold_px apply only to the empty-ladder single
    // pass.
    double affine_robust_threshold_px = 0.0;
    double grid_robust_threshold_px = 0.0;
    // Post-pass trim: measures deviating from their (scene, cell) group's
    // residual median by more than this many robust sigmas (1.4826 * MAD,
    // floored at trim_floor_px) become candidates, applied per point
    // (worst-first; see the notes above). <= 0: no trim (the last pass).
    // Recommended ladder: 4 -> 3 -> 2 -> 0.
    double trim_mad_k = 0.0;
};

struct RpcBaRobustOptions : RpcBaGridOptions {
    // The pass ladder, loose -> tight. Empty means a single un-tuned pass
    // with the base options' loss settings and no trim.
    std::vector<RpcBaRobustPass> passes;
    // MAD-trim floor in px: the trim radius of a (scene, cell) group is
    // max(trim_mad_k * 1.4826 * MAD, trim_floor_px).
    double trim_floor_px = 1.0;
};

struct RpcBaRobustReport {
    // True when every pass solved; `corrected` then holds the last pass's
    // solution. A failed pass stops the ladder, leaves `corrected` at
    // the last SUCCESSFUL pass's solution (a pass that cannot solve
    // should not contribute), and reports false.
    bool ok = false;
    std::vector<RpcBaTwoStageReport> passes;  // one per executed pass
    int num_measures_trimmed = 0;             // total across passes
    // Points removed entirely: at their measure floor (2 ties / 1 GCP)
    // with a still-gross candidate -- no majority to heal from.
    int num_points_dropped = 0;
    std::string message;
};

// Robust multi-pass solve: runs the two-stage driver per pass over a
// shrinking control network (see above), warm-starting `corrected`
// throughout. `corrected` is one RpcAffineGridded per scene as in the
// two-stage solve.
RpcBaRobustReport solve_rpc_bundle_adjust_robust(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffineGridded>& corrected,
    const RpcBaRobustOptions& options = {});

}  // namespace zproj::crs
