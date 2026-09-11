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

#include <limits>
#include <string>
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
    // Huber threshold in pixels; <= 0 disables the robust loss.
    double robust_threshold_px = 0.0;
    // Tikhonov weight pulling each floated affine towards identity. 0
    // disables. Under zero_mean_affines it regularizes the differential
    // parameters only.
    double affine_prior_weight = 0.0;
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

}  // namespace zproj::crs
