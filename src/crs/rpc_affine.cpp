// Ceres-based solver for the RPC image-space affine bias correction
// (see rpc_affine.hpp for the model and formulation).
//
// The problem is a miniature bundle adjustment in the style of ASP's
// BaReprojErr cost: per-image affine blocks (6 or fewer free parameters
// each, restricted through SubsetManifold by RpcAffineDoF), one free ground
// block per stereo match initialized by ray triangulation, constant ground
// blocks for GCPs, and 2-residual reprojection costs that evaluate only the
// forward RPC -- a smooth rational polynomial, so autodiff derivatives are
// exact. The cost functors and solve tail are shared with the N-view
// bundle adjustment (rpc_bundle_adjust.cpp) through rpc_affine_detail.h.
#include "zproj/crs/rpc_affine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "zproj/crs/wgs84.hpp"

#include "rpc_affine_detail.h"

namespace zproj::crs {
namespace {

using refine_detail::AffinePrior;
using refine_detail::kAffineIdentity;
using refine_detail::MakeAffineManifold;
using refine_detail::MakeReprojCost;
using refine_detail::NewLoss;
using refine_detail::ResidualSq;

// The mirrored affine (2*identity - p), used for the right image when
// zero_mean_affines is on.
std::array<double, 6> MirrorAffine(const std::array<double, 6>& p) {
    std::array<double, 6> out{};
    for (int i = 0; i < 6; ++i) {
        out[i] = 2.0 * kAffineIdentity[i] - p[i];
    }
    return out;
}

// RMS over every scalar residual component (2 per observation): matches in
// both images plus GCPs, without the robust loss so outliers always show up.
double ReprojectionRms(const RpcInfo& left,
                       const RpcInfo& right,
                       const std::vector<RpcMatch>& matches,
                       const std::vector<std::array<double, 3>>& ground,
                       const std::vector<RpcGcp>& left_gcps,
                       const std::vector<RpcGcp>& right_gcps,
                       const std::vector<std::array<double, 3>>& gcp_ground,
                       const double* left_aff,
                       const double* right_aff) {
    double sum = 0.0;
    std::size_t n_res = 0;
    for (std::size_t i = 0; i < matches.size(); ++i) {
        sum += ResidualSq(left,
                          left_aff,
                          ground[i].data(),
                          matches[i].left_col,
                          matches[i].left_row);
        sum += ResidualSq(right,
                          right_aff,
                          ground[i].data(),
                          matches[i].right_col,
                          matches[i].right_row);
        n_res += 4;
    }
    std::size_t gcp_i = 0;
    const auto AccumGcps = [&](const RpcInfo& info,
                               const std::vector<RpcGcp>& gcps,
                               const double* aff) {
        for (const RpcGcp& g : gcps) {
            sum +=
                ResidualSq(info, aff, gcp_ground[gcp_i].data(), g.col, g.row);
            n_res += 2;
            ++gcp_i;
        }
    };
    AccumGcps(left, left_gcps, left_aff);
    AccumGcps(right, right_gcps, right_aff);
    return (n_res > 0) ? std::sqrt(sum / static_cast<double>(n_res)) : 0.0;
}

// Ground-block initialization for one match: back-project both
// (affine-corrected) pixels to rays and intersect, then convert the ECEF
// intersection to the solver's (lon deg, lat deg, h m) parameterization.
bool InitMatchGround(const RpcInfo& left,
                     const RpcInverseInit& left_init,
                     const RpcAffine& left_affine,
                     const RpcInfo& right,
                     const RpcInverseInit& right_init,
                     const RpcAffine& right_affine,
                     const RpcMatch& m,
                     double h_low,
                     double h_high,
                     std::array<double, 3>& lonlath) {
    RpcRay ray_l;
    RpcRay ray_r;
    Ecef p;
    double err = 0.0;
    if (!rpc_ray_affine(left,
                        left_init,
                        left_affine,
                        m.left_col,
                        m.left_row,
                        h_low,
                        h_high,
                        ray_l) ||
        !rpc_ray_affine(right,
                        right_init,
                        right_affine,
                        m.right_col,
                        m.right_row,
                        h_low,
                        h_high,
                        ray_r) ||
        !triangulate_pair(ray_l, ray_r, p, err) || !std::isfinite(p.x()) ||
        !std::isfinite(p.y()) || !std::isfinite(p.z())) {
        return false;
    }
    const Geodetic g = from_ecef(p);
    lonlath = {g.x() * kRadToDeg, g.y() * kRadToDeg, g.z()};
    return true;
}

}  // namespace

RpcAffineReport solve_rpc_affine(const RpcInfo& left,
                                 const RpcInfo& right,
                                 const std::vector<RpcMatch>& matches,
                                 const std::vector<RpcGcp>& left_gcps,
                                 const std::vector<RpcGcp>& right_gcps,
                                 RpcAffine& left_affine,
                                 RpcAffine& right_affine,
                                 const RpcAffineOptions& options) {
    RpcAffineReport report;
    const int num_gcps = static_cast<int>(left_gcps.size() + right_gcps.size());

    if (matches.empty() && num_gcps == 0) {
        report.message = "no observations (matches and GCPs are both empty)";
        return report;
    }

    // Constraint accounting: a match contributes 4 residuals but 3 new
    // unknowns (its ground block), net 1; each GCP observation contributes 2
    // residuals against a constant ground block, net 2. Require the net
    // count to reach the floated affine parameters -- unless a prior is
    // active, whose Tikhonov terms make the normal equations full rank
    // regardless. With zero_mean_affines one block drives both images.
    const int n_affine_params =
        (options.zero_mean_affines ? 1 : 2) * static_cast<int>(options.dof);
    const bool has_prior = options.affine_prior_weight > 0.0;
    const auto UnderDetermined = [&](int n_matches) {
        return (static_cast<double>(n_matches) + 2.0 * num_gcps) <
               static_cast<double>(n_affine_params);
    };
    if (!has_prior && UnderDetermined(static_cast<int>(matches.size()))) {
        report.message = "under-determined: " + std::to_string(matches.size()) +
                         " matches + " + std::to_string(num_gcps) +
                         " GCP observations cannot float " +
                         std::to_string(n_affine_params) + " affine parameters";
        return report;
    }

    // Ground blocks for the matches (mutated by the solve; `ground_init`
    // keeps the triangulated starting values for the before-RMS). The height
    // span follows ASP's RPCModel::point_and_dir clamp.
    const RpcModel left_model(left);
    const RpcModel right_model(right);
    const double span = std::min(0.9 * left.height_scale, 50.0);
    const double h_low = left.height_off - span;
    const double h_high = left.height_off + span;

    // Initialization is pure per-point math (four analytic inverses + a
    // ray intersection per match); at 1e5 matches it would dominate the wall
    // time, so it runs in parallel (OpenMP when linked, serial otherwise)
    // into pre-sized arrays and is compacted serially below.
    std::vector<std::array<double, 3>> ground_all(matches.size());
    std::vector<unsigned char> init_ok(matches.size(), 0);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(matches.size());
         ++i) {
        init_ok[i] = InitMatchGround(left,
                                     left_model.inverse_init(),
                                     left_affine,
                                     right,
                                     right_model.inverse_init(),
                                     right_affine,
                                     matches[static_cast<std::size_t>(i)],
                                     h_low,
                                     h_high,
                                     ground_all[static_cast<std::size_t>(i)])
                         ? 1
                         : 0;
    }
    std::vector<RpcMatch> used;
    used.reserve(matches.size());
    std::vector<std::array<double, 3>> ground;
    ground.reserve(matches.size());
    for (std::size_t i = 0; i < matches.size(); ++i) {
        if (!init_ok[i]) {
            ++report.num_matches_skipped;
            continue;
        }
        // A finite match height (e.g. sampled from a DEM) pins the ground
        // block's height; the free-triangulation value only seeds lon/lat.
        if (std::isfinite(matches[i].height)) {
            ground_all[i][2] = matches[i].height;
        }
        ground.push_back(ground_all[i]);
        used.push_back(matches[i]);
    }
    report.num_matches = static_cast<int>(used.size());

    if (used.empty() && num_gcps == 0) {
        report.message =
            "no usable observations (every match failed to triangulate)";
        return report;
    }
    if (!has_prior && UnderDetermined(static_cast<int>(used.size()))) {
        report.message = "under-determined after skipping failed matches: " +
                         std::to_string(used.size()) + " matches + " +
                         std::to_string(num_gcps) +
                         " GCP observations cannot float " +
                         std::to_string(n_affine_params) + " affine parameters";
        return report;
    }
    const std::vector<std::array<double, 3>> ground_init = ground;

    // Problem assembly. The affine blocks live in local storage so a failed
    // solve leaves the caller's affines untouched. With zero_mean_affines a
    // single block drives both images: the right image's residuals evaluate
    // the mirrored affine (2*identity - la), an exact mean-to-identity
    // constraint.
    std::array<double, 6> la = left_affine.p;
    std::array<double, 6> ra = right_affine.p;
    const bool zero_mean = options.zero_mean_affines;
    double* right_block = zero_mean ? la.data() : ra.data();
    ceres::Problem problem;
    problem.AddParameterBlock(la.data(), 6);
    if (!zero_mean) {
        problem.AddParameterBlock(ra.data(), 6);
        if (auto manifold = MakeAffineManifold(options.dof)) {
            problem.SetManifold(ra.data(), manifold.release());
        }
    }
    if (auto manifold = MakeAffineManifold(options.dof)) {
        problem.SetManifold(la.data(), manifold.release());
    }

    for (std::size_t i = 0; i < used.size(); ++i) {
        problem.AddParameterBlock(ground[i].data(), 3);
        if (std::isfinite(used[i].height)) {
            // DEM-anchored match: hold the height constant (index 2).
            problem.SetManifold(ground[i].data(),
                                new ceres::SubsetManifold(3, {2}));
        }
        problem.AddResidualBlock(
            MakeReprojCost(
                left, used[i].left_col, used[i].left_row, options.pixel_sigma),
            NewLoss(options.robust_threshold_px, options.pixel_sigma),
            la.data(),
            ground[i].data());
        problem.AddResidualBlock(
            MakeReprojCost(right,
                           used[i].right_col,
                           used[i].right_row,
                           options.pixel_sigma,
                           zero_mean),
            NewLoss(options.robust_threshold_px, options.pixel_sigma),
            right_block,
            ground[i].data());
    }

    // GCP observations: constant ground blocks (the ground coordinates ARE
    // the measurement). Blocks must stay alive for the problem's lifetime.
    std::vector<std::array<double, 3>> gcp_ground;
    gcp_ground.reserve(left_gcps.size() + right_gcps.size());
    const auto AddGcpBlocks = [&](const RpcInfo& info,
                                  const std::vector<RpcGcp>& gcps,
                                  double* aff,
                                  bool mirror) {
        for (const RpcGcp& g : gcps) {
            gcp_ground.push_back({g.lon, g.lat, g.height});
            problem.AddParameterBlock(gcp_ground.back().data(), 3);
            problem.SetParameterBlockConstant(gcp_ground.back().data());
            problem.AddResidualBlock(
                MakeReprojCost(info, g.col, g.row, options.pixel_sigma, mirror),
                NewLoss(options.robust_threshold_px, options.pixel_sigma),
                aff,
                gcp_ground.back().data());
        }
    };
    AddGcpBlocks(left, left_gcps, la.data(), false);
    AddGcpBlocks(right, right_gcps, right_block, zero_mean);

    if (options.affine_prior_weight > 0.0) {
        const double w = options.affine_prior_weight;
        // Under zero_mean this regularizes the differential parameters only
        // (there is just the one block).
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                new AffinePrior(w)),
            nullptr,
            la.data());
        if (!zero_mean) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                    new AffinePrior(w)),
                nullptr,
                ra.data());
        }
    }

    // Under zero_mean the effective right affine is the mirror of la (the
    // caller's initial right affine is overruled by the constraint); keep ra
    // in sync so the RMS loops and the write-back agree on one truth.
    if (zero_mean) {
        ra = MirrorAffine(la);
    }

    // The "before" RMS uses the triangulated ground blocks and the initial
    // affines, both ahead of the solve.
    const double rms_before = ReprojectionRms(left,
                                              right,
                                              used,
                                              ground_init,
                                              left_gcps,
                                              right_gcps,
                                              gcp_ground,
                                              la.data(),
                                              ra.data());

    std::vector<std::pair<double*, RpcAffine*>> affine_blocks{
        {la.data(), &left_affine}};
    if (!zero_mean) {
        affine_blocks.emplace_back(ra.data(), &right_affine);
    }
    // The problem has classic bundle-adjustment sparsity: a handful of
    // affine blocks ("cameras") and one 3-parameter ground block per match
    // ("points"), every residual touching one of each. DENSE_SCHUR
    // eliminates the ground blocks and solves the <= 12 x 12 reduced affine
    // system instead: O(N) per iteration.
    RpcAffineReport solved =
        refine_detail::SolveAndReport(problem,
                                      options.max_iterations,
                                      options.num_threads,
                                      options.verbose,
                                      ceres::DENSE_SCHUR,
                                      affine_blocks);
    if (zero_mean) {
        ra = MirrorAffine(la);
        if (solved.ok) {
            right_affine.p = ra;
        }
    }
    solved.num_matches = static_cast<int>(used.size());
    solved.num_matches_skipped = static_cast<int>(matches.size() - used.size());
    solved.num_gcps = num_gcps;
    solved.rms_before_px = rms_before;
    solved.rms_after_px = ReprojectionRms(left,
                                          right,
                                          used,
                                          ground,
                                          left_gcps,
                                          right_gcps,
                                          gcp_ground,
                                          la.data(),
                                          ra.data());
    return solved;
}

RpcAffineReport solve_rpc_affine(const RpcInfo& info,
                                 const std::vector<RpcGcp>& gcps,
                                 RpcAffine& affine,
                                 const RpcAffineOptions& options) {
    RpcAffineReport report;
    if (gcps.empty()) {
        report.message = "no observations (GCP list is empty)";
        return report;
    }

    const int n_affine_params = static_cast<int>(options.dof);
    if (options.affine_prior_weight <= 0.0 &&
        2.0 * static_cast<double>(gcps.size()) <
            static_cast<double>(n_affine_params)) {
        report.message = "under-determined: " + std::to_string(gcps.size()) +
                         " GCPs cannot float " +
                         std::to_string(n_affine_params) + " affine parameters";
        return report;
    }

    std::array<double, 6> a = affine.p;
    ceres::Problem problem;
    problem.AddParameterBlock(a.data(), 6);
    if (auto manifold = MakeAffineManifold(options.dof)) {
        problem.SetManifold(a.data(), manifold.release());
    }

    std::vector<std::array<double, 3>> gcp_ground;
    gcp_ground.reserve(gcps.size());
    for (const RpcGcp& g : gcps) {
        gcp_ground.push_back({g.lon, g.lat, g.height});
        problem.AddParameterBlock(gcp_ground.back().data(), 3);
        problem.SetParameterBlockConstant(gcp_ground.back().data());
        problem.AddResidualBlock(
            MakeReprojCost(info, g.col, g.row, options.pixel_sigma),
            NewLoss(options.robust_threshold_px, options.pixel_sigma),
            a.data(),
            gcp_ground.back().data());
    }

    if (options.affine_prior_weight > 0.0) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                new AffinePrior(options.affine_prior_weight)),
            nullptr,
            a.data());
    }

    double sum_before = 0.0;
    for (std::size_t i = 0; i < gcps.size(); ++i) {
        sum_before += ResidualSq(
            info, a.data(), gcp_ground[i].data(), gcps[i].col, gcps[i].row);
    }
    const double rms_before =
        std::sqrt(sum_before / (2.0 * static_cast<double>(gcps.size())));

    // No free ground blocks here, so there is nothing for a Schur solver to
    // eliminate; the plain dense QR is the right (and fastest) choice.
    RpcAffineReport solved =
        refine_detail::SolveAndReport(problem,
                                      options.max_iterations,
                                      options.num_threads,
                                      options.verbose,
                                      ceres::DENSE_QR,
                                      {{a.data(), &affine}});
    solved.num_gcps = static_cast<int>(gcps.size());
    solved.rms_before_px = rms_before;

    double sum_after = 0.0;
    for (std::size_t i = 0; i < gcps.size(); ++i) {
        sum_after += ResidualSq(
            info, a.data(), gcp_ground[i].data(), gcps[i].col, gcps[i].row);
    }
    solved.rms_after_px =
        std::sqrt(sum_after / (2.0 * static_cast<double>(gcps.size())));
    return solved;
}

}  // namespace zproj::crs
