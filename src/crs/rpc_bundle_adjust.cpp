// Ceres-based N-view bundle adjustment over an RPC + per-view image-affine
// control network (see rpc_bundle_adjust.hpp for the model and formulation).
//
// The assembly mirrors ASP's bundle_adjust: one 2-residual reprojection cost
// per (point, view) measure, a free 3-parameter ground block per tie point
// (eliminated by DENSE_SCHUR -- ASP's solver choice below ~100 cameras),
// constant ground blocks for GCPs, and a Tikhonov prior on the floated
// affines. The zero-mean gauge is enforced exactly by parameterization: the
// last view's affine is derived from the others (the two-view mirror
// generalized to N), so no residual ever has to trade constraint tightness
// against gauge correctness.
#include "zproj/crs/rpc_bundle_adjust.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/wgs84.hpp"

#include "rpc_affine_detail.h"

namespace zproj::crs {
namespace {

using refine_detail::AffinePrior;
using refine_detail::IdentityPriorPx;
using refine_detail::kAffineIdentity;
using refine_detail::MakeAffineManifold;
using refine_detail::MakeReprojCost;
using refine_detail::NewLoss;
using refine_detail::ResidualSq;

// Reprojection cost for a measure in the DERIVED (last) view under
// zero_mean_affines: the affine applied is
//
//     p_derived = V*identity - sum(free affine blocks),   V = num_free + 1,
//
// so the network's affines sum to V*identity exactly -- the two-view mirror
// (2*identity - p) generalized to N views. Dynamic autodiff over the
// num_free affine blocks plus the ground block (parameter count varies with
// the number of views, so the static AutoDiffCostFunction cannot express
// it). Holds a pointer to the shared RpcInfo for the same memory reason as
// ReprojError.
struct DerivedReprojError {
    DerivedReprojError(const RpcInfo* info,
                       double obs_col,
                       double obs_row,
                       double pixel_sigma,
                       int num_free)
        : info_(info),
          obs_col_(obs_col),
          obs_row_(obs_row),
          num_free_(num_free) {
        inv_sigma_ = 1.0 / ((pixel_sigma > 0.0) ? pixel_sigma : 1.0);
        // (num_free + 1) * identity: what the affine sum must reach for the
        // MEAN to be the identity.
        for (int k = 0; k < 6; ++k) {
            scaled_identity_[k] =
                static_cast<double>(num_free_ + 1) * kAffineIdentity[k];
        }
    }

    template<typename T>
    bool operator()(T const* const* params, T* residuals) const {
        T e0 = T(scaled_identity_[0]);
        T e1 = T(scaled_identity_[1]);
        T e2 = T(scaled_identity_[2]);
        T f0 = T(scaled_identity_[3]);
        T f1 = T(scaled_identity_[4]);
        T f2 = T(scaled_identity_[5]);
        for (int i = 0; i < num_free_; ++i) {
            const T* p = params[i];
            e0 -= p[0];
            e1 -= p[1];
            e2 -= p[2];
            f0 -= p[3];
            f1 -= p[4];
            f2 -= p[5];
        }
        const T* lonlath = params[num_free_];
        T col;
        T row;
        detail::rpc_forward_point_core(
            *info_, lonlath[0], lonlath[1], lonlath[2], col, row);
        const T corr_col = e0 + (e1 * col) + (e2 * row);
        const T corr_row = f0 + (f1 * col) + (f2 * row);
        residuals[0] = (corr_col - T(obs_col_)) * T(inv_sigma_);
        residuals[1] = (corr_row - T(obs_row_)) * T(inv_sigma_);
        return true;
    }

    const RpcInfo* info_;
    double obs_col_;
    double obs_row_;
    double inv_sigma_;
    int num_free_;
    std::array<double, 6> scaled_identity_{};
};

ceres::CostFunction* MakeDerivedReprojCost(const RpcInfo& info,
                                           double obs_col,
                                           double obs_row,
                                           double pixel_sigma,
                                           int num_free) {
    auto* fn = new ceres::DynamicAutoDiffCostFunction<DerivedReprojError>(
        new DerivedReprojError(&info, obs_col, obs_row, pixel_sigma, num_free));
    fn->SetNumResiduals(2);
    for (int i = 0; i < num_free; ++i) {
        fn->AddParameterBlock(6);
    }
    fn->AddParameterBlock(3);
    return fn;
}

// The derived affine (V*identity - sum of the first num_free blocks) at
// double precision, for the RMS loops and the write-back.
RpcAffine DerivedAffine(const std::vector<std::array<double, 6>>& blocks,
                        int num_free) {
    RpcAffine out;
    const double v = static_cast<double>(num_free + 1);
    for (int k = 0; k < 6; ++k) {
        out.p[k] = v * kAffineIdentity[k];
    }
    for (int i = 0; i < num_free; ++i) {
        for (int k = 0; k < 6; ++k) {
            out.p[k] -= blocks[static_cast<std::size_t>(i)][k];
        }
    }
    return out;
}

// Ground-block initialization for one tie point: back-project every
// observing view's (affine-corrected) pixel to a viewing ray and intersect
// the bundle in the least-squares sense (triangulate_nview -- VW
// StereoModel::triangulate_point; two valid rays take the closed form
// inside). Each view uses its own height span, the two-view solver's
// ASP point_and_dir clamp.
bool InitPointGround(const std::vector<RpcInfo>& views,
                     const std::vector<RpcModel>& models,
                     const std::vector<RpcAffine>& affines,
                     const RpcBaPoint& pt,
                     std::array<double, 3>& lonlath) {
    std::vector<RpcRay> rays;
    rays.reserve(pt.measures.size());
    for (const RpcBaMeasure& m : pt.measures) {
        const RpcInfo& v = views[static_cast<std::size_t>(m.view)];
        const double span = std::min(0.9 * v.height_scale, 50.0);
        RpcRay ray;
        if (rpc_ray_affine(
                v,
                models[static_cast<std::size_t>(m.view)].inverse_init(),
                affines[static_cast<std::size_t>(m.view)],
                m.col,
                m.row,
                v.height_off - span,
                v.height_off + span,
                ray)) {
            rays.push_back(ray);
        }
    }
    if (rays.size() < 2) {
        return false;
    }
    Ecef p;
    double rms = 0.0;
    if (!triangulate_nview(
            rays.data(), static_cast<int>(rays.size()), p, rms) ||
        !std::isfinite(p.x()) || !std::isfinite(p.y()) ||
        !std::isfinite(p.z())) {
        return false;
    }
    const Geodetic g = from_ecef(p);
    lonlath = {g.x() * kRadToDeg, g.y() * kRadToDeg, g.z()};
    return true;
}

}  // namespace

RpcBaReport solve_rpc_bundle_adjust(const std::vector<RpcInfo>& views,
                                    const std::vector<RpcBaPoint>& points,
                                    std::vector<RpcAffine>& affines,
                                    const RpcBaOptions& options) {
    RpcBaReport report;
    const int num_views = static_cast<int>(views.size());
    if (num_views < 2) {
        report.message =
            "need >= 2 views (got " + std::to_string(num_views) + ")";
        return report;
    }
    if (affines.size() != views.size()) {
        report.message = "affines must have one entry per view (got " +
                         std::to_string(affines.size()) + ", expected " +
                         std::to_string(views.size()) + ")";
        return report;
    }

    // ---- Sanitize the control network ------------------------------------
    // Drop measures with out-of-range views (the whole point: malformed
    // input), keep the FIRST measure per view within a point (duplicates
    // would double-count one observation), then require tie points to have
    // >= 2 measures and GCPs to have finite ground plus >= 1 measure.
    std::vector<RpcBaPoint> ties;
    std::vector<RpcBaPoint> gcps;
    int skipped = 0;
    for (const RpcBaPoint& pt : points) {
        if (pt.ground_fixed &&
            (!std::isfinite(pt.lon) || !std::isfinite(pt.lat) ||
             !std::isfinite(pt.height))) {
            ++skipped;
            continue;
        }
        RpcBaPoint clean = pt;
        clean.measures.clear();
        bool bad = false;
        for (const RpcBaMeasure& m : pt.measures) {
            if (m.view < 0 || m.view >= num_views) {
                bad = true;
                break;
            }
            bool dup = false;
            for (const RpcBaMeasure& k : clean.measures) {
                dup = dup || (k.view == m.view);
            }
            if (!dup) {
                clean.measures.push_back(m);
            }
        }
        if (bad) {
            ++skipped;
            continue;
        }
        const std::size_t need = pt.ground_fixed ? 1 : 2;
        if (clean.measures.size() < need) {
            ++skipped;
            continue;
        }
        (pt.ground_fixed ? gcps : ties).push_back(std::move(clean));
    }

    // ---- Ground-block initialization for the tie points ------------------
    // Pure per-point math (two analytic inverses per measure + one N-view
    // intersection), parallelized exactly like the two-view solver.
    std::vector<RpcModel> models;
    models.reserve(views.size());
    for (const RpcInfo& v : views) {
        models.emplace_back(v);
    }

    std::vector<std::array<double, 3>> ground_all(ties.size());
    std::vector<unsigned char> init_ok(ties.size(), 0);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(ties.size());
         ++i) {
        const auto idx = static_cast<std::size_t>(i);
        init_ok[idx] =
            InitPointGround(views, models, affines, ties[idx], ground_all[idx])
                ? 1
                : 0;
    }
    // Compact the initialized points in place; a finite tie-point height
    // (e.g. from a DEM) pins the ground block's height, the triangulated
    // value only seeds lon/lat.
    std::vector<std::array<double, 3>> ground;
    ground.reserve(ties.size());
    std::size_t kept = 0;
    for (std::size_t i = 0; i < ties.size(); ++i) {
        if (!init_ok[i]) {
            ++skipped;
            continue;
        }
        if (std::isfinite(ties[i].height)) {
            ground_all[i][2] = ties[i].height;
        }
        ground.push_back(ground_all[i]);
        if (kept != i) {
            ties[kept] = std::move(ties[i]);
        }
        ++kept;
    }
    ties.resize(kept);
    report.num_points = static_cast<int>(ties.size());
    report.num_gcps = static_cast<int>(gcps.size());

    if (ties.empty() && gcps.empty()) {
        report.message =
            "no usable observations (every control point was malformed or "
            "failed to triangulate)";
        report.num_points_skipped = skipped;
        return report;
    }

    // ---- Constraint accounting -------------------------------------------
    // A tie point with m measures contributes 2m residuals but 3 new
    // unknowns (2 when its height is pinned); each GCP measure contributes
    // 2 residuals against a constant ground block. Require the net count to
    // reach the floated affine parameters -- unless a prior is active. With
    // zero_mean_affines the derived view adds no free parameters.
    const bool zero_mean = options.zero_mean_affines;
    const int num_free = zero_mean ? num_views - 1 : num_views;
    const int n_affine_params = num_free * static_cast<int>(options.dof);
    const bool has_prior =
        options.affine_prior_weight > 0.0 || options.identity_prior_px > 0.0;
    double net = 0.0;
    for (const RpcBaPoint& t : ties) {
        net += 2.0 * static_cast<double>(t.measures.size()) -
               (std::isfinite(t.height) ? 2.0 : 3.0);
    }
    for (const RpcBaPoint& g : gcps) {
        net += 2.0 * static_cast<double>(g.measures.size());
    }
    if (!has_prior && net < static_cast<double>(n_affine_params)) {
        report.message = "under-determined: " + std::to_string(ties.size()) +
                         " tie points + " + std::to_string(gcps.size()) +
                         " GCPs net only " + std::to_string(net) +
                         " constraints and cannot float " +
                         std::to_string(n_affine_params) + " affine parameters";
        report.num_points_skipped = skipped;
        return report;
    }
    const std::vector<std::array<double, 3>> ground_init = ground;

    // ---- Problem assembly -------------------------------------------------
    // The affine blocks live in local storage so a failed solve leaves the
    // caller's affines untouched. Views 0..num_free-1 are free parameter
    // blocks; under zero_mean the last view's affine is DERIVED from them.
    std::vector<std::array<double, 6>> blocks(
        static_cast<std::size_t>(num_views));
    for (int v = 0; v < num_views; ++v) {
        blocks[static_cast<std::size_t>(v)] =
            affines[static_cast<std::size_t>(v)].p;
    }

    ceres::Problem problem;
    std::vector<double*> free_blocks;
    free_blocks.reserve(static_cast<std::size_t>(num_free));
    for (int v = 0; v < num_free; ++v) {
        double* block = blocks[static_cast<std::size_t>(v)].data();
        problem.AddParameterBlock(block, 6);
        if (auto manifold = MakeAffineManifold(options.dof)) {
            problem.SetManifold(block, manifold.release());
        }
        free_blocks.push_back(block);
    }

    // Schur ordering: eliminate the free ground blocks (group 0); the
    // affine blocks form the reduced camera system (group 1). Constant
    // (GCP) blocks are not part of the solved system.
    auto ordering = std::make_unique<ceres::ParameterBlockOrdering>();

    const auto AddMeasures = [&](const RpcBaPoint& pt, double* ground_block) {
        for (const RpcBaMeasure& m : pt.measures) {
            const RpcInfo& v = views[static_cast<std::size_t>(m.view)];
            if (zero_mean && m.view == num_views - 1) {
                // Derived view: one dynamic cost over every free affine
                // block plus the ground block (the derived affine couples
                // them all; DENSE_SCHUR folds the coupling into the reduced
                // camera system).
                std::vector<double*> params = free_blocks;
                params.push_back(ground_block);
                problem.AddResidualBlock(
                    MakeDerivedReprojCost(
                        v, m.col, m.row, options.pixel_sigma, num_free),
                    NewLoss(options.loss_kind,
                            options.robust_threshold_px,
                            options.pixel_sigma),
                    params);
                continue;
            }
            problem.AddResidualBlock(
                MakeReprojCost(v, m.col, m.row, options.pixel_sigma),
                NewLoss(options.loss_kind,
                        options.robust_threshold_px,
                        options.pixel_sigma),
                blocks[static_cast<std::size_t>(m.view)].data(),
                ground_block);
        }
    };

    // Tie points: free ground block, height held constant when pinned.
    for (std::size_t i = 0; i < ties.size(); ++i) {
        double* g = ground[i].data();
        problem.AddParameterBlock(g, 3);
        if (std::isfinite(ties[i].height)) {
            problem.SetManifold(g, new ceres::SubsetManifold(3, {2}));
        }
        ordering->AddElementToGroup(g, 0);
        AddMeasures(ties[i], g);
    }

    // GCPs: the ground coordinates ARE the measurement, so the block stays
    // constant and the pixel measures pin the affines absolutely (ASP's
    // --fix-gcp_xyz mode). Blocks must stay alive for the problem's
    // lifetime.
    std::vector<std::array<double, 3>> gcp_ground;
    gcp_ground.reserve(gcps.size());
    for (std::size_t i = 0; i < gcps.size(); ++i) {
        gcp_ground.push_back({gcps[i].lon, gcps[i].lat, gcps[i].height});
        double* g = gcp_ground.back().data();
        problem.AddParameterBlock(g, 3);
        problem.SetParameterBlockConstant(g);
        AddMeasures(gcps[i], g);
    }

    for (double* b : free_blocks) {
        ordering->AddElementToGroup(b, 1);
    }

    if (has_prior) {
        // Under zero_mean this regularizes the differential parameters only
        // (the derived view adds none).
        for (int v = 0; v < num_free; ++v) {
            double* b = free_blocks[static_cast<std::size_t>(v)];
            if (options.identity_prior_px > 0.0) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<IdentityPriorPx, 6, 6>(
                        new IdentityPriorPx(
                            options.identity_prior_px,
                            views[static_cast<std::size_t>(v)].samp_scale,
                            views[static_cast<std::size_t>(v)].line_scale)),
                    nullptr,
                    b);
            }
            if (options.affine_prior_weight > 0.0) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                        new AffinePrior(options.affine_prior_weight)),
                    nullptr,
                    b);
            }
        }
    }

    // ---- RMS before / solve / write-back ----------------------------------
    // Effective per-view affines (the derived view under zero_mean is a
    // function of the free blocks, so the caller's initial affine for it is
    // overruled by the constraint).
    std::vector<RpcAffine> effective(static_cast<std::size_t>(num_views));
    const auto SyncEffective = [&]() {
        for (int v = 0; v < num_free; ++v) {
            effective[static_cast<std::size_t>(v)].p =
                blocks[static_cast<std::size_t>(v)];
        }
        if (zero_mean) {
            effective[static_cast<std::size_t>(num_views - 1)] =
                DerivedAffine(blocks, num_free);
        }
    };

    const auto ReprojectionRms =
        [&](const std::vector<RpcAffine>& eff,
            const std::vector<std::array<double, 3>>& tie_ground) {
            double sum = 0.0;
            std::size_t n_res = 0;
            const auto Accum = [&](const RpcBaPoint& pt,
                                   const std::array<double, 3>& g) {
                for (const RpcBaMeasure& m : pt.measures) {
                    sum += ResidualSq(
                        views[static_cast<std::size_t>(m.view)],
                        eff[static_cast<std::size_t>(m.view)].p.data(),
                        g.data(),
                        m.col,
                        m.row);
                    n_res += 2;
                }
            };
            for (std::size_t i = 0; i < ties.size(); ++i) {
                Accum(ties[i], tie_ground[i]);
            }
            for (std::size_t i = 0; i < gcps.size(); ++i) {
                Accum(gcps[i], gcp_ground[i]);
            }
            return (n_res > 0) ? std::sqrt(sum / static_cast<double>(n_res))
                               : 0.0;
        };

    SyncEffective();
    const double rms_before = ReprojectionRms(effective, ground_init);

    std::vector<std::pair<double*, RpcAffine*>> affine_blocks;
    affine_blocks.reserve(free_blocks.size());
    for (int v = 0; v < num_free; ++v) {
        affine_blocks.emplace_back(free_blocks[static_cast<std::size_t>(v)],
                                   &affines[static_cast<std::size_t>(v)]);
    }
    // With free ground blocks (any tie point) the problem has the classic
    // cameras-vs-points sparsity and DENSE_SCHUR eliminates the 3-parameter
    // point blocks; a GCP-only problem has nothing to eliminate and takes
    // the dense QR, like the single-image solver.
    const ceres::LinearSolverType solver_type =
        ties.empty() ? ceres::DENSE_QR : ceres::DENSE_SCHUR;
    std::unique_ptr<ceres::ParameterBlockOrdering> solve_ordering;
    if (!ties.empty()) {
        solve_ordering = std::move(ordering);
    }
    RpcAffineReport solved =
        refine_detail::SolveAndReport(problem,
                                      options.max_iterations,
                                      options.num_threads,
                                      options.verbose,
                                      solver_type,
                                      affine_blocks,
                                      std::move(solve_ordering));
    if (zero_mean) {
        blocks[static_cast<std::size_t>(num_views - 1)] =
            DerivedAffine(blocks, num_free).p;
    }

    SyncEffective();
    if (solved.ok) {
        for (int v = 0; v < num_views; ++v) {
            affines[static_cast<std::size_t>(v)] =
                effective[static_cast<std::size_t>(v)];
        }
    }

    RpcBaReport out;
    out.ok = solved.ok;
    out.num_points = report.num_points;
    out.num_points_skipped = skipped;
    out.num_gcps = report.num_gcps;
    out.num_measures = 0;
    for (const RpcBaPoint& t : ties) {
        out.num_measures += static_cast<int>(t.measures.size());
    }
    for (const RpcBaPoint& g : gcps) {
        out.num_measures += static_cast<int>(g.measures.size());
    }
    out.rms_before_px = rms_before;
    out.rms_after_px = ReprojectionRms(effective, ground);
    out.message = std::move(solved.message);
    return out;
}

}  // namespace zproj::crs
