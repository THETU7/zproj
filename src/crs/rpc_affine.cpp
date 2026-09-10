// Ceres-based solver for the RPC image-space affine bias correction
// (see rpc_affine.hpp for the model and formulation).
//
// The problem is a miniature bundle adjustment in the style of ASP's
// BaReprojErr: per-image affine blocks (6 or fewer free parameters each,
// restricted through SubsetManifold by RpcAffineDoF), one free ground block
// per stereo match initialized by ray triangulation, constant ground blocks
// for GCPs, and 2-residual reprojection costs that evaluate only the forward
// RPC -- a smooth rational polynomial, so autodiff derivatives are exact.
#include "zproj/crs/rpc_affine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "zproj/crs/wgs84.hpp"

#include <ceres/ceres.h>

namespace zproj::crs {
namespace {

// Identity parameter block, the prior's target.
constexpr std::array<double, 6> kAffineIdentity{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

// Reprojection cost: projects one ground block (lon deg, lat deg, h m)
// through one image's RPC and affine, and compares against the observed
// pixel. Residuals in units of pixel_sigma. Forward evaluation only.
//
// Holds a POINTER to the shared RpcInfo instead of a copy: at 1e5 matches
// there are 2e5 of these functors, and 80 copied coefficients each would
// waste ~130 MB. The pointee (the caller's RpcInfo, passed by reference
// into solve_rpc_affine) outlives the ceres::Problem.
struct ReprojError {
    ReprojError(const RpcInfo* info,
                double obs_col,
                double obs_row,
                double pixel_sigma)
        : info_(info), obs_col_(obs_col), obs_row_(obs_row) {
        inv_sigma_ = 1.0 / ((pixel_sigma > 0.0) ? pixel_sigma : 1.0);
    }

    template<typename T>
    bool operator()(const T* const affine,
                    const T* const lonlath,
                    T* residuals) const {
        T col;
        T row;
        detail::rpc_forward_point_core(
            *info_, lonlath[0], lonlath[1], lonlath[2], col, row);
        const T corr_col = affine[0] + (affine[1] * col) + (affine[2] * row);
        const T corr_row = affine[3] + (affine[4] * col) + (affine[5] * row);
        residuals[0] = (corr_col - T(obs_col_)) * T(inv_sigma_);
        residuals[1] = (corr_row - T(obs_row_)) * T(inv_sigma_);
        return true;
    }

    const RpcInfo* info_;
    double obs_col_;
    double obs_row_;
    double inv_sigma_;
};

ceres::CostFunction* MakeReprojCost(const RpcInfo& info,
                                    double obs_col,
                                    double obs_row,
                                    double pixel_sigma) {
    return new ceres::AutoDiffCostFunction<ReprojError, 2, 6, 3>(
        new ReprojError(&info, obs_col, obs_row, pixel_sigma));
}

// Tikhonov prior on one affine block: residuals = w * (p - identity).
struct AffinePrior {
    explicit AffinePrior(double weight) : weight_(weight) {}

    template<typename T>
    bool operator()(const T* const affine, T* residuals) const {
        for (int i = 0; i < 6; ++i) {
            residuals[i] = T(weight_) * (affine[i] - T(kAffineIdentity[i]));
        }
        return true;
    }

    double weight_;
};

// Parameter-block layout [e0, e1, e2, f0, f1, f2]; held-constant indices per
// RpcAffineDoF. Null for Full (no manifold).
std::unique_ptr<ceres::Manifold> MakeAffineManifold(RpcAffineDoF dof) {
    switch (dof) {
        case RpcAffineDoF::Translation:
            return std::make_unique<ceres::SubsetManifold>(
                6, std::vector<int>{1, 2, 4, 5});
        case RpcAffineDoF::TranslationScale:
            return std::make_unique<ceres::SubsetManifold>(
                6, std::vector<int>{2, 5});
        case RpcAffineDoF::Full:
            return nullptr;
    }
    return nullptr;
}

// Loss function factory: Ceres' Problem takes ownership of each residual
// block's loss function, so a FRESH instance must be handed to every
// AddResidualBlock. Null return disables the robust loss.
ceres::LossFunction* NewLoss(const RpcAffineOptions& o) {
    if (o.robust_threshold_px <= 0.0) {
        return nullptr;
    }
    // Residuals are already normalized by pixel_sigma, so divide the pixel
    // threshold by the same sigma to keep its pixel meaning.
    const double sigma = (o.pixel_sigma > 0.0) ? o.pixel_sigma : 1.0;
    return new ceres::HuberLoss(o.robust_threshold_px / sigma);
}

// One observation's squared reprojection error, evaluated outside Ceres (for
// the before/after RMS in the report). `aff` is a 6-block, `lonlath` a 3-block.
double ResidualSq(const RpcInfo& info,
                  const double* aff,
                  const double* lonlath,
                  double obs_col,
                  double obs_row) {
    double col = 0.0;
    double row = 0.0;
    rpc_forward_point(info, lonlath[0], lonlath[1], lonlath[2], col, row);
    const double rc = aff[0] + (aff[1] * col) + (aff[2] * row) - obs_col;
    const double rr = aff[3] + (aff[4] * col) + (aff[5] * row) - obs_row;
    return (rc * rc) + (rr * rr);
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

// Shared solve tail: runs Levenberg-Marquardt and, when Ceres returns a
// usable solution, copies each affine block into its caller-facing RpcAffine.
// `linear_solver_type` is chosen by the caller: DENSE_SCHUR for the two-view
// problem (see below), DENSE_QR for the tiny GCP-only problem.
RpcAffineReport SolveAndReport(
    ceres::Problem& problem,
    const RpcAffineOptions& options,
    ceres::LinearSolverType linear_solver_type,
    const std::vector<std::pair<double*, RpcAffine*>>& affine_blocks) {
    ceres::Solver::Options solver_options;
    // The two-view problem has classic bundle-adjustment sparsity: a handful
    // of affine blocks ("cameras") and one 3-parameter ground block per
    // match ("points"), every residual touching one of each. DENSE_QR
    // factorizes the full (4N) x (12 + 3N) Jacobian -- O(N^3), minutes at
    // N ~ 1e3 and impossible at 1e5. DENSE_SCHUR eliminates the ground
    // blocks and solves the <= 12 x 12 reduced affine system instead: O(N)
    // per iteration.
    solver_options.linear_solver_type = linear_solver_type;
    solver_options.max_num_iterations =
        (options.max_iterations > 0) ? options.max_iterations : 1;
    solver_options.num_threads =
        (options.num_threads > 0)
            ? options.num_threads
            : std::max(1u, std::thread::hardware_concurrency());
    solver_options.minimizer_progress_to_stdout = options.verbose;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    RpcAffineReport report;
    report.ok = summary.IsSolutionUsable();
    report.message = summary.BriefReport();
    if (report.ok) {
        for (const auto& [block, out] : affine_blocks) {
            std::copy(block, block + 6, out->p.begin());
        }
    }
    return report;
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
    // count to reach the floated affine parameters (both images) -- unless a
    // prior is active, whose Tikhonov terms make the normal equations full
    // rank regardless.
    const int n_affine_params = 2 * static_cast<int>(options.dof);
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
    // solve leaves the caller's affines untouched.
    std::array<double, 6> la = left_affine.p;
    std::array<double, 6> ra = right_affine.p;
    ceres::Problem problem;
    problem.AddParameterBlock(la.data(), 6);
    problem.AddParameterBlock(ra.data(), 6);
    if (auto manifold = MakeAffineManifold(options.dof)) {
        problem.SetManifold(la.data(), manifold.release());
    }
    if (auto manifold = MakeAffineManifold(options.dof)) {
        problem.SetManifold(ra.data(), manifold.release());
    }

    for (std::size_t i = 0; i < used.size(); ++i) {
        problem.AddParameterBlock(ground[i].data(), 3);
        problem.AddResidualBlock(
            MakeReprojCost(
                left, used[i].left_col, used[i].left_row, options.pixel_sigma),
            NewLoss(options),
            la.data(),
            ground[i].data());
        problem.AddResidualBlock(MakeReprojCost(right,
                                                used[i].right_col,
                                                used[i].right_row,
                                                options.pixel_sigma),
                                 NewLoss(options),
                                 ra.data(),
                                 ground[i].data());
    }

    // GCP observations: constant ground blocks (the ground coordinates ARE
    // the measurement). Blocks must stay alive for the problem's lifetime.
    std::vector<std::array<double, 3>> gcp_ground;
    gcp_ground.reserve(left_gcps.size() + right_gcps.size());
    const auto AddGcpBlocks =
        [&](const RpcInfo& info, const std::vector<RpcGcp>& gcps, double* aff) {
            for (const RpcGcp& g : gcps) {
                gcp_ground.push_back({g.lon, g.lat, g.height});
                problem.AddParameterBlock(gcp_ground.back().data(), 3);
                problem.SetParameterBlockConstant(gcp_ground.back().data());
                problem.AddResidualBlock(
                    MakeReprojCost(info, g.col, g.row, options.pixel_sigma),
                    NewLoss(options),
                    aff,
                    gcp_ground.back().data());
            }
        };
    AddGcpBlocks(left, left_gcps, la.data());
    AddGcpBlocks(right, right_gcps, ra.data());

    if (options.affine_prior_weight > 0.0) {
        const double w = options.affine_prior_weight;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                new AffinePrior(w)),
            nullptr,
            la.data());
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                new AffinePrior(w)),
            nullptr,
            ra.data());
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

    RpcAffineReport solved =
        SolveAndReport(problem,
                       options,
                       ceres::DENSE_SCHUR,
                       {{la.data(), &left_affine}, {ra.data(), &right_affine}});
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
            NewLoss(options),
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
    RpcAffineReport solved = SolveAndReport(
        problem, options, ceres::DENSE_QR, {{a.data(), &affine}});
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
