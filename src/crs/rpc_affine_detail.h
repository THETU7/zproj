// Shared internals of the zproj_refine solvers (rpc_affine.cpp's two-view /
// single-image solves and rpc_bundle_adjust.cpp's N-view bundle adjustment):
// the autodiff reprojection cost, the affine prior / manifold / loss
// factories, and the report-tail Levenberg-Marquardt driver. Private to the
// library -- no public header includes this, so no Ceres dependency leaks
// out of zproj_refine.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_affine.hpp"

#include <ceres/ceres.h>

namespace zproj::crs::refine_detail {

// Identity parameter block, the prior's target and the zero-mean gauge's
// reference.
inline constexpr std::array<double, 6> kAffineIdentity{
    0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

// Reprojection cost: projects one ground block (lon deg, lat deg, h m)
// through one image's RPC and affine, and compares against the observed
// pixel. Residuals in units of pixel_sigma. Forward evaluation only.
//
// Holds a POINTER to the shared RpcInfo instead of a copy: at 1e5 matches
// there are 2e5 of these functors, and 80 copied coefficients each would
// waste ~130 MB. The pointee (the caller's RpcInfo, passed by reference
// into the solve) outlives the ceres::Problem.
//
// `mirror` implements the two-view zero-mean constraint: the affine applied
// is 2*identity - `affine`, i.e. the right image's block is parameterized as
// the mirror of the left's, so the two affines' MEAN is exactly the
// identity without any soft-constraint weight tuning. The N-view
// generalization (sum p_i = V*identity) lives in rpc_bundle_adjust.cpp.
struct ReprojError {
    ReprojError(const RpcInfo* info,
                double obs_col,
                double obs_row,
                double pixel_sigma,
                bool mirror = false)
        : info_(info), obs_col_(obs_col), obs_row_(obs_row), mirror_(mirror) {
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
        const T e0 =
            mirror_ ? (T(2.0 * kAffineIdentity[0]) - affine[0]) : affine[0];
        const T e1 =
            mirror_ ? (T(2.0 * kAffineIdentity[1]) - affine[1]) : affine[1];
        const T e2 =
            mirror_ ? (T(2.0 * kAffineIdentity[2]) - affine[2]) : affine[2];
        const T f0 =
            mirror_ ? (T(2.0 * kAffineIdentity[3]) - affine[3]) : affine[3];
        const T f1 =
            mirror_ ? (T(2.0 * kAffineIdentity[4]) - affine[4]) : affine[4];
        const T f2 =
            mirror_ ? (T(2.0 * kAffineIdentity[5]) - affine[5]) : affine[5];
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
    bool mirror_;
};

inline ceres::CostFunction* MakeReprojCost(const RpcInfo& info,
                                           double obs_col,
                                           double obs_row,
                                           double pixel_sigma,
                                           bool mirror = false) {
    return new ceres::AutoDiffCostFunction<ReprojError, 2, 6, 3>(
        new ReprojError(&info, obs_col, obs_row, pixel_sigma, mirror));
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

// Identity prior with residuals in PIXEL units -- the one that can
// actually pin a satellite affine. The plain AffinePrior weights the
// raw parameters, but they live at wildly different scales (translations
// ~0, linear terms ~1 with physically meaningful deviations ~1e-4), so a
// unit weight barely touches a tilt that moves image edges by whole
// pixels. Here each deviation is mapped to the pixel shift it causes at
// the RPC validity domain's edge first:
//
//     translations e0/f0:  the shift itself [px]
//     col-linear e1/f1:    deviation * col_half  (px at the domain edge)
//     row-linear e2/f2:    deviation * row_half
//
// (`col_half`/`row_half` are samp_scale/line_scale: the domain edge sits
// one scale from the offset.) Residual = deviation_px / sigma_px, so e.g.
// sigma = 2 px means "a 2-sigma solution tilts the scene edge by 4 px" --
// the right knob for RPC products whose absolute georeferencing is near
// correct and only needs a few-pixel correction.
struct IdentityPriorPx {
    IdentityPriorPx(double sigma_px, double col_half, double row_half)
        : inv_sigma_(1.0 / ((sigma_px > 0.0) ? sigma_px : 1.0)),
          col_half_(col_half),
          row_half_(row_half) {}

    template<typename T>
    bool operator()(const T* const affine, T* residuals) const {
        residuals[0] = T(inv_sigma_) * (affine[0] - T(kAffineIdentity[0]));
        residuals[1] =
            T(inv_sigma_) * T(col_half_) * (affine[1] - T(kAffineIdentity[1]));
        residuals[2] =
            T(inv_sigma_) * T(row_half_) * (affine[2] - T(kAffineIdentity[2]));
        residuals[3] = T(inv_sigma_) * (affine[3] - T(kAffineIdentity[3]));
        residuals[4] =
            T(inv_sigma_) * T(col_half_) * (affine[4] - T(kAffineIdentity[4]));
        residuals[5] =
            T(inv_sigma_) * T(row_half_) * (affine[5] - T(kAffineIdentity[5]));
        return true;
    }

    double inv_sigma_;
    double col_half_;
    double row_half_;
};

// An affine's largest deviation from `other`, in PIXEL units at the
// domain edge (same mapping as IdentityPriorPx) -- the fair way to
// compare two affines' practical difference (translations and tilts on
// one scale).
inline double AffineDeltaPx(const RpcAffine& a,
                            const RpcAffine& b,
                            double col_half,
                            double row_half) {
    const double scale[6] = {1.0, col_half, row_half, 1.0, col_half, row_half};
    double worst = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double d = std::fabs(a.p[static_cast<std::size_t>(i)] -
                                   b.p[static_cast<std::size_t>(i)]) *
                         scale[static_cast<std::size_t>(i)];
        worst = std::max(worst, d);
    }
    return worst;
}

// Parameter-block layout [e0, e1, e2, f0, f1, f2]; held-constant indices per
// RpcAffineDoF. Null for Full (no manifold).
inline std::unique_ptr<ceres::Manifold> MakeAffineManifold(RpcAffineDoF dof) {
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
inline ceres::LossFunction* NewLoss(RpcLossKind kind,
                                    double robust_threshold_px,
                                    double pixel_sigma) {
    // Residuals are already normalized by pixel_sigma, so divide the pixel
    // threshold by the same sigma to keep its pixel meaning.
    const double a =
        robust_threshold_px / ((pixel_sigma > 0.0) ? pixel_sigma : 1.0);
    switch (kind) {
        case RpcLossKind::Huber:
            return (robust_threshold_px > 0.0) ? new ceres::HuberLoss(a)
                                               : nullptr;
        case RpcLossKind::Cauchy:
            return (robust_threshold_px > 0.0) ? new ceres::CauchyLoss(a)
                                               : nullptr;
        case RpcLossKind::Tukey:
            return (robust_threshold_px > 0.0) ? new ceres::TukeyLoss(a)
                                               : nullptr;
        case RpcLossKind::None:
            break;
    }
    return nullptr;
}

// One observation's squared reprojection error, evaluated outside Ceres (for
// the before/after RMS in the reports). `aff` is a 6-block, `lonlath` a
// 3-block.
inline double ResidualSq(const RpcInfo& info,
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

// Shared solve tail: runs Levenberg-Marquardt and, when Ceres returns a
// usable solution, copies each affine block into its caller-facing
// RpcAffine. `linear_solver_type` is chosen by the caller: DENSE_SCHUR for
// problems with free ground blocks (the classic cameras-vs-points sparsity;
// ASP uses DENSE_SCHUR below ~100 cameras), DENSE_QR for the tiny
// GCP-only problem. `ordering`, when set, names the Schur ordering (group 0
// = eliminated ground blocks); the N-view solver passes one explicitly so
// the elimination partition is deterministic.
inline RpcAffineReport SolveAndReport(
    ceres::Problem& problem,
    int max_iterations,
    int num_threads,
    bool verbose,
    ceres::LinearSolverType linear_solver_type,
    const std::vector<std::pair<double*, RpcAffine*>>& affine_blocks,
    std::unique_ptr<ceres::ParameterBlockOrdering> ordering = nullptr) {
    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = linear_solver_type;
    solver_options.max_num_iterations =
        (max_iterations > 0) ? max_iterations : 1;
    solver_options.num_threads =
        (num_threads > 0) ? num_threads
                          : static_cast<int>(std::max(
                                1u, std::thread::hardware_concurrency()));
    solver_options.minimizer_progress_to_stdout = verbose;
    if (ordering) {
        solver_options.linear_solver_ordering = std::move(ordering);
    }

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

}  // namespace zproj::crs::refine_detail
