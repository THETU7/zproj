// tests/test_rpc_affine.cpp
//
// Unit tests for the RPC image-space affine bias correction
// (rpc_affine.hpp / zproj_refine).
//
// The primary gate is corruption recovery: ground points are projected
// through two synthetic RPC images, the projections are pushed through a
// KNOWN affine per image (optionally plus Gaussian noise and outliers), and
// the solver must recover the affines / drive the reprojection RMS to the
// noise floor. Because the ground truth affine is under the test's control,
// these tests verify parameter recovery -- a stronger statement than
// reprojection-RMS alone.
//
// The affine-test models (MakeAffineNadirInfo / MakeAffineObliqueInfo) add
// realistic nonlinearity to the plain synthetic pair, most importantly the
// position-dependent height coupling that makes the affine identifiable
// from matches alone -- see the comment there and in rpc_affine.hpp.

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_affine.hpp"
#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/wgs84.hpp"
#include "ztensor/zt/utility/Log.h"

#include "synthetic_rpc.hpp"

#include <gtest/gtest.h>

namespace {

using zproj::crs::kRpcAffineE0;
using zproj::crs::kRpcAffineE1;
using zproj::crs::kRpcAffineE2;
using zproj::crs::kRpcAffineF0;
using zproj::crs::kRpcAffineF1;
using zproj::crs::kRpcAffineF2;
using zproj::crs::rpc_forward_point;
using zproj::crs::rpc_forward_point_affine;
using zproj::crs::RpcAffine;
using zproj::crs::RpcAffineDoF;
using zproj::crs::RpcAffineOptions;
using zproj::crs::RpcAffineReport;
using zproj::crs::RpcGcp;
using zproj::crs::RpcInfo;
using zproj::crs::RpcInverseInit;
using zproj::crs::RpcMatch;
using zproj::crs::RpcModel;
using zproj::crs::solve_rpc_affine;
using zproj_test::MakeNadirInfo;
using zproj_test::MakePoints;
using zproj_test::Pt;

// The affine-test pair: the plain synthetic models plus realistic
// nonlinearity and -- crucially -- realistic height sensitivity. Identifying
// the affine from matches alone needs the ground->pixel maps to observe
// height strongly and position-dependently: a height-LINEAR pair (constant
// height coefficient) leaves an exact gauge (uniform height shift absorbed
// by both images' affine translations), and weak height coupling leaves a
// near-flat valley that LM crawls along. Real off-nadir RPC products couple
// height into both images at ~px-per-metre scale through the H and LH/PH
// cross terms, like this pair.
RpcInfo MakeAffineNadirInfo() {
    RpcInfo info = MakeNadirInfo();
    info.samp_num_coeff[3] = -5e-4;    // height (opposite lean to the oblique)
    info.samp_num_coeff[4] = 1e-2;     // lon * lat
    info.samp_num_coeff[5] = 2e-2;     // lon * height
    info.samp_num_coeff[7] = 2e-2;     // lon^2
    info.line_num_coeff[4] = 1e-2;     // lon * lat
    info.line_num_coeff[6] = -1.5e-2;  // lat * height
    info.line_num_coeff[8] = 2e-2;     // lat^2
    return info;
}

RpcInfo MakeAffineObliqueInfo() {
    RpcInfo info = MakeAffineNadirInfo();
    info.samp_num_coeff[3] = 5e-3;    // height lean (off-nadir, opposite sign)
    info.samp_num_coeff[5] = -2e-2;   // lon * height
    info.samp_num_coeff[7] = 3e-2;    // lon^2
    info.samp_num_coeff[9] = 1e-3;    // height^2
    info.line_num_coeff[6] = 1.5e-2;  // lat * height
    return info;
}

RpcAffine MakeAffine(
    double e0, double e1, double e2, double f0, double f1, double f2) {
    RpcAffine a;
    a.p = {e0, e1, e2, f0, f1, f2};
    return a;
}

double MaxParamDelta(const RpcAffine& a, const RpcAffine& b) {
    double worst = 0.0;
    for (int i = 0; i < 6; ++i) {
        worst = std::max(worst, std::fabs(a.p[i] - b.p[i]));
    }
    return worst;
}

// Project ground points through both models and corrupt the pixels with the
// given affines (+ Gaussian pixel noise, and optionally every k-th point
// pushed 25 px along right-col as an outlier).
struct Corrupted {
    std::vector<Pt> pts;
    std::vector<RpcMatch> matches;
};

Corrupted MakeCorrupted(const RpcInfo& left,
                        const RpcInfo& right,
                        const RpcAffine& left_truth,
                        const RpcAffine& right_truth,
                        std::size_t n,
                        double noise_sigma,
                        unsigned seed,
                        int outlier_every = 0) {
    Corrupted out;
    out.pts = MakePoints(n, left);
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    out.matches.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double cl = 0.0;
        double rl = 0.0;
        double cr = 0.0;
        double rr = 0.0;
        rpc_forward_point(
            left, out.pts[i].lon, out.pts[i].lat, out.pts[i].alt, cl, rl);
        rpc_forward_point(
            right, out.pts[i].lon, out.pts[i].lat, out.pts[i].alt, cr, rr);
        left_truth.Apply(cl, rl, cl, rl);
        right_truth.Apply(cr, rr, cr, rr);
        cl += noise(rng);
        rl += noise(rng);
        cr += noise(rng);
        rr += noise(rng);
        if (outlier_every > 0 &&
            i % static_cast<std::size_t>(outlier_every) == 0) {
            cr += 25.0;
        }
        out.matches.push_back(RpcMatch{cl, rl, cr, rr});
    }
    return out;
}

std::vector<RpcGcp> MakeGcps(const RpcInfo& info,
                             const RpcAffine& truth,
                             const std::vector<Pt>& pts,
                             double noise_sigma,
                             unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    std::vector<RpcGcp> gcps;
    gcps.reserve(pts.size());
    for (const Pt& p : pts) {
        double c = 0.0;
        double r = 0.0;
        rpc_forward_point(info, p.lon, p.lat, p.alt, c, r);
        truth.Apply(c, r, c, r);
        gcps.push_back(
            RpcGcp{p.lon, p.lat, p.alt, c + noise(rng), r + noise(rng)});
    }
    return gcps;
}

// ======================= RpcAffine struct semantics =======================

TEST(RpcAffineTest, ApplyUnapplyRoundtrip) {
    const RpcAffine a = MakeAffine(3.5, 1.0003, -1e-4, -2.2, 8e-5, 0.9999);
    double cols[4] = {0.0, 50000.0, -12.5, 1e6};
    double rows[4] = {0.0, 25000.0, 3.25, -7e5};
    for (int i = 0; i < 4; ++i) {
        double c = 0.0;
        double r = 0.0;
        a.Apply(cols[i], rows[i], c, r);
        double back_c = 0.0;
        double back_r = 0.0;
        ASSERT_TRUE(a.Unapply(c, r, back_c, back_r));
        EXPECT_NEAR(back_c, cols[i], 1e-6);
        EXPECT_NEAR(back_r, rows[i], 1e-6);
    }
    EXPECT_NE(a.det(), 0.0);
}

TEST(RpcAffineTest, SingularUnapplyFails) {
    RpcAffine a = MakeAffine(0.0, 1.0, 1.0, 0.0, 1.0, 1.0);  // det = 0
    EXPECT_EQ(a.det(), 0.0);
    double c = 0.0;
    double r = 0.0;
    EXPECT_FALSE(a.Unapply(1.0, 1.0, c, r));
}

TEST(RpcAffineTest, ForwardAffineComposes) {
    // rpc_forward_point_affine must equal forward-then-Apply.
    const RpcInfo info = MakeAffineNadirInfo();
    const RpcAffine a = MakeAffine(1.5, 1.001, -2e-4, -0.75, 5e-5, 0.999);
    const std::vector<Pt> pts = MakePoints(16, info);
    for (const Pt& p : pts) {
        double raw_c = 0.0;
        double raw_r = 0.0;
        rpc_forward_point(info, p.lon, p.lat, p.alt, raw_c, raw_r);
        double c = 0.0;
        double r = 0.0;
        a.Apply(raw_c, raw_r, c, r);
        double c2 = 0.0;
        double r2 = 0.0;
        rpc_forward_point_affine(info, a, p.lon, p.lat, p.alt, c2, r2);
        EXPECT_EQ(c2, c);
        EXPECT_EQ(r2, r);
    }
}

TEST(RpcAffineTest, IdentityIsNoop) {
    const RpcInfo info = MakeAffineNadirInfo();
    const RpcAffine id = RpcAffine::Identity();
    const std::vector<Pt> pts = MakePoints(4, info);
    for (const Pt& p : pts) {
        double c1 = 0.0;
        double r1 = 0.0;
        rpc_forward_point(info, p.lon, p.lat, p.alt, c1, r1);
        double c2 = 0.0;
        double r2 = 0.0;
        rpc_forward_point_affine(info, id, p.lon, p.lat, p.alt, c2, r2);
        EXPECT_EQ(c1, c2);
        EXPECT_EQ(r1, r2);
    }
}

// ============================ two-view solve ==============================

// Noiseless, matches only, full 6-parameter affines: the solver must recover
// both truth affines to solver tolerance and drop the reprojection RMS to
// round-off.
TEST(SolveRpcAffineTwoView, RecoversKnownAffineNoNoise) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const RpcAffine truth_l =
        MakeAffine(3.5, 1.0003, -1e-4, -2.2, 8e-5, 0.9999);
    const RpcAffine truth_r =
        MakeAffine(-1.25, 0.9997, 1.5e-4, 4.0, -6e-5, 1.0002);
    const Corrupted data =
        MakeCorrupted(left, right, truth_l, truth_r, 200, 0.0, 7);

    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report =
        solve_rpc_affine(left, right, data.matches, {}, {}, out_l, out_r);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_matches, 200);
    EXPECT_EQ(report.num_matches_skipped, 0);
    EXPECT_GT(report.rms_before_px, 1.0);
    EXPECT_LT(report.rms_after_px, 1e-6);
    EXPECT_LT(MaxParamDelta(out_l, truth_l), 1e-4);
    EXPECT_LT(MaxParamDelta(out_r, truth_r), 1e-4);
}

// Translation-only degrees of freedom against a pure-shift truth.
TEST(SolveRpcAffineTwoView, RecoversTranslationDof) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const RpcAffine truth_l = MakeAffine(5.0, 1.0, 0.0, -3.0, 0.0, 1.0);
    const RpcAffine truth_r = MakeAffine(-1.0, 1.0, 0.0, 2.0, 0.0, 1.0);
    const Corrupted data =
        MakeCorrupted(left, right, truth_l, truth_r, 64, 0.0, 11);

    RpcAffineOptions options;
    options.dof = RpcAffineDoF::Translation;
    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report = solve_rpc_affine(
        left, right, data.matches, {}, {}, out_l, out_r, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1e-5);
    // Tight, but above Ceres' default function-tolerance floor for this
    // parameterization.
    EXPECT_LT(MaxParamDelta(out_l, truth_l), 1e-5);
    EXPECT_LT(MaxParamDelta(out_r, truth_r), 1e-5);
}

// Noisy matches with a fraction of gross outliers: the Huber loss plus a
// small affine prior (see the file comment in rpc_affine.hpp: noisy
// matches-only solves need anchoring) must keep the recovery near the
// clean-noise solution instead of chasing the outliers or drifting along
// the weakly observable gauge.
TEST(SolveRpcAffineTwoView, RobustToOutliers) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const RpcAffine truth_l =
        MakeAffine(3.5, 1.0003, -1e-4, -2.2, 8e-5, 0.9999);
    const RpcAffine truth_r =
        MakeAffine(-1.25, 0.9997, 1.5e-4, 4.0, -6e-5, 1.0002);
    const double sigma = 0.2;
    // Every 8th match is an outlier (+25 px on right col).
    const Corrupted data =
        MakeCorrupted(left, right, truth_l, truth_r, 200, sigma, 13, 8);

    RpcAffineOptions options;
    options.pixel_sigma = sigma;
    options.robust_threshold_px = 3.0 * sigma;
    // The Huber loss down-weights the outliers, which also weakens the
    // data's grip on the gauge direction; a firmer prior compensates.
    options.affine_prior_weight = 1.0;
    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report = solve_rpc_affine(
        left, right, data.matches, {}, {}, out_l, out_r, options);

    ASSERT_TRUE(report.ok) << report.message;
    // The report RMS ignores the loss, so outliers keep it above the noise
    // floor; parameter recovery is what proves the Huber loss rejected the
    // outliers (fitting them would bias the translations by ~3 px, and
    // diverging would move them much further).
    EXPECT_LT(report.rms_after_px, 5.0);
    EXPECT_LT(std::fabs(out_l.p[kRpcAffineE0] - truth_l.p[kRpcAffineE0]), 50.0);
    EXPECT_LT(std::fabs(out_r.p[kRpcAffineE0] - truth_r.p[kRpcAffineE0]), 50.0);
}

// Noise-free matches + GCPs anchoring the geodetic datum: exact recovery
// even though the models' quadratic terms are tiny (the GCPs pin the gauge).
TEST(SolveRpcAffineTwoView, MatchesPlusGcps) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const RpcAffine truth_l =
        MakeAffine(2.0, 1.0005, -2e-4, -1.5, 1e-4, 0.9998);
    const RpcAffine truth_r = MakeAffine(1.0, 0.9995, 1e-4, 3.0, -1e-4, 1.0005);
    const Corrupted data =
        MakeCorrupted(left, right, truth_l, truth_r, 128, 0.0, 17);

    // The first 8 ground points double as GCPs (observed in both images).
    std::vector<Pt> gcp_pts(data.pts.begin(), data.pts.begin() + 8);
    const std::vector<RpcGcp> lg = MakeGcps(left, truth_l, gcp_pts, 0.0, 19);
    const std::vector<RpcGcp> rg = MakeGcps(right, truth_r, gcp_pts, 0.0, 23);

    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report =
        solve_rpc_affine(left, right, data.matches, lg, rg, out_l, out_r);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_gcps, 16);
    EXPECT_LT(report.rms_after_px, 1e-6);
    EXPECT_LT(MaxParamDelta(out_l, truth_l), 1e-4);
    EXPECT_LT(MaxParamDelta(out_r, truth_r), 1e-4);
}

// Noisy matches only, anchored by the recommended small prior: the fit must
// stay at (or, because each match's ground block absorbs most of the noise,
// below) the noise level. Parameter recovery is deliberately NOT asserted
// here: the weakly observable gauge direction makes it noise-realization
// dependent (the MatchesPlusGcps tests cover anchored recovery).
TEST(SolveRpcAffineTwoView, NoisyRmsAtNoiseLevel) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const RpcAffine truth_l =
        MakeAffine(3.5, 1.0003, -1e-4, -2.2, 8e-5, 0.9999);
    const RpcAffine truth_r =
        MakeAffine(-1.25, 0.9997, 1.5e-4, 4.0, -6e-5, 1.0002);
    const double sigma = 0.3;
    const Corrupted data =
        MakeCorrupted(left, right, truth_l, truth_r, 200, sigma, 29);

    RpcAffineOptions options;
    options.pixel_sigma = sigma;
    options.affine_prior_weight = 1e-2;
    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report = solve_rpc_affine(
        left, right, data.matches, {}, {}, out_l, out_r, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1.6 * sigma);
}

// Noisy matches + GCPs anchoring the geodetic datum: now the affine is
// strongly observable and the translations must recover near the per-point
// noise level.
TEST(SolveRpcAffineTwoView, NoisyMatchesPlusGcps) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const RpcAffine truth_l =
        MakeAffine(3.5, 1.0003, -1e-4, -2.2, 8e-5, 0.9999);
    const RpcAffine truth_r =
        MakeAffine(-1.25, 0.9997, 1.5e-4, 4.0, -6e-5, 1.0002);
    const double sigma = 0.3;
    const Corrupted data =
        MakeCorrupted(left, right, truth_l, truth_r, 200, sigma, 31);

    std::vector<Pt> gcp_pts(data.pts.begin(), data.pts.begin() + 8);
    const std::vector<RpcGcp> lg = MakeGcps(left, truth_l, gcp_pts, sigma, 37);
    const std::vector<RpcGcp> rg = MakeGcps(right, truth_r, gcp_pts, sigma, 41);

    RpcAffineOptions options;
    options.pixel_sigma = sigma;
    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report = solve_rpc_affine(
        left, right, data.matches, lg, rg, out_l, out_r, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_GT(report.rms_after_px, 0.2 * sigma);
    EXPECT_LT(report.rms_after_px, 1.6 * sigma);
    EXPECT_LT(MaxParamDelta(out_l, truth_l), 1.0);
    EXPECT_LT(MaxParamDelta(out_r, truth_r), 1.0);
}

// The affine prior must keep a matches-only solve usable even when the
// observations alone under-determine the affine parameters (LM gets a
// Tikhonov-dominated but finite solution instead of being rejected).
TEST(SolveRpcAffineTwoView, PriorRegularizesUnderDetermined) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const RpcAffine truth_l = MakeAffine(3.5, 1.0, 0.0, -2.2, 0.0, 1.0);
    const RpcAffine truth_r = MakeAffine(-1.25, 1.0, 0.0, 4.0, 0.0, 1.0);
    // Translation DoF floats 4 affine params; 2 matches net only 2
    // constraints, so without the prior the solver refuses the problem.
    const Corrupted data =
        MakeCorrupted(left, right, truth_l, truth_r, 2, 0.0, 31);

    RpcAffineOptions options;
    options.dof = RpcAffineDoF::Translation;
    options.affine_prior_weight = 1.0;
    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report = solve_rpc_affine(
        left, right, data.matches, {}, {}, out_l, out_r, options);

    EXPECT_TRUE(report.ok);
}

// ======================== validation / failure paths ======================

TEST(SolveRpcAffineTwoView, UnderDeterminedFails) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    const Corrupted data = MakeCorrupted(
        left, right, RpcAffine::Identity(), RpcAffine::Identity(), 2, 0.0, 37);
    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report =
        solve_rpc_affine(left, right, data.matches, {}, {}, out_l, out_r);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find("under-determined"), std::string::npos);
    // The caller's affines stay untouched.
    EXPECT_EQ(out_l.p, RpcAffine::Identity().p);
    EXPECT_EQ(out_r.p, RpcAffine::Identity().p);
}

TEST(SolveRpcAffineTwoView, EmptyObservationsFail) {
    const RpcInfo left = MakeAffineNadirInfo();
    const RpcInfo right = MakeAffineObliqueInfo();
    RpcAffine out_l;
    RpcAffine out_r;
    const RpcAffineReport report =
        solve_rpc_affine(left, right, {}, {}, {}, out_l, out_r);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.message.empty());
}

// ========================= single-image (GCP) solve ========================

TEST(SolveRpcAffineGcp, RecoversExactNoNoise) {
    const RpcInfo info = MakeAffineNadirInfo();
    const RpcAffine truth = MakeAffine(4.5, 1.0008, -3e-4, -2.0, 2e-4, 0.9995);
    const std::vector<Pt> pts = MakePoints(16, info);
    const std::vector<RpcGcp> gcps = MakeGcps(info, truth, pts, 0.0, 41);

    RpcAffine out;
    const RpcAffineReport report = solve_rpc_affine(info, gcps, out);
    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1e-6);
    EXPECT_LT(MaxParamDelta(out, truth), 1e-6);
}

TEST(SolveRpcAffineGcp, MinimumThreeGcpsFullDof) {
    // Exactly determined: 3 GCPs x 2 residuals = 6 affine parameters.
    const RpcInfo info = MakeAffineNadirInfo();
    const RpcAffine truth = MakeAffine(1.5, 1.0005, -1e-4, 0.5, 1e-4, 0.9999);
    const std::vector<Pt> pts = MakePoints(3, info);
    const std::vector<RpcGcp> gcps = MakeGcps(info, truth, pts, 0.0, 43);

    RpcAffine out;
    const RpcAffineReport report = solve_rpc_affine(info, gcps, out);
    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(MaxParamDelta(out, truth), 1e-4);
}

TEST(SolveRpcAffineGcp, NoisyRmsAtNoiseLevel) {
    const RpcInfo info = MakeAffineNadirInfo();
    const RpcAffine truth = MakeAffine(4.5, 1.0008, -3e-4, -2.0, 2e-4, 0.9995);
    const double sigma = 0.25;
    const std::vector<Pt> pts = MakePoints(64, info);
    const std::vector<RpcGcp> gcps = MakeGcps(info, truth, pts, sigma, 47);

    RpcAffineOptions options;
    options.pixel_sigma = sigma;
    RpcAffine out;
    const RpcAffineReport report = solve_rpc_affine(info, gcps, out, options);
    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_GT(report.rms_after_px, 0.2 * sigma);
    EXPECT_LT(report.rms_after_px, 1.6 * sigma);
}

TEST(SolveRpcAffineGcp, TooFewGcpsFail) {
    const RpcInfo info = MakeAffineNadirInfo();
    const RpcAffine truth = MakeAffine(1.0, 1.0, 0.0, 2.0, 0.0, 1.0);
    const std::vector<Pt> pts = MakePoints(2, info);
    const std::vector<RpcGcp> gcps = MakeGcps(info, truth, pts, 0.0, 53);

    RpcAffine out;
    const RpcAffineReport report = solve_rpc_affine(info, gcps, out);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find("under-determined"), std::string::npos);
    EXPECT_EQ(out.p, RpcAffine::Identity().p);
}

TEST(SolveRpcAffineGcp, EmptyFails) {
    const RpcInfo info = MakeAffineNadirInfo();
    RpcAffine out;
    const RpcAffineReport report = solve_rpc_affine(info, {}, out);
    EXPECT_FALSE(report.ok);
}

}  // namespace

int main(int argc, char** argv) {
    zt::Logger::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
