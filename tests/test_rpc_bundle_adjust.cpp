// tests/test_rpc_bundle_adjust.cpp
//
// Unit tests for the N-view RPC bundle adjustment (rpc_bundle_adjust.hpp /
// zproj_refine): the control-network generalization of the two-view affine
// solver.
//
// As with test_rpc_affine.cpp, the primary gate is corruption recovery:
// ground points are projected through V synthetic RPC views, each view's
// projections are pushed through a KNOWN per-view affine (optionally plus
// Gaussian noise and outliers), and the solver must recover the affines /
// drive the reprojection RMS to the noise floor. Partial visibility (each
// point seen in a rotating subset of the views) exercises the
// control-network connectivity that distinguishes this from the two-view
// solver, and the zero-mean / GCP / DEM-height tests cover the N-view gauge
// handling.

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_affine.hpp"
#include "zproj/crs/rpc_bundle_adjust.hpp"
#include "ztensor/zt/utility/Log.h"

#include "synthetic_rpc.hpp"

#include <gtest/gtest.h>

namespace {

using zproj::crs::rpc_forward_point;
using zproj::crs::RpcAffine;
using zproj::crs::RpcAffineDoF;
using zproj::crs::RpcBaMeasure;
using zproj::crs::RpcBaOptions;
using zproj::crs::RpcBaPoint;
using zproj::crs::RpcBaReport;
using zproj::crs::RpcInfo;
using zproj::crs::solve_rpc_bundle_adjust;
using zproj_test::MakeNadirInfo;
using zproj_test::MakePoints;
using zproj_test::Pt;

// The bundle-adjust test views: the synthetic nadir base plus the
// position-dependent height coupling and nonlinearity of the two-view
// affine tests, varied per view (lean azimuth and sign, cross-term signs
// and magnitudes) so every view's ground->pixel map differs and the
// per-view affines are identifiable from tie points alone (see
// rpc_affine.hpp: weakly height-coupled networks leave flat gauge
// directions that LM crawls along).
RpcInfo MakeBaViewInfo(int view) {
    RpcInfo info = MakeNadirInfo();
    // Height lean: rotating azimuth, alternating sign, ~11 deg off vertical.
    constexpr double kAzimuths[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    const double lean = 4e-3 * ((view % 2 == 0) ? 1.0 : -1.0);
    info.samp_num_coeff[3] = lean * kAzimuths[view % 4][0];
    info.line_num_coeff[3] = lean * kAzimuths[view % 4][1];
    // Cross terms: the identifiability keys (H, LH, PH).
    info.samp_num_coeff[4] = 1e-2;                              // lon * lat
    info.line_num_coeff[4] = -1e-2;                             // lon * lat
    info.samp_num_coeff[5] = ((view % 3) - 1) * 2e-2;           // lon * height
    info.line_num_coeff[6] = (((view + 1) % 3) - 1) * -1.5e-2;  // lat * height
    info.samp_num_coeff[7] = 2e-2 + 1e-3 * view;                // lon^2
    info.line_num_coeff[8] = 2e-2 - 1e-3 * view;                // lat^2
    return info;
}

std::vector<RpcInfo> MakeBaViews(int num_views) {
    std::vector<RpcInfo> views;
    views.reserve(static_cast<std::size_t>(num_views));
    for (int v = 0; v < num_views; ++v) {
        views.push_back(MakeBaViewInfo(v));
    }
    return views;
}

// Distinct per-view truth affines: translations and ~1e-4-scale linear
// terms, the magnitude a real RPC product's systematic error maps to in
// pixels (same scale as the two-view tests).
RpcAffine MakeTruthAffine(int view) {
    RpcAffine a;
    a.p = {3.0 + 0.5 * view,
           1.0 + 1e-4 * ((view % 3) - 1),
           -1e-4 * ((view % 2) ? 1 : -1),
           -2.0 - 0.25 * view,
           5e-5 * ((view % 3) - 1),
           1.0 + 1e-4 * ((view % 2) ? 1 : 0)};
    return a;
}

std::vector<RpcAffine> MakeTruthAffines(int num_views) {
    std::vector<RpcAffine> truth;
    truth.reserve(static_cast<std::size_t>(num_views));
    for (int v = 0; v < num_views; ++v) {
        truth.push_back(MakeTruthAffine(v));
    }
    return truth;
}

double MaxParamDelta(const RpcAffine& a, const RpcAffine& b) {
    double worst = 0.0;
    for (int i = 0; i < 6; ++i) {
        worst = std::max(worst, std::fabs(a.p[i] - b.p[i]));
    }
    return worst;
}

// The mean of a set of affines (componentwise).
RpcAffine MeanAffine(const std::vector<RpcAffine>& affines) {
    RpcAffine mean;
    for (int k = 0; k < 6; ++k) {
        mean.p[k] = 0.0;
    }
    for (const RpcAffine& a : affines) {
        for (int k = 0; k < 6; ++k) {
            mean.p[k] += a.p[k];
        }
    }
    for (int k = 0; k < 6; ++k) {
        mean.p[k] /= static_cast<double>(affines.size());
    }
    return mean;
}

struct Network {
    std::vector<Pt> pts;             // ground truth behind every control point
    std::vector<RpcBaPoint> points;  // the (corrupted) control network
};

// Project ground points into rotating subsets of the views (each point seen
// in min_views..V views, window starting at i % V -- the connectivity
// pattern a real multi-view match pipeline produces), corrupt each
// projection with the view's truth affine + Gaussian noise, and push the
// first measure of every `outlier_every`-th point +25 px along col.
Network MakeNetwork(const std::vector<RpcInfo>& views,
                    const std::vector<RpcAffine>& truth,
                    std::size_t n,
                    int min_views,
                    double noise_sigma,
                    unsigned seed,
                    int outlier_every = 0) {
    const int num_views = static_cast<int>(views.size());
    Network out;
    out.pts = MakePoints(n, views[0]);
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    out.points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int start =
            static_cast<int>(i % static_cast<std::size_t>(num_views));
        const int count =
            min_views + static_cast<int>(i % static_cast<std::size_t>(
                                                 num_views + 1 - min_views));
        const bool outlier = outlier_every > 0 &&
                             (i % static_cast<std::size_t>(outlier_every) == 0);
        RpcBaPoint pt;
        for (int k = 0; k < count; ++k) {
            const int v = (start + k) % num_views;
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              out.pts[i].lon,
                              out.pts[i].lat,
                              out.pts[i].alt,
                              c,
                              r);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            c += noise(rng);
            r += noise(rng);
            if (outlier && k == 0) {
                c += 25.0;
            }
            pt.measures.push_back(RpcBaMeasure{v, c, r});
        }
        out.points.push_back(pt);
    }
    return out;
}

// GCPs: ground-fixed control points, each observed in two rotating views
// with the truth affine and Gaussian noise applied to the pixels.
std::vector<RpcBaPoint> MakeBaGcps(const std::vector<RpcInfo>& views,
                                   const std::vector<RpcAffine>& truth,
                                   const std::vector<Pt>& pts,
                                   double noise_sigma,
                                   unsigned seed) {
    const int num_views = static_cast<int>(views.size());
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    std::vector<RpcBaPoint> gcps;
    gcps.reserve(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        RpcBaPoint pt;
        pt.ground_fixed = true;
        pt.lon = pts[i].lon;
        pt.lat = pts[i].lat;
        pt.height = pts[i].alt;
        for (int k = 0; k < 2; ++k) {
            const int v = static_cast<int>((i + static_cast<std::size_t>(k)) %
                                           static_cast<std::size_t>(num_views));
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              pt.lon,
                              pt.lat,
                              pt.height,
                              c,
                              r);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            pt.measures.push_back(
                RpcBaMeasure{v, c + noise(rng), r + noise(rng)});
        }
        gcps.push_back(pt);
    }
    return gcps;
}

std::vector<RpcAffine> IdentityAffines(int num_views) {
    return std::vector<RpcAffine>(static_cast<std::size_t>(num_views),
                                  RpcAffine::Identity());
}

// ============================ recovery ====================================

// Noiseless, full visibility, full 6-parameter affines on five views: the
// solver must recover every view's truth affine and drop the reprojection
// RMS to round-off.
TEST(SolveRpcBundleAdjust, RecoversKnownAffinesFiveViews) {
    const std::vector<RpcInfo> views = MakeBaViews(5);
    const std::vector<RpcAffine> truth = MakeTruthAffines(5);
    const Network data = MakeNetwork(views, truth, 150, 5, 0.0, 7);

    std::vector<RpcAffine> out = IdentityAffines(5);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, data.points, out);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 150);
    EXPECT_EQ(report.num_points_skipped, 0);
    EXPECT_EQ(report.num_measures, 150 * 5);
    EXPECT_EQ(report.num_gcps, 0);
    EXPECT_GT(report.rms_before_px, 1.0);
    EXPECT_LT(report.rms_after_px, 1e-6);
    for (int v = 0; v < 5; ++v) {
        EXPECT_LT(MaxParamDelta(out[static_cast<std::size_t>(v)],
                                truth[static_cast<std::size_t>(v)]),
                  1e-4)
            << "view " << v;
    }
}

// Partial visibility (each point in 3..6 of six views, rotating windows):
// the control network stays connected through overlapping subsets and the
// recovery still lands on the truth.
TEST(SolveRpcBundleAdjust, RecoversWithPartialVisibility) {
    const std::vector<RpcInfo> views = MakeBaViews(6);
    const std::vector<RpcAffine> truth = MakeTruthAffines(6);
    const Network data = MakeNetwork(views, truth, 240, 3, 0.0, 11);

    std::vector<RpcAffine> out = IdentityAffines(6);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, data.points, out);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 240);
    EXPECT_LT(report.num_measures, 240 * 6);
    EXPECT_GT(report.num_measures, 240 * 3);
    EXPECT_LT(report.rms_after_px, 1e-6);
    for (int v = 0; v < 6; ++v) {
        EXPECT_LT(MaxParamDelta(out[static_cast<std::size_t>(v)],
                                truth[static_cast<std::size_t>(v)]),
                  1e-4)
            << "view " << v;
    }
}

// Translation-only DoF against pure-shift truths.
TEST(SolveRpcBundleAdjust, RecoversTranslationDof) {
    const std::vector<RpcInfo> views = MakeBaViews(4);
    std::vector<RpcAffine> truth = MakeTruthAffines(4);
    for (int v = 0; v < 4; ++v) {
        truth[static_cast<std::size_t>(v)] = RpcAffine{};
        truth[static_cast<std::size_t>(v)].p = {
            2.0 + v, 1.0, 0.0, -1.0 - 0.5 * v, 0.0, 1.0};
    }
    const Network data = MakeNetwork(views, truth, 64, 4, 0.0, 13);

    RpcBaOptions options;
    options.dof = RpcAffineDoF::Translation;
    std::vector<RpcAffine> out = IdentityAffines(4);
    const RpcBaReport report =
        solve_rpc_bundle_adjust(views, data.points, out, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1e-5);
    for (int v = 0; v < 4; ++v) {
        EXPECT_LT(MaxParamDelta(out[static_cast<std::size_t>(v)],
                                truth[static_cast<std::size_t>(v)]),
                  1e-5)
            << "view " << v;
    }
}

// Noisy matches with a fraction of gross outliers: the Huber loss plus the
// affine prior must keep the recovery near the clean-noise solution instead
// of chasing the outliers (the report RMS ignores the loss, so outliers
// keep it above the noise floor; parameter recovery is the assertion).
TEST(SolveRpcBundleAdjust, RobustToOutliers) {
    const std::vector<RpcInfo> views = MakeBaViews(5);
    const std::vector<RpcAffine> truth = MakeTruthAffines(5);
    const double sigma = 0.2;
    // Every 8th point's first measure is an outlier (+25 px on col).
    const Network data = MakeNetwork(views, truth, 200, 4, sigma, 17, 8);

    RpcBaOptions options;
    options.pixel_sigma = sigma;
    options.robust_threshold_px = 3.0 * sigma;
    options.affine_prior_weight = 1.0;
    std::vector<RpcAffine> out = IdentityAffines(5);
    const RpcBaReport report =
        solve_rpc_bundle_adjust(views, data.points, out, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 5.0);
    for (int v = 0; v < 5; ++v) {
        EXPECT_LT(std::fabs(out[static_cast<std::size_t>(v)].p[0] -
                            truth[static_cast<std::size_t>(v)].p[0]),
                  50.0)
            << "view " << v << " translation e0";
    }
}

// ============================== gauge =====================================

// Zero-mean constraint (no GCP, no prior, noisy): the V affines' MEAN is
// the identity exactly (the constraint's convention), and each view's
// deviation from identity recovers the truth's deviation from the truth's
// mean -- the differential part tie points actually observe.
TEST(SolveRpcBundleAdjust, ZeroMeanConstrainsCommonMode) {
    const std::vector<RpcInfo> views = MakeBaViews(5);
    const std::vector<RpcAffine> truth = MakeTruthAffines(5);
    const RpcAffine truth_mean = MeanAffine(truth);
    const double sigma = 0.3;
    const Network data = MakeNetwork(views, truth, 200, 4, sigma, 29);

    RpcBaOptions options;
    options.pixel_sigma = sigma;
    options.zero_mean_affines = true;
    std::vector<RpcAffine> out = IdentityAffines(5);
    const RpcBaReport report =
        solve_rpc_bundle_adjust(views, data.points, out, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1.6 * sigma);

    const RpcAffine identity = RpcAffine::Identity();
    for (int k = 0; k < 6; ++k) {
        double sum = 0.0;
        for (int v = 0; v < 5; ++v) {
            sum += out[static_cast<std::size_t>(v)].p[k];
        }
        EXPECT_NEAR(sum, 5.0 * identity.p[k], 1e-9) << "mean param " << k;
    }
    // Each view lands at the zero-mean convention: out_v - identity must
    // match truth_v - mean(truth). Heights are free, so the
    // height-vs-differential-translation direction stays weakly observable;
    // with V views there are more differential translations sharing that
    // soft direction than in the two-view case, hence the wider budget
    // (pinning match heights tightens it to 0.5, see below).
    for (int v = 0; v < 5; ++v) {
        for (int k = 0; k < 6; ++k) {
            const double want =
                truth[static_cast<std::size_t>(v)].p[k] - truth_mean.p[k];
            EXPECT_NEAR(out[static_cast<std::size_t>(v)].p[k] - identity.p[k],
                        want,
                        6.0)
                << "view " << v << " differential param " << k;
        }
    }
}

// Zero-mean plus per-point DEM heights: both gauge families removed exactly
// -- the recommended no-GCP configuration (same as the two-view
// PinnedHeightsAnchorDatum).
TEST(SolveRpcBundleAdjust, PinnedHeightsAnchorDatum) {
    const std::vector<RpcInfo> views = MakeBaViews(5);
    const std::vector<RpcAffine> truth = MakeTruthAffines(5);
    const RpcAffine truth_mean = MeanAffine(truth);
    const double sigma = 0.3;
    Network data = MakeNetwork(views, truth, 200, 4, sigma, 31);

    // Sample every point's height from the truth (a perfect DEM; a coarse
    // one would add metre-scale height noise instead).
    for (std::size_t i = 0; i < data.points.size(); ++i) {
        data.points[i].height = data.pts[i].alt;
    }

    RpcBaOptions options;
    options.pixel_sigma = sigma;
    options.zero_mean_affines = true;
    std::vector<RpcAffine> out = IdentityAffines(5);
    const RpcBaReport report =
        solve_rpc_bundle_adjust(views, data.points, out, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1.6 * sigma);
    const RpcAffine identity = RpcAffine::Identity();
    for (int v = 0; v < 5; ++v) {
        for (int k = 0; k < 6; ++k) {
            const double want =
                truth[static_cast<std::size_t>(v)].p[k] - truth_mean.p[k];
            EXPECT_NEAR(out[static_cast<std::size_t>(v)].p[k] - identity.p[k],
                        want,
                        0.5)
                << "view " << v << " differential param " << k;
        }
    }
}

// GCPs anchoring the geodetic datum: ground-fixed points observed in two
// rotating views each make the ABSOLUTE affines observable, so the recovery
// is near the truth with no gauge convention in play (the mean is NOT the
// identity here).
TEST(SolveRpcBundleAdjust, GcpsAnchorAbsolute) {
    const std::vector<RpcInfo> views = MakeBaViews(5);
    const std::vector<RpcAffine> truth = MakeTruthAffines(5);
    const double sigma = 0.3;
    Network data = MakeNetwork(views, truth, 200, 4, sigma, 31);

    // The first 10 ground points double as GCPs (20 pixel observations
    // anchor the common mode; the two-view test uses 16).
    const std::vector<Pt> gcp_pts(data.pts.begin(), data.pts.begin() + 10);
    const std::vector<RpcBaPoint> gcps =
        MakeBaGcps(views, truth, gcp_pts, sigma, 37);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());

    RpcBaOptions options;
    options.pixel_sigma = sigma;
    std::vector<RpcAffine> out = IdentityAffines(5);
    const RpcBaReport report =
        solve_rpc_bundle_adjust(views, data.points, out, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_gcps, 10);
    EXPECT_GT(report.rms_after_px, 0.2 * sigma);
    EXPECT_LT(report.rms_after_px, 1.6 * sigma);
    for (int v = 0; v < 5; ++v) {
        EXPECT_LT(MaxParamDelta(out[static_cast<std::size_t>(v)],
                                truth[static_cast<std::size_t>(v)]),
                  1.0)
            << "view " << v;
    }
}

// GCP-only network (no tie points): each ground-fixed point observed in two
// views pins those views absolutely -- the multi-image generalization of
// the single-image GCP solve, running the dense-QR path (no free ground
// blocks to eliminate).
TEST(SolveRpcBundleAdjust, GcpOnlyRecoversAbsolutely) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    const std::vector<Pt> gcp_pts = MakePoints(9, views[0]);
    // 9 GCPs x 2 measures x 2 residuals = 36 observations >= 18 params.
    const std::vector<RpcBaPoint> gcps =
        MakeBaGcps(views, truth, gcp_pts, 0.0, 43);

    std::vector<RpcAffine> out = IdentityAffines(3);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, gcps, out);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 0);
    EXPECT_EQ(report.num_gcps, 9);
    EXPECT_LT(report.rms_after_px, 1e-6);
    for (int v = 0; v < 3; ++v) {
        EXPECT_LT(MaxParamDelta(out[static_cast<std::size_t>(v)],
                                truth[static_cast<std::size_t>(v)]),
                  1e-5)
            << "view " << v;
    }
}

// The N-view solver at V = 2 must agree with the two-view solve_rpc_affine
// on the same synthetic pair (the two-view mirror and the N-view derived
// block are the same constraint).
TEST(SolveRpcBundleAdjust, TwoViewsMatchPairSolver) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    const Network data = MakeNetwork(views, truth, 128, 2, 0.0, 19);

    std::vector<RpcAffine> out = IdentityAffines(2);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, data.points, out);
    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1e-6);
    for (int v = 0; v < 2; ++v) {
        EXPECT_LT(MaxParamDelta(out[static_cast<std::size_t>(v)],
                                truth[static_cast<std::size_t>(v)]),
                  1e-4)
            << "view " << v;
    }
}

// ======================== validation / failure paths ======================

TEST(SolveRpcBundleAdjust, SingleViewRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(1);
    std::vector<RpcAffine> out = IdentityAffines(1);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, {}, out);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find(">= 2 views"), std::string::npos);
}

TEST(SolveRpcBundleAdjust, AffineCountMismatchRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    std::vector<RpcAffine> out = IdentityAffines(2);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, {}, out);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find("one entry per view"), std::string::npos);
}

TEST(SolveRpcBundleAdjust, EmptyNetworkRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    std::vector<RpcAffine> out = IdentityAffines(3);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, {}, out);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.message.empty());
    // The caller's affines stay untouched.
    for (const RpcAffine& a : out) {
        EXPECT_EQ(a.p, RpcAffine::Identity().p);
    }
}

// A single two-view tie point nets 4 - 3 = 1 constraint against 5 views x
// 6 affine parameters: rejected as under-determined.
TEST(SolveRpcBundleAdjust, UnderDeterminedRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(5);
    const std::vector<RpcAffine> truth = MakeTruthAffines(5);
    const Network data = MakeNetwork(views, truth, 1, 2, 0.0, 23);

    std::vector<RpcAffine> out = IdentityAffines(5);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, data.points, out);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find("under-determined"), std::string::npos);
}

// Malformed points (out-of-range view, single-measure tie) are skipped and
// counted without sinking the solve.
TEST(SolveRpcBundleAdjust, MalformedPointsSkipped) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    const Network data = MakeNetwork(views, truth, 64, 3, 0.0, 5);

    std::vector<RpcBaPoint> points = data.points;
    // A measure pointing at a nonexistent view: the whole point is dropped.
    RpcBaPoint bad_view = points.front();
    bad_view.measures.front().view = 99;
    points.push_back(bad_view);
    // A lone measure (tie point with < 2 measures): dropped.
    RpcBaPoint lone = points.front();
    lone.measures.resize(1);
    points.push_back(lone);

    std::vector<RpcAffine> out = IdentityAffines(3);
    const RpcBaReport report = solve_rpc_bundle_adjust(views, points, out);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 64);
    EXPECT_EQ(report.num_points_skipped, 2);
    EXPECT_LT(report.rms_after_px, 1e-6);
}

// =================== banded (global affine + band shifts) =================

using zproj::crs::RpcBaBand;
using zproj::crs::RpcBaBandBasis;
using zproj::crs::RpcBaBandedOptions;
using zproj::crs::solve_rpc_bundle_adjust_banded;

// The synthetic models' row span: rows land in [0, 2*line_off] = [0, 50000].
constexpr double kRowSpan = 50000.0;

// One full-period row wave: max at the top/bottom rows, min at the middle,
// zeros at the quarter rows -- the observed wavy-error pattern. Evaluated
// at the RAW RPC row (before any affine).
double RowWave(double row, double amp) {
    constexpr double kTwoPi = 6.2831853071795865;
    return amp * std::cos(kTwoPi * row / kRowSpan);
}

// Like MakeNetwork, plus a wave term added to every corrupted pixel (the
// GCP variant observes every scene for a stronger absolute anchor).
Network MakeWaveNetwork(const std::vector<RpcInfo>& views,
                        const std::vector<RpcAffine>& truth,
                        double col_amp,
                        double row_amp,
                        std::size_t n,
                        int min_views,
                        double noise_sigma,
                        unsigned seed) {
    const int num_views = static_cast<int>(views.size());
    Network out;
    out.pts = MakePoints(n, views[0]);
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    out.points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int start =
            static_cast<int>(i % static_cast<std::size_t>(num_views));
        const int count =
            min_views + static_cast<int>(i % static_cast<std::size_t>(
                                                 num_views + 1 - min_views));
        RpcBaPoint pt;
        for (int k = 0; k < count; ++k) {
            const int v = (start + k) % num_views;
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              out.pts[i].lon,
                              out.pts[i].lat,
                              out.pts[i].alt,
                              c,
                              r);
            const double wave_c = RowWave(r, col_amp);
            const double wave_r = RowWave(r, row_amp);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            c += wave_c + noise(rng);
            r += wave_r + noise(rng);
            pt.measures.push_back(RpcBaMeasure{v, c, r});
        }
        out.points.push_back(pt);
    }
    return out;
}

std::vector<RpcBaPoint> MakeWaveGcps(const std::vector<RpcInfo>& views,
                                     const std::vector<RpcAffine>& truth,
                                     double col_amp,
                                     double row_amp,
                                     const std::vector<Pt>& pts,
                                     double noise_sigma,
                                     unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    std::vector<RpcBaPoint> gcps;
    gcps.reserve(pts.size());
    for (const Pt& p : pts) {
        RpcBaPoint pt;
        pt.ground_fixed = true;
        pt.lon = p.lon;
        pt.lat = p.lat;
        pt.height = p.alt;
        for (std::size_t v = 0; v < views.size(); ++v) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[v], p.lon, p.lat, p.alt, c, r);
            const double wave_c = RowWave(r, col_amp);
            const double wave_r = RowWave(r, row_amp);
            truth[v].Apply(c, r, c, r);
            pt.measures.push_back(RpcBaMeasure{static_cast<int>(v),
                                               c + wave_c + noise(rng),
                                               r + wave_r + noise(rng)});
        }
        gcps.push_back(pt);
    }
    return gcps;
}

// Uniform row bands tiling [0, kRowSpan) per scene.
std::vector<RpcBaBand> MakeUniformBands(int num_scenes, int bands_per_scene) {
    std::vector<RpcBaBand> out;
    out.reserve(static_cast<std::size_t>(num_scenes * bands_per_scene));
    for (int s = 0; s < num_scenes; ++s) {
        for (int i = 0; i < bands_per_scene; ++i) {
            out.push_back(RpcBaBand{s,
                                    kRowSpan * static_cast<double>(i) /
                                        static_cast<double>(bands_per_scene),
                                    kRowSpan * static_cast<double>(i + 1) /
                                        static_cast<double>(bands_per_scene)});
        }
    }
    return out;
}

std::vector<std::array<double, 2>> ZeroShifts(std::size_t num_bands) {
    return std::vector<std::array<double, 2>>(num_bands, {0.0, 0.0});
}

// Wave + affine corruption, GCP-anchored, tent basis: the scene affines
// recover the truth affine and the band shifts track the wave at the band
// centers (the parameterization splits scene translation vs zero-mean wave
// exactly, because the uniform centers' wave samples sum to zero).
TEST(SolveRpcBundleAdjustBanded, RecoversWaveTentWithGcps) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    constexpr double kColAmp = 4.0;
    constexpr double kRowAmp = 3.0;
    Network data =
        MakeWaveNetwork(views, truth, kColAmp, kRowAmp, 150, 3, 0.0, 5);
    const std::vector<Pt> gcp_pts(data.pts.begin(), data.pts.begin() + 8);
    const std::vector<RpcBaPoint> gcps =
        MakeWaveGcps(views, truth, kColAmp, kRowAmp, gcp_pts, 0.0, 9);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());

    const std::vector<RpcBaBand> bands = MakeUniformBands(3, 12);
    RpcBaBandedOptions options;
    options.basis = RpcBaBandBasis::Linear;
    std::vector<RpcAffine> aff = IdentityAffines(3);
    std::vector<std::array<double, 2>> shifts = ZeroShifts(bands.size());
    const RpcBaReport report = solve_rpc_bundle_adjust_banded(
        views, bands, data.points, aff, shifts, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 150);
    EXPECT_EQ(report.num_gcps, 8);
    // Tent approximation error ~0.15 px at these amplitudes.
    EXPECT_GT(report.rms_before_px, 3.0);
    EXPECT_LT(report.rms_after_px, 0.3);
    for (int s = 0; s < 3; ++s) {
        EXPECT_LT(MaxParamDelta(aff[static_cast<std::size_t>(s)],
                                truth[static_cast<std::size_t>(s)]),
                  0.5)
            << "scene " << s;
    }
    // The shifts track the wave at the band centers. The budget is above
    // the rms approximation error: near the wave extrema (top/bottom/middle
    // rows, where the curvature peaks) the tent fit locally overshoots and
    // the LS solution re-balances neighboring shifts, so individual control
    // values deviate more than the overall rms suggests.
    for (std::size_t b = 0; b < bands.size(); ++b) {
        const double center = 0.5 * (bands[b].row_lo + bands[b].row_hi);
        EXPECT_NEAR(shifts[b][0], RowWave(center, kColAmp), 0.5)
            << "band " << b << " col shift";
        EXPECT_NEAR(shifts[b][1], RowWave(center, kRowAmp), 0.5)
            << "band " << b << " row shift";
    }
}

// Same wave data through both bases: the C0 tent basis must fit the smooth
// wave strictly better than the piecewise-constant one.
TEST(SolveRpcBundleAdjustBanded, TentBeatsConstant) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    Network data = MakeWaveNetwork(views, truth, 4.0, 3.0, 150, 3, 0.0, 5);
    const std::vector<Pt> gcp_pts(data.pts.begin(), data.pts.begin() + 8);
    const std::vector<RpcBaPoint> gcps =
        MakeWaveGcps(views, truth, 4.0, 3.0, gcp_pts, 0.0, 9);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());
    const std::vector<RpcBaBand> bands = MakeUniformBands(3, 12);

    double rms[2] = {0.0, 0.0};
    const RpcBaBandBasis bases[2] = {RpcBaBandBasis::Linear,
                                     RpcBaBandBasis::Constant};
    for (int i = 0; i < 2; ++i) {
        RpcBaBandedOptions options;
        options.basis = bases[i];
        std::vector<RpcAffine> aff = IdentityAffines(3);
        std::vector<std::array<double, 2>> shifts = ZeroShifts(bands.size());
        const RpcBaReport report = solve_rpc_bundle_adjust_banded(
            views, bands, data.points, aff, shifts, options);
        ASSERT_TRUE(report.ok) << report.message;
        rms[i] = report.rms_after_px;
    }
    EXPECT_LT(rms[0], rms[1]);
    EXPECT_LT(rms[1], 1.5);
}

// A single band per scene carries no shift structure: the banded solve
// degenerates to the plain per-view solve (shifts pinned to zero by the
// zero-mean-per-scene parameterization).
TEST(SolveRpcBundleAdjustBanded, SingleBandMatchesPlainSolver) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    const Network data = MakeNetwork(views, truth, 128, 3, 0.0, 19);

    std::vector<RpcAffine> plain = IdentityAffines(3);
    ASSERT_TRUE(solve_rpc_bundle_adjust(views, data.points, plain).ok);

    const std::vector<RpcBaBand> bands = MakeUniformBands(3, 1);
    std::vector<RpcAffine> banded = IdentityAffines(3);
    std::vector<std::array<double, 2>> shifts = ZeroShifts(3);
    const RpcBaReport report = solve_rpc_bundle_adjust_banded(
        views, bands, data.points, banded, shifts);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1e-6);
    for (int s = 0; s < 3; ++s) {
        EXPECT_LT(MaxParamDelta(plain[static_cast<std::size_t>(s)],
                                banded[static_cast<std::size_t>(s)]),
                  1e-5)
            << "scene " << s;
        EXPECT_EQ(shifts[static_cast<std::size_t>(s)][0], 0.0);
        EXPECT_EQ(shifts[static_cast<std::size_t>(s)][1], 0.0);
    }
}

// No-GCP banded solve: the per-row-region common modes (bands of all
// scenes drifting together, absorbed by the ground blocks) are invisible
// to zero-mean -- the band-shift prior is the anchor, per-point DEM
// heights pin the height direction. The exact constraints hold by
// construction: the scene affines' mean is the identity and every scene's
// band shifts sum to zero.
TEST(SolveRpcBundleAdjustBanded, ZeroMeanPlusDemAnchors) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    const double sigma = 0.3;
    Network data = MakeWaveNetwork(views, truth, 4.0, 3.0, 120, 2, sigma, 31);
    for (std::size_t i = 0; i < data.points.size(); ++i) {
        data.points[i].height = data.pts[i].alt;  // perfect DEM
    }

    const std::vector<RpcBaBand> bands = MakeUniformBands(2, 8);
    RpcBaBandedOptions options;
    options.pixel_sigma = sigma;
    options.robust_threshold_px = 3.0 * sigma;
    options.zero_mean_affines = true;
    options.band_shift_prior_weight = 1.0;
    std::vector<RpcAffine> aff = IdentityAffines(2);
    std::vector<std::array<double, 2>> shifts = ZeroShifts(bands.size());
    const RpcBaReport report = solve_rpc_bundle_adjust_banded(
        views, bands, data.points, aff, shifts, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1.5);
    const RpcAffine identity = RpcAffine::Identity();
    for (int k = 0; k < 6; ++k) {
        EXPECT_NEAR(aff[0].p[static_cast<std::size_t>(k)] +
                        aff[1].p[static_cast<std::size_t>(k)],
                    2.0 * identity.p[static_cast<std::size_t>(k)],
                    1e-9)
            << "scene affine mean param " << k;
    }
    for (int s = 0; s < 2; ++s) {
        double sx = 0.0;
        double sy = 0.0;
        for (std::size_t b = 0; b < bands.size(); ++b) {
            if (bands[b].scene != s) {
                continue;
            }
            sx += shifts[b][0];
            sy += shifts[b][1];
        }
        EXPECT_NEAR(sx, 0.0, 1e-9) << "scene " << s << " shifts sum (col)";
        EXPECT_NEAR(sy, 0.0, 1e-9) << "scene " << s << " shifts sum (row)";
    }
}

// ======================== validation / failure paths ======================

TEST(SolveRpcBundleAdjustBanded, BadBandSceneRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    std::vector<RpcBaBand> bands = MakeUniformBands(2, 4);
    bands.back().scene = 99;
    std::vector<RpcAffine> aff = IdentityAffines(2);
    std::vector<std::array<double, 2>> shifts = ZeroShifts(bands.size());
    const RpcBaReport report =
        solve_rpc_bundle_adjust_banded(views, bands, {}, aff, shifts);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find("out of range"), std::string::npos);
}

TEST(SolveRpcBundleAdjustBanded, BadBandRangeRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    std::vector<RpcBaBand> bands = MakeUniformBands(2, 4);
    bands.front().row_hi = bands.front().row_lo;
    std::vector<RpcAffine> aff = IdentityAffines(2);
    std::vector<std::array<double, 2>> shifts = ZeroShifts(bands.size());
    const RpcBaReport report =
        solve_rpc_bundle_adjust_banded(views, bands, {}, aff, shifts);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find("empty row range"), std::string::npos);
}

TEST(SolveRpcBundleAdjustBanded, SizeMismatchRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcBaBand> bands = MakeUniformBands(2, 4);
    std::vector<RpcAffine> aff = IdentityAffines(2);
    std::vector<std::array<double, 2>> shifts = ZeroShifts(bands.size());
    std::vector<RpcAffine> wrong_aff = IdentityAffines(3);
    const RpcBaReport bad_affines =
        solve_rpc_bundle_adjust_banded(views, bands, {}, wrong_aff, shifts);
    EXPECT_FALSE(bad_affines.ok);
    EXPECT_NE(bad_affines.message.find("one entry per scene"),
              std::string::npos);
    std::vector<std::array<double, 2>> wrong_shifts =
        ZeroShifts(bands.size() + 1);
    const RpcBaReport bad_shifts =
        solve_rpc_bundle_adjust_banded(views, bands, {}, aff, wrong_shifts);
    EXPECT_FALSE(bad_shifts.ok);
    EXPECT_NE(bad_shifts.message.find("one entry per band"), std::string::npos);
}

// Measures referencing a nonexistent scene drop their point without
// sinking the solve; rows outside the bands' range clamp to the nearest
// band.
TEST(SolveRpcBundleAdjustBanded, MalformedInputSkipped) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    const Network data = MakeNetwork(views, truth, 64, 3, 0.0, 5);

    std::vector<RpcBaPoint> points = data.points;
    RpcBaPoint bad = points.front();
    bad.measures.front().view = 99;
    points.push_back(bad);

    // Bands covering only the middle rows: top/bottom measures clamp.
    std::vector<RpcBaBand> bands;
    for (int s = 0; s < 3; ++s) {
        for (int i = 0; i < 6; ++i) {
            bands.push_back(
                RpcBaBand{s, 10000.0 + 5000.0 * i, 15000.0 + 5000.0 * i});
        }
    }
    std::vector<RpcAffine> aff = IdentityAffines(3);
    std::vector<std::array<double, 2>> shifts = ZeroShifts(bands.size());
    const RpcBaReport report =
        solve_rpc_bundle_adjust_banded(views, bands, points, aff, shifts);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 64);
    EXPECT_EQ(report.num_points_skipped, 1);
}

}  // namespace

int main(int argc, char** argv) {
    zt::Logger::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
