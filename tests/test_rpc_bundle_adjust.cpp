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

#include <algorithm>
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

// ============== gridded (global affine + cell shifts) ==============

using zproj::crs::RpcAffineGridBasis;
using zproj::crs::RpcAffineGridded;
using zproj::crs::RpcBaGridOptions;
using zproj::crs::RpcBaTwoStageReport;
using zproj::crs::solve_rpc_bundle_adjust_gridded;

// The synthetic models' col span: cols land in
// [samp_off - 0.5*samp_scale, samp_off + 0.5*samp_scale] = [25000, 75000].
constexpr double kColLo = 25000.0;
constexpr double kColHi = 75000.0;

// One full-period col wave: max at the left/right cols, min at the middle,
// zeros at the quarter cols -- the cross-track error pattern. Evaluated at
// the RAW RPC col (before any affine).
double ColWave(double col, double amp) {
    constexpr double kTwoPi = 6.2831853071795865;
    return amp * std::cos(kTwoPi * (col - kColLo) / (kColHi - kColLo));
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
            const double wave_c = ColWave(c, col_amp);
            const double wave_r = ColWave(c, row_amp);
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
            const double wave_c = ColWave(c, col_amp);
            const double wave_r = ColWave(c, row_amp);
            truth[v].Apply(c, r, c, r);
            pt.measures.push_back(RpcBaMeasure{static_cast<int>(v),
                                               c + wave_c + noise(rng),
                                               r + wave_r + noise(rng)});
        }
        gcps.push_back(pt);
    }
    return gcps;
}

// Point indices at COLUMN strides: project each ground point through view
// 0, sort by col, take `n` even strides. GCPs anchor the per-col-region
// common modes (see rpc_bundle_adjust.hpp), so the anchors must SPREAD
// across the columns, not cluster by luck of the draw.
std::vector<std::size_t> ColStridedIndices(const std::vector<RpcInfo>& views,
                                           const std::vector<Pt>& pts,
                                           std::size_t n) {
    std::vector<std::size_t> order(pts.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    const auto ColIn = [&](std::size_t i) {
        double c = 0.0;
        double r = 0.0;
        rpc_forward_point(views[0], pts[i].lon, pts[i].lat, pts[i].alt, c, r);
        return c;
    };
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return ColIn(a) < ColIn(b);
    });
    std::vector<std::size_t> out;
    out.reserve(n);
    for (std::size_t k = 0; k < n && !order.empty(); ++k) {
        out.push_back(order[(k * order.size()) / n]);
    }
    return out;
}

// The fresh-solve starting point: identity affines with
// `cells_per_scene` uniform zero-shift tent cells over the col span (a
// one-row grid -- pure column banding).
std::vector<RpcAffineGridded> FreshGriddedNet(int num_scenes,
                                              int cells_per_scene,
                                              RpcAffineGridBasis basis) {
    std::vector<RpcAffineGridded> net(static_cast<std::size_t>(num_scenes),
                                      RpcAffineGridded{});
    for (int s = 0; s < num_scenes; ++s) {
        net[static_cast<std::size_t>(s)] = *RpcAffineGridded::MakeUniform(
            RpcAffine::Identity(), kColLo, kColHi, cells_per_scene, basis);
    }
    return net;
}

// Wave + affine corruption, GCP-anchored, tent basis: the scene affines
// recover the truth affine and the band shifts track the wave at the band
// centers (the parameterization splits scene translation vs zero-mean wave
// exactly, because the uniform centers' wave samples sum to zero).
TEST(SolveRpcBundleAdjustGridded, RecoversWaveTentWithGcps) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    constexpr double kColAmp = 4.0;
    constexpr double kRowAmp = 3.0;
    Network data =
        MakeWaveNetwork(views, truth, kColAmp, kRowAmp, 150, 3, 0.0, 5);
    std::vector<Pt> gcp_pts;
    for (std::size_t idx : ColStridedIndices(views, data.pts, 8)) {
        gcp_pts.push_back(data.pts[idx]);
    }
    const std::vector<RpcBaPoint> gcps =
        MakeWaveGcps(views, truth, kColAmp, kRowAmp, gcp_pts, 0.0, 9);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());

    std::vector<RpcAffineGridded> net =
        FreshGriddedNet(3, 12, RpcAffineGridBasis::Linear);
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, data.points, net);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 150);
    EXPECT_EQ(report.num_gcps, 8);
    // Tent approximation error ~0.15 px at these amplitudes.
    EXPECT_GT(report.rms_before_px, 3.0);
    EXPECT_LT(report.rms_after_px, 0.3);
    // The COMPOSITE recovers truth affine + wave. The decomposition itself
    // is not the metric: an affine translation+tilt trades exactly against
    // a zero-mean ramp in the band shifts (the e1/f1 col-linear terms
    // against the ramp's slope, the translations against its constant
    // part), a pure gauge no data can see -- the documented property, the
    // same invariant the example's worst-case check uses. Budget above the
    // rms approximation error: near the wave extrema (left/right/middle
    // cols, where the curvature peaks) the tent fit locally overshoots, so
    // individual control values deviate more than the overall rms.
    for (int s = 0; s < 3; ++s) {
        const RpcAffineGridded& obj = net[static_cast<std::size_t>(s)];
        for (int i = 0; i < obj.num_cells(); ++i) {
            const double center =
                0.5 * (obj.cell(i).col_lo + obj.cell(i).col_hi);
            const RpcAffine eff = obj.EffectiveAffine(center, 25000.0);
            RpcAffine want = truth[static_cast<std::size_t>(s)];
            want.p[0] += ColWave(center, kColAmp);
            want.p[3] += ColWave(center, kRowAmp);
            double ec = 0.0;
            double er = 0.0;
            double wc = 0.0;
            double wr = 0.0;
            eff.Apply(center, 25000.0, ec, er);
            want.Apply(center, 25000.0, wc, wr);
            EXPECT_NEAR(ec, wc, 0.5)
                << "scene " << s << " cell " << i << " col";
            EXPECT_NEAR(er, wr, 0.5)
                << "scene " << s << " cell " << i << " row";
        }
    }
}

// Same wave data through both bases: the C0 tent basis must fit the smooth
// wave strictly better than the piecewise-constant one.
TEST(SolveRpcBundleAdjustGridded, TentBeatsConstant) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    Network data = MakeWaveNetwork(views, truth, 4.0, 3.0, 150, 3, 0.0, 5);
    std::vector<Pt> gcp_pts;
    for (std::size_t idx : ColStridedIndices(views, data.pts, 8)) {
        gcp_pts.push_back(data.pts[idx]);
    }
    const std::vector<RpcBaPoint> gcps =
        MakeWaveGcps(views, truth, 4.0, 3.0, gcp_pts, 0.0, 9);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());

    double rms[2] = {0.0, 0.0};
    const RpcAffineGridBasis bases[2] = {RpcAffineGridBasis::Linear,
                                         RpcAffineGridBasis::Constant};
    for (int i = 0; i < 2; ++i) {
        std::vector<RpcAffineGridded> net = FreshGriddedNet(3, 12, bases[i]);
        const RpcBaReport report =
            solve_rpc_bundle_adjust_gridded(views, data.points, net);
        ASSERT_TRUE(report.ok) << report.message;
        rms[i] = report.rms_after_px;
    }
    EXPECT_LT(rms[0], rms[1]);
    EXPECT_LT(rms[1], 1.5);
}

// A single cell per scene carries no shift structure: the gridded solve
// degenerates to the plain per-view solve (shifts pinned to zero by the
// zero-mean-per-scene parameterization).
TEST(SolveRpcBundleAdjustGridded, SingleCellMatchesPlainSolver) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    const Network data = MakeNetwork(views, truth, 128, 3, 0.0, 19);

    std::vector<RpcAffine> plain = IdentityAffines(3);
    ASSERT_TRUE(solve_rpc_bundle_adjust(views, data.points, plain).ok);

    std::vector<RpcAffineGridded> net =
        FreshGriddedNet(3, 1, RpcAffineGridBasis::Linear);
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, data.points, net);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1e-6);
    for (int s = 0; s < 3; ++s) {
        EXPECT_LT(MaxParamDelta(plain[static_cast<std::size_t>(s)],
                                net[static_cast<std::size_t>(s)].affine()),
                  1e-5)
            << "scene " << s;
        EXPECT_EQ(net[static_cast<std::size_t>(s)].cell(0).dx, 0.0);
        EXPECT_EQ(net[static_cast<std::size_t>(s)].cell(0).dy, 0.0);
    }
}

// A scene with NO bands is a plain affine scene and mixes freely with
// gridded ones: scene 0 carries the wave (and absorbs it through its
// bands), scene 1 does not (and needs only its affine).
TEST(SolveRpcBundleAdjustGridded, AffineOnlySceneMixesWithGridded) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    constexpr double kColAmp = 4.0;
    constexpr double kRowAmp = 3.0;
    Network data =
        MakeWaveNetwork(views, truth, kColAmp, kRowAmp, 120, 2, 0.0, 7);
    // Strip the wave from scene 1's measures: rebuild them without it.
    {
        std::mt19937 rng(23);
        std::vector<RpcBaPoint> rebuilt;
        rebuilt.reserve(data.points.size());
        for (std::size_t i = 0; i < data.points.size(); ++i) {
            RpcBaPoint pt;
            for (const RpcBaMeasure& m : data.points[i].measures) {
                double c = 0.0;
                double r = 0.0;
                rpc_forward_point(views[static_cast<std::size_t>(m.view)],
                                  data.pts[i].lon,
                                  data.pts[i].lat,
                                  data.pts[i].alt,
                                  c,
                                  r);
                const double raw_c = c;
                truth[static_cast<std::size_t>(m.view)].Apply(c, r, c, r);
                if (m.view == 0) {
                    c += ColWave(raw_c, kColAmp);
                    r += ColWave(raw_c, kRowAmp);
                }
                pt.measures.push_back(RpcBaMeasure{m.view, c, r});
            }
            rebuilt.push_back(pt);
        }
        data.points = std::move(rebuilt);
    }
    // GCPs observed in both scenes; only scene 0 carries the wave.
    {
        std::vector<Pt> gcp_pts;
        for (std::size_t idx : ColStridedIndices(views, data.pts, 8)) {
            gcp_pts.push_back(data.pts[idx]);
        }
        for (const Pt& p : gcp_pts) {
            RpcBaPoint pt;
            pt.ground_fixed = true;
            pt.lon = p.lon;
            pt.lat = p.lat;
            pt.height = p.alt;
            for (int v = 0; v < 2; ++v) {
                double c = 0.0;
                double r = 0.0;
                rpc_forward_point(views[static_cast<std::size_t>(v)],
                                  pt.lon,
                                  pt.lat,
                                  pt.height,
                                  c,
                                  r);
                const double raw_c = c;
                truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
                if (v == 0) {
                    c += ColWave(raw_c, kColAmp);
                    r += ColWave(raw_c, kRowAmp);
                }
                pt.measures.push_back(RpcBaMeasure{v, c, r});
            }
            data.points.push_back(pt);
        }
    }

    std::vector<RpcAffineGridded> net =
        FreshGriddedNet(2, 10, RpcAffineGridBasis::Linear);
    net[1] = RpcAffineGridded{};  // scene 1: affine only, no bands
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, data.points, net);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 0.3);
    EXPECT_EQ(net[1].num_cells(), 0);
    EXPECT_LT(MaxParamDelta(net[1].affine(), truth[1]), 0.5);
    // Scene 0's COMPOSITE (not its decomposition -- the affine-tilt-vs-
    // shift-ramp gauge, see RecoversWaveTentWithGcps) carries the wave,
    // anchored only by its own GCP measures and the cross-scene ties
    // (scene 1 carries no wave), so the budget is looser than the pure
    // tent approximation error.
    for (int i = 0; i < net[0].num_cells(); ++i) {
        const double center =
            0.5 * (net[0].cell(i).col_lo + net[0].cell(i).col_hi);
        const RpcAffine eff = net[0].EffectiveAffine(center, 25000.0);
        RpcAffine want = truth[0];
        want.p[0] += ColWave(center, kColAmp);
        want.p[3] += ColWave(center, kRowAmp);
        double ec = 0.0;
        double er = 0.0;
        double wc = 0.0;
        double wr = 0.0;
        eff.Apply(center, 25000.0, ec, er);
        want.Apply(center, 25000.0, wc, wr);
        EXPECT_NEAR(ec, wc, 1.0) << "cell " << i << " col";
        EXPECT_NEAR(er, wr, 1.0) << "cell " << i << " row";
    }
}

// No-GCP gridded solve: the per-col-region common modes (cells of all
// scenes drifting together, absorbed by the ground blocks) are invisible
// to zero-mean -- the band-shift prior is the anchor, per-point DEM
// heights pin the height direction. The exact constraints hold by
// construction: the scene affines' mean is the identity and every scene's
// band shifts sum to zero.
TEST(SolveRpcBundleAdjustGridded, ZeroMeanPlusDemAnchors) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    const double sigma = 0.3;
    Network data = MakeWaveNetwork(views, truth, 4.0, 3.0, 120, 2, sigma, 31);
    for (std::size_t i = 0; i < data.points.size(); ++i) {
        data.points[i].height = data.pts[i].alt;  // perfect DEM
    }

    RpcBaGridOptions options;
    options.pixel_sigma = sigma;
    options.robust_threshold_px = 3.0 * sigma;
    options.zero_mean_affines = true;
    options.cell_shift_prior_weight = 1.0;
    std::vector<RpcAffineGridded> net =
        FreshGriddedNet(2, 8, RpcAffineGridBasis::Linear);
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, data.points, net, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 1.5);
    const RpcAffine identity = RpcAffine::Identity();
    for (int k = 0; k < 6; ++k) {
        EXPECT_NEAR(net[0].affine().p[static_cast<std::size_t>(k)] +
                        net[1].affine().p[static_cast<std::size_t>(k)],
                    2.0 * identity.p[static_cast<std::size_t>(k)],
                    1e-9)
            << "scene affine mean param " << k;
    }
    for (int s = 0; s < 2; ++s) {
        double sx = 0.0;
        double sy = 0.0;
        for (int i = 0; i < net[static_cast<std::size_t>(s)].num_cells(); ++i) {
            sx += net[static_cast<std::size_t>(s)].cell(i).dx;
            sy += net[static_cast<std::size_t>(s)].cell(i).dy;
        }
        EXPECT_NEAR(sx, 0.0, 1e-9) << "scene " << s << " shifts sum (col)";
        EXPECT_NEAR(sy, 0.0, 1e-9) << "scene " << s << " shifts sum (row)";
    }
}

// ======================== validation / failure paths ======================

TEST(SolveRpcBundleAdjustGridded, SizeMismatchRejected) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    std::vector<RpcAffineGridded> net =
        FreshGriddedNet(3, 4, RpcAffineGridBasis::Linear);
    const RpcBaReport report = solve_rpc_bundle_adjust_gridded(views, {}, net);
    EXPECT_FALSE(report.ok);
    EXPECT_NE(report.message.find("one entry per scene"), std::string::npos);
}

TEST(SolveRpcBundleAdjustGridded, EmptyScenesRejected) {
    std::vector<RpcAffineGridded> net;
    const RpcBaReport report = solve_rpc_bundle_adjust_gridded({}, {}, net);
    EXPECT_FALSE(report.ok);
    EXPECT_FALSE(report.message.empty());
}

// Measures referencing a nonexistent scene drop their point without
// sinking the solve; columns outside the bands' range clamp to the nearest
// band.
TEST(SolveRpcBundleAdjustGridded, MalformedInputSkipped) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    const Network data = MakeNetwork(views, truth, 64, 3, 0.0, 5);

    std::vector<RpcBaPoint> points = data.points;
    RpcBaPoint bad = points.front();
    bad.measures.front().view = 99;
    points.push_back(bad);

    // Bands covering only the middle columns: left/right measures clamp.
    std::vector<RpcAffineGridded::Cell> middle;
    for (int i = 0; i < 6; ++i) {
        middle.push_back(RpcAffineGridded::Cell{
            35000.0 + 5000.0 * i, 40000.0 + 5000.0 * i, 0.0, 1.0, 0.0, 0.0});
    }
    std::vector<RpcAffineGridded> net;
    for (int s = 0; s < 3; ++s) {
        net.push_back(*RpcAffineGridded::Make(
            RpcAffine::Identity(), middle, RpcAffineGridBasis::Linear));
    }
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, points, net);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.num_points, 64);
    EXPECT_EQ(report.num_points_skipped, 1);
}

// ==================== RpcAffineGridded (consumer side) =====================

// The band-blending semantics: Constant picks the containing (else
// nearest-center) band; Linear interpolates between band centers with
// constant extension; Make sorts the bands and validates; EffectiveAffine
// folds the shift into the translations; Apply is affine-then-shift.
TEST(RpcAffineGridded, BlendingSemantics) {
    using Cell = RpcAffineGridded::Cell;
    // Identity affine + three cells, passed in SHUFFLED order.
    const std::vector<Cell> shuffled{
        Cell{20.0, 30.0, 0.0, 1.0, -2.0, 2.0},
        Cell{0.0, 10.0, 0.0, 1.0, 1.0, -1.0},
        Cell{10.0, 20.0, 0.0, 1.0, 3.0, 1.0},
    };

    const auto constant = RpcAffineGridded::Make(
        RpcAffine::Identity(), shuffled, RpcAffineGridBasis::Constant);
    ASSERT_TRUE(constant.has_value());
    EXPECT_EQ(constant->num_cells(), 3);
    const auto ConstantShift = [&](double col, double dx, double dy) {
        double gx = 99.0;
        double gy = 99.0;
        constant->ShiftAt(col, 0.5, gx, gy);
        EXPECT_NEAR(gx, dx, 1e-12) << "col " << col;
        EXPECT_NEAR(gy, dy, 1e-12) << "col " << col;
    };
    ConstantShift(0.0, 1.0, -1.0);
    ConstantShift(9.9, 1.0, -1.0);
    ConstantShift(10.0, 3.0, 1.0);  // [col_lo, col_hi) containment
    ConstantShift(15.0, 3.0, 1.0);
    ConstantShift(25.0, -2.0, 2.0);
    ConstantShift(-7.0, 1.0, -1.0);  // outside: nearest center is 5
    ConstantShift(40.0, -2.0, 2.0);  // outside: nearest center is 25

    const auto linear = RpcAffineGridded::Make(
        RpcAffine::Identity(), shuffled, RpcAffineGridBasis::Linear);
    ASSERT_TRUE(linear.has_value());
    const auto LinearShift = [&](double col, double dx, double dy) {
        double gx = 99.0;
        double gy = 99.0;
        linear->ShiftAt(col, 0.5, gx, gy);
        EXPECT_NEAR(gx, dx, 1e-12) << "col " << col;
        EXPECT_NEAR(gy, dy, 1e-12) << "col " << col;
    };
    // Centers at 5 / 15 / 25: tent interpolation, constant extension.
    LinearShift(5.0, 1.0, -1.0);
    LinearShift(10.0, 2.0, 0.0);  // midpoint of bands 0 and 1
    LinearShift(15.0, 3.0, 1.0);
    LinearShift(20.0, 0.5, 1.5);  // midpoint of bands 1 and 2
    LinearShift(25.0, -2.0, 2.0);
    LinearShift(-3.0, 1.0, -1.0);
    LinearShift(29.0, -2.0, 2.0);

    // EffectiveAffine folds the shift into the translations only; Apply is
    // the affine followed by the shift.
    RpcAffine a;
    a.p = {2.0, 1.001, 1e-4, -1.0, 2e-5, 0.999};
    const auto shifted = RpcAffineGridded::Make(
        a,
        std::vector<Cell>{Cell{0.0, 100.0, 0.0, 1.0, 4.0, -3.0}},
        RpcAffineGridBasis::Linear);
    ASSERT_TRUE(shifted.has_value());
    const RpcAffine eff = shifted->EffectiveAffine(42.0, 0.5);
    for (int k = 0; k < 6; ++k) {
        const double want = (k == 0)   ? a.p[0] + 4.0
                            : (k == 3) ? a.p[3] - 3.0
                                       : a.p[static_cast<std::size_t>(k)];
        EXPECT_DOUBLE_EQ(eff.p[static_cast<std::size_t>(k)], want);
    }
    double c = 0.0;
    double r = 0.0;
    a.Apply(10.0, 20.0, c, r);
    double gc = 0.0;
    double gr = 0.0;
    shifted->Apply(10.0, 20.0, gc, gr);
    EXPECT_DOUBLE_EQ(gc, c + 4.0);
    EXPECT_DOUBLE_EQ(gr, r - 3.0);

    // Default construction: identity affine, no bands, no shift.
    const RpcAffineGridded plain;
    double dx = 99.0;
    double dy = 99.0;
    plain.ShiftAt(123.0, 0.5, dx, dy);
    EXPECT_EQ(dx, 0.0);
    EXPECT_EQ(dy, 0.0);
    plain.Apply(10.0, 20.0, gc, gr);
    EXPECT_DOUBLE_EQ(gc, 10.0);
    EXPECT_DOUBLE_EQ(gr, 20.0);

    // MakeUniform: the fresh-solve starting point.
    const auto uniform = RpcAffineGridded::MakeUniform(
        RpcAffine::Identity(), kColLo, kColHi, 12, RpcAffineGridBasis::Linear);
    ASSERT_TRUE(uniform.has_value());
    EXPECT_EQ(uniform->num_cells(), 12);
    EXPECT_NEAR(uniform->cell(0).col_lo, kColLo, 1e-9);
    EXPECT_NEAR(uniform->cell(11).col_hi, kColHi, 1e-9);
    uniform->ShiftAt(50000.0, 0.5, dx, dy);
    EXPECT_EQ(dx, 0.0);
    EXPECT_EQ(dy, 0.0);
    EXPECT_FALSE(
        RpcAffineGridded::MakeUniform(
            RpcAffine::Identity(), 10.0, 0.0, 4, RpcAffineGridBasis::Linear)
            .has_value());

    // The storage cap: kMaxCells accepted, one more rejected; a reversed
    // col range is rejected.
    const std::vector<Cell> max_bands(
        static_cast<std::size_t>(RpcAffineGridded::kMaxCells),
        Cell{0.0, 1.0, 0.0, 1.0, 0.0, 0.0});
    EXPECT_TRUE(RpcAffineGridded::Make(RpcAffine::Identity(),
                                       max_bands,
                                       RpcAffineGridBasis::Linear)
                    .has_value());
    const std::vector<Cell> too_many(
        static_cast<std::size_t>(RpcAffineGridded::kMaxCells) + 1);
    EXPECT_FALSE(RpcAffineGridded::Make(RpcAffine::Identity(),
                                        too_many,
                                        RpcAffineGridBasis::Linear)
                     .has_value());
    EXPECT_FALSE(RpcAffineGridded::Make(
                     RpcAffine::Identity(),
                     std::vector<Cell>{Cell{10.0, 10.0, 0.0, 1.0, 0.0, 0.0}},
                     RpcAffineGridBasis::Linear)
                     .has_value());
}

// End-to-end consistency between the solver's internal evaluation and the
// objects it returns: corrupt the pixels through an RpcAffineGridded built
// from KNOWN truth (zero-mean band shifts, so the truth sits exactly
// inside the solver's constrained family), solve, and require the returned
// composites to agree -- pins the consumer-side blending to the solver's,
// bit for bit.
TEST(SolveRpcBundleAdjustGridded, ConsumerReproducesSolve) {
    constexpr double kTwoPi = 6.2831853071795865;
    constexpr int kBandsPerScene = 8;
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);

    // Zero-mean shift patterns over the uniform band centers (full-period
    // harmonics), distinct per scene.
    std::vector<RpcAffineGridded> truth_gridded;
    for (int s = 0; s < 3; ++s) {
        const double span = kColHi - kColLo;
        std::vector<RpcAffineGridded::Cell> bs;
        for (int i = 0; i < kBandsPerScene; ++i) {
            bs.push_back(RpcAffineGridded::Cell{
                kColLo + span * static_cast<double>(i) /
                             static_cast<double>(kBandsPerScene),
                kColLo + span * static_cast<double>(i + 1) /
                             static_cast<double>(kBandsPerScene),
                0.0,
                1.0,
                3.0 * std::sin((kTwoPi * i / kBandsPerScene) + s),
                2.0 * std::cos((kTwoPi * i / kBandsPerScene) - s)});
        }
        truth_gridded.push_back(
            *RpcAffineGridded::Make(truth[static_cast<std::size_t>(s)],
                                    std::move(bs),
                                    RpcAffineGridBasis::Linear));
    }

    // Corrupt the pixels through the truth composites (noiseless). GCPs
    // at col strides anchor the per-col-region common modes -- without
    // them the composites drift by a smooth common pattern that the free
    // ground blocks absorb (the documented observability caveat), and the
    // composite-vs-truth comparison below would be meaningless.
    const std::vector<Pt> pts = MakePoints(120, views[0]);
    std::vector<RpcBaPoint> points;
    points.reserve(pts.size() + 8);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        RpcBaPoint pt;
        for (int v = 0; v < 3; ++v) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              pts[i].lon,
                              pts[i].lat,
                              pts[i].alt,
                              c,
                              r);
            truth_gridded[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            pt.measures.push_back(RpcBaMeasure{v, c, r});
        }
        points.push_back(pt);
    }
    for (std::size_t idx : ColStridedIndices(views, pts, 8)) {
        RpcBaPoint gcp = points[idx];
        gcp.ground_fixed = true;
        gcp.lon = pts[idx].lon;
        gcp.lat = pts[idx].lat;
        gcp.height = pts[idx].alt;
        points.push_back(gcp);
    }

    std::vector<RpcAffineGridded> net =
        FreshGriddedNet(3, kBandsPerScene, RpcAffineGridBasis::Linear);
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, points, net);

    ASSERT_TRUE(report.ok) << report.message;
    // The truth lies inside the model family: the fit is exact (the small
    // residue is the col-argument offset between corruption and solve).
    EXPECT_LT(report.rms_after_px, 0.05);

    for (int s = 0; s < 3; ++s) {
        const RpcAffineGridded& got = net[static_cast<std::size_t>(s)];
        EXPECT_EQ(got.num_cells(), kBandsPerScene) << "scene " << s;
        // Compare on the band-center span, where the composite is uniquely
        // determined: beyond the outermost centers an affine tilt trades
        // against per-point ground gradients, so the constant-extension
        // zones are only data-anchored (a documented property of the tent
        // basis, not a solver defect).
        const double col_lo = 0.5 * got.cell(0).col_hi;
        const double col_hi = 0.5 * (got.cell(kBandsPerScene - 1).col_lo +
                                     got.cell(kBandsPerScene - 1).col_hi);
        for (double col = col_lo + 1.0; col < col_hi; col += 733.0) {
            for (double row = 3.0; row < 100000.0; row += 6151.0) {
                double want_c = 0.0;
                double want_r = 0.0;
                double got_c = 0.0;
                double got_r = 0.0;
                truth_gridded[static_cast<std::size_t>(s)].Apply(
                    col, row, want_c, want_r);
                got.Apply(col, row, got_c, got_r);
                EXPECT_NEAR(got_c, want_c, 0.05)
                    << "scene " << s << " col " << col;
                EXPECT_NEAR(got_r, want_r, 0.05)
                    << "scene " << s << " col " << col;
            }
        }
    }
}

// ==================== 2D grid: blending semantics ==========================

TEST(RpcAffineGridded, Bilinear2DBlending) {
    using Cell = RpcAffineGridded::Cell;
    // A 2x2 tensor grid over cols [0,20) x rows [0,10), passed SHUFFLED.
    const std::vector<Cell> shuffled{
        Cell{10.0, 20.0, 5.0, 10.0, 4.0, -4.0},
        Cell{0.0, 10.0, 0.0, 5.0, 1.0, -1.0},
        Cell{10.0, 20.0, 0.0, 5.0, 3.0, 1.0},
        Cell{0.0, 10.0, 5.0, 10.0, -2.0, 2.0},
    };
    const auto linear = RpcAffineGridded::Make(
        RpcAffine::Identity(), shuffled, RpcAffineGridBasis::Linear);
    ASSERT_TRUE(linear.has_value());
    EXPECT_EQ(linear->num_cells(), 4);
    EXPECT_EQ(linear->num_cols(), 2);
    // Canonical row-major order (sorted by row center, then col center).
    EXPECT_DOUBLE_EQ(linear->cell(0).dx, 1.0);
    EXPECT_DOUBLE_EQ(linear->cell(1).dx, 3.0);
    EXPECT_DOUBLE_EQ(linear->cell(2).dx, -2.0);
    EXPECT_DOUBLE_EQ(linear->cell(3).dx, 4.0);

    const auto Shift = [&](double col, double row, double dx, double dy) {
        double gx = 99.0;
        double gy = 99.0;
        linear->ShiftAt(col, row, gx, gy);
        EXPECT_NEAR(gx, dx, 1e-12) << col << "," << row;
        EXPECT_NEAR(gy, dy, 1e-12) << col << "," << row;
    };
    Shift(5.0, 2.5, 1.0, -1.0);  // the cells' centers reproduce exactly
    Shift(15.0, 2.5, 3.0, 1.0);
    Shift(5.0, 7.5, -2.0, 2.0);
    Shift(15.0, 7.5, 4.0, -4.0);
    Shift(10.0, 5.0, 1.5, -0.5);  // the center corner: mean of the four
    Shift(0.0, 0.0, 1.0, -1.0);   // constant extension beyond each span
    Shift(25.0, 20.0, 4.0, -4.0);
    Shift(5.0, 20.0, -2.0, 2.0);  // row extension, col center

    // Constant basis: the containing cell, else nearest center.
    const auto constant = RpcAffineGridded::Make(
        RpcAffine::Identity(), shuffled, RpcAffineGridBasis::Constant);
    ASSERT_TRUE(constant.has_value());
    double gx = 0.0;
    double gy = 0.0;
    constant->ShiftAt(7.0, 3.0, gx, gy);  // inside the top-left cell
    EXPECT_DOUBLE_EQ(gx, 1.0);
    EXPECT_DOUBLE_EQ(gy, -1.0);
    constant->ShiftAt(15.0, 4.0, gx, gy);  // [col_lo, col_hi) containment
    EXPECT_DOUBLE_EQ(gx, 3.0);
    constant->ShiftAt(-5.0, 9.0, gx, gy);  // outside: nearest center (5, 7.5)
    EXPECT_DOUBLE_EQ(gx, -2.0);

    // Linear rejects a non-tensor layout (the second row's col keys
    // disagree with the first's); Constant accepts it.
    const std::vector<Cell> ragged{
        Cell{0.0, 10.0, 0.0, 5.0, 0.0, 0.0},
        Cell{10.0, 20.0, 0.0, 5.0, 0.0, 0.0},
        Cell{0.0, 20.0, 5.0, 10.0, 0.0, 0.0},
    };
    EXPECT_FALSE(RpcAffineGridded::Make(
                     RpcAffine::Identity(), ragged, RpcAffineGridBasis::Linear)
                     .has_value());
    EXPECT_TRUE(RpcAffineGridded::Make(
                    RpcAffine::Identity(), ragged, RpcAffineGridBasis::Constant)
                    .has_value());

    // MakeUniform2D: dims, canonical layout, bounds, and the cell cap.
    const auto grid =
        RpcAffineGridded::MakeUniform2D(RpcAffine::Identity(),
                                        100.0,
                                        140.0,
                                        0.0,
                                        10.0,
                                        4,
                                        2,
                                        RpcAffineGridBasis::Linear);
    ASSERT_TRUE(grid.has_value());
    EXPECT_EQ(grid->num_cells(), 8);
    EXPECT_EQ(grid->num_cols(), 4);
    EXPECT_DOUBLE_EQ(grid->cell(0).col_lo, 100.0);
    EXPECT_DOUBLE_EQ(grid->cell(7).col_hi, 140.0);
    EXPECT_DOUBLE_EQ(grid->cell(7).row_hi, 10.0);
    EXPECT_FALSE(RpcAffineGridded::MakeUniform2D(RpcAffine::Identity(),
                                                 0.0,
                                                 10.0,
                                                 0.0,
                                                 10.0,
                                                 9,
                                                 9,
                                                 RpcAffineGridBasis::Linear)
                     .has_value());  // 81 cells > kMaxCells
}

// ================== two-stage: frozen affine + 2D field ===================

// The synthetic models' row span: rows land in [0, 2*line_off] =
// [0, 50000].
constexpr double kRowLo = 0.0;
constexpr double kRowHi = 50000.0;

// One full-period row wave, evaluated at the RAW RPC row.
double RowWave(double row, double amp) {
    constexpr double kTwoPi = 6.2831853071795865;
    return amp * std::cos(kTwoPi * (row - kRowLo) / (kRowHi - kRowLo));
}

// A genuinely 2D error field: separable col + row waves.
double Field2D(double col, double row, double col_amp, double row_amp) {
    return ColWave(col, col_amp) + RowWave(row, row_amp);
}

// One GCP per grid cell: the tie point whose view-0 pixel is nearest the
// cell center. Every cell region needs an absolute anchor -- the
// per-cell common modes are invisible to ties (the ground blocks absorb
// them), and stride sampling leaves gaps that drift by whole pixels.
std::vector<std::size_t> CellNearestIndices(const std::vector<RpcInfo>& views,
                                            const std::vector<Pt>& pts,
                                            const RpcAffineGridded& grid) {
    std::vector<std::size_t> out;
    std::vector<char> taken(pts.size(), 0);
    for (int i = 0; i < grid.num_cells(); ++i) {
        const double cc = 0.5 * (grid.cell(i).col_lo + grid.cell(i).col_hi);
        const double rc = 0.5 * (grid.cell(i).row_lo + grid.cell(i).row_hi);
        double best = std::numeric_limits<double>::infinity();
        std::size_t pick = pts.size();
        for (std::size_t k = 0; k < pts.size(); ++k) {
            if (taken[k]) {
                continue;
            }
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(
                views[0], pts[k].lon, pts[k].lat, pts[k].alt, c, r);
            const double d = ((c - cc) * (c - cc)) + ((r - rc) * (r - rc));
            if (d < best) {
                best = d;
                pick = k;
            }
        }
        if (pick != pts.size()) {
            taken[pick] = 1;
            out.push_back(pick);
        }
    }
    return out;
}

// The full two-stage pipeline over a genuinely 2D error field: stage 1
// lands the global affines (pulled near identity by the pixel prior --
// the satellite regime), stage 2 freezes them and lets the 2D grid
// absorb the local structure. The COMPOSITE must recover truth affine +
// field (the prior biases the affine decomposition towards identity by
// design; the grid spans what the affine left).
TEST(SolveRpcBundleAdjustTwoStage, Recovers2DErrorField) {
    constexpr double kColAmp = 3.0;
    constexpr double kRowAmp = 2.0;
    constexpr int kNumCol = 8;
    constexpr int kNumRow = 6;
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);

    const std::vector<Pt> pts = MakePoints(200, views[0]);
    std::vector<RpcBaPoint> points;
    points.reserve(pts.size() + 12);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        RpcBaPoint pt;
        for (int v = 0; v < 3; ++v) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              pts[i].lon,
                              pts[i].lat,
                              pts[i].alt,
                              c,
                              r);
            const double wc = Field2D(c, r, kColAmp, kRowAmp);
            const double wr = Field2D(c, r, 0.6 * kColAmp, 0.5 * kRowAmp);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            pt.measures.push_back(RpcBaMeasure{v, c + wc, r + wr});
        }
        points.push_back(pt);
    }
    std::vector<RpcAffineGridded> net;
    for (int s = 0; s < 3; ++s) {
        net.push_back(
            *RpcAffineGridded::MakeUniform2D(RpcAffine::Identity(),
                                             kColLo,
                                             kColHi,
                                             kRowLo,
                                             kRowHi,
                                             kNumCol,
                                             kNumRow,
                                             RpcAffineGridBasis::Linear));
    }
    for (std::size_t idx : CellNearestIndices(views, pts, net[0])) {
        RpcBaPoint gcp = points[idx];
        gcp.ground_fixed = true;
        gcp.lon = pts[idx].lon;
        gcp.lat = pts[idx].lat;
        gcp.height = pts[idx].alt;
        points.push_back(gcp);
    }
    RpcBaGridOptions options;
    options.identity_prior_px = 2.0;  // the satellite regime: near-identity
    const RpcBaTwoStageReport report =
        zproj::crs::solve_rpc_bundle_adjust_two_stage(
            views, points, net, options);

    ASSERT_TRUE(report.affine_stage.ok) << report.affine_stage.message;
    ASSERT_TRUE(report.grid_stage.ok) << report.grid_stage.message;
    EXPECT_LT(report.grid_stage.rms_after_px, 0.5);

    // Stage 2 froze the affines at stage 1's solution: a manual stage-1
    // run over the same network must agree bit for bit.
    {
        std::vector<RpcAffine> stage1(3, RpcAffine::Identity());
        const RpcBaReport stage1_report =
            solve_rpc_bundle_adjust(views, points, stage1, options);
        ASSERT_TRUE(stage1_report.ok) << stage1_report.message;
        for (int s = 0; s < 3; ++s) {
            for (int k = 0; k < 6; ++k) {
                EXPECT_NEAR(net[static_cast<std::size_t>(s)]
                                .affine()
                                .p[static_cast<std::size_t>(k)],
                            stage1[static_cast<std::size_t>(s)]
                                .p[static_cast<std::size_t>(k)],
                            1e-9)
                    << "scene " << s << " param " << k;
            }
        }
    }

    for (int s = 0; s < 3; ++s) {
        const RpcAffineGridded& obj = net[static_cast<std::size_t>(s)];
        for (int i = 0; i < obj.num_cells(); ++i) {
            const double cc = 0.5 * (obj.cell(i).col_lo + obj.cell(i).col_hi);
            const double rc = 0.5 * (obj.cell(i).row_lo + obj.cell(i).row_hi);
            const RpcAffine eff = obj.EffectiveAffine(cc, rc);
            RpcAffine want = truth[static_cast<std::size_t>(s)];
            want.p[0] += Field2D(cc, rc, kColAmp, kRowAmp);
            want.p[3] += Field2D(cc, rc, 0.6 * kColAmp, 0.5 * kRowAmp);
            double ec = 0.0;
            double er = 0.0;
            double wc = 0.0;
            double wr = 0.0;
            eff.Apply(cc, rc, ec, er);
            want.Apply(cc, rc, wc, wr);
            EXPECT_NEAR(ec, wc, 0.8) << "scene " << s << " cell " << i;
            EXPECT_NEAR(er, wr, 0.8) << "scene " << s << " cell " << i;
        }
    }
}

// fix_scene_affines: the affines are CONSTANTS of the solve (returned bit
// for bit) and the cells absorb the wave around them.
TEST(SolveRpcBundleAdjustGridded, FixSceneAffinesFreezesDecomposition) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    constexpr double kColAmp = 4.0;
    constexpr double kRowAmp = 3.0;
    Network data =
        MakeWaveNetwork(views, truth, kColAmp, kRowAmp, 120, 2, 0.0, 7);
    std::vector<Pt> gcp_pts;
    for (std::size_t idx : ColStridedIndices(views, data.pts, 8)) {
        gcp_pts.push_back(data.pts[idx]);
    }
    const std::vector<RpcBaPoint> gcps =
        MakeWaveGcps(views, truth, kColAmp, kRowAmp, gcp_pts, 0.0, 9);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());

    std::vector<RpcAffineGridded> net;
    for (int s = 0; s < 2; ++s) {
        net.push_back(
            *RpcAffineGridded::MakeUniform(truth[static_cast<std::size_t>(s)],
                                           kColLo,
                                           kColHi,
                                           10,
                                           RpcAffineGridBasis::Linear));
    }
    RpcBaGridOptions options;
    options.fix_scene_affines = true;
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, data.points, net, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_LT(report.rms_after_px, 0.3);
    for (int s = 0; s < 2; ++s) {
        for (int k = 0; k < 6; ++k) {
            EXPECT_DOUBLE_EQ(net[static_cast<std::size_t>(s)]
                                 .affine()
                                 .p[static_cast<std::size_t>(k)],
                             truth[static_cast<std::size_t>(s)]
                                 .p[static_cast<std::size_t>(k)])
                << "scene " << s << " param " << k;
        }
    }
    for (int i = 0; i < net[0].num_cells(); ++i) {
        const double cc = 0.5 * (net[0].cell(i).col_lo + net[0].cell(i).col_hi);
        const RpcAffine eff = net[0].EffectiveAffine(cc, 25000.0);
        RpcAffine want = truth[0];
        want.p[0] += ColWave(cc, kColAmp);
        want.p[3] += ColWave(cc, kRowAmp);
        double ec = 0.0;
        double er = 0.0;
        double wc = 0.0;
        double wr = 0.0;
        eff.Apply(cc, 25000.0, ec, er);
        want.Apply(cc, 25000.0, wc, wr);
        EXPECT_NEAR(ec, wc, 0.6) << "cell " << i;
        EXPECT_NEAR(er, wr, 0.6) << "cell " << i;
    }
}

// ============== identity_prior_px: pinning the drifted solve ================

// The pixel-unit identity prior tames a drifted solve: a deliberately
// far warm start (30 px) over a matches-only network with free heights
// (the classical drift regime) must land back near identity under the
// prior, and no further from it than the unconstrained run.
TEST(SolveRpcBundleAdjust, IdentityPriorPxPinsDriftedWarmStart) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const Network data = MakeNetwork(views, IdentityAffines(2), 40, 2, 0.3, 11);

    const RpcAffine drifted = [] {
        RpcAffine a;
        a.p[0] = 30.0;  // a large deliberate drift
        a.p[3] = -25.0;
        return a;
    }();
    // A local pixel-unit delta (the library's AffineDeltaPx is private to
    // zproj_refine): translations directly, linear terms scaled to the
    // domain edge.
    const auto DeltaPx = [&](const RpcAffine& a) {
        const double scale[6] = {1.0,
                                 views[0].samp_scale,
                                 views[0].line_scale,
                                 1.0,
                                 views[0].samp_scale,
                                 views[0].line_scale};
        double worst = 0.0;
        for (int k = 0; k < 6; ++k) {
            worst = std::max(
                worst,
                std::fabs(
                    a.p[static_cast<std::size_t>(k)] -
                    RpcAffine::Identity().p[static_cast<std::size_t>(k)]) *
                    scale[static_cast<std::size_t>(k)]);
        }
        return worst;
    };

    RpcBaOptions prior_options;
    prior_options.zero_mean_affines = true;
    prior_options.identity_prior_px = 2.0;
    std::vector<RpcAffine> pinned(2, drifted);
    const RpcBaReport pinned_report =
        solve_rpc_bundle_adjust(views, data.points, pinned, prior_options);
    ASSERT_TRUE(pinned_report.ok) << pinned_report.message;

    RpcBaOptions free_options;
    free_options.zero_mean_affines = true;
    std::vector<RpcAffine> free_net(2, drifted);
    const RpcBaReport free_report =
        solve_rpc_bundle_adjust(views, data.points, free_net, free_options);
    ASSERT_TRUE(free_report.ok) << free_report.message;

    for (int v = 0; v < 2; ++v) {
        EXPECT_LT(DeltaPx(pinned[static_cast<std::size_t>(v)]), 6.0)
            << "view " << v;  // ~3 prior sigmas
        EXPECT_LE(DeltaPx(pinned[static_cast<std::size_t>(v)]),
                  DeltaPx(free_net[static_cast<std::size_t>(v)]))
            << "view " << v;
    }
}

// Noisy matches with a fraction of gross outliers: the redescending
// losses (Cauchy, Tukey) must hold the recovery at least as well as the
// Huber reference above -- same network, same budgets.
TEST(SolveRpcBundleAdjust, CauchyTukeyLossesRobustToOutliers) {
    const std::vector<RpcInfo> views = MakeBaViews(5);
    const std::vector<RpcAffine> truth = MakeTruthAffines(5);
    const double sigma = 0.2;
    const Network data = MakeNetwork(views, truth, 200, 4, sigma, 17, 8);

    const zproj::crs::RpcLossKind kinds[2] = {zproj::crs::RpcLossKind::Cauchy,
                                              zproj::crs::RpcLossKind::Tukey};
    for (const zproj::crs::RpcLossKind kind : kinds) {
        RpcBaOptions options;
        options.pixel_sigma = sigma;
        options.robust_threshold_px = 3.0 * sigma;
        options.loss_kind = kind;
        options.affine_prior_weight = 1.0;
        std::vector<RpcAffine> out = IdentityAffines(5);
        const RpcBaReport report =
            solve_rpc_bundle_adjust(views, data.points, out, options);

        ASSERT_TRUE(report.ok) << report.message;
        // The report RMS ignores the loss, so the 25 gross outliers keep
        // it high; the redescending shapes also fit the clean bulk a bit
        // looser than Huber. Parameter recovery is the assertion below.
        EXPECT_LT(report.rms_after_px, 8.0);
        for (int v = 0; v < 5; ++v) {
            EXPECT_LT(std::fabs(out[static_cast<std::size_t>(v)].p[0] -
                                truth[static_cast<std::size_t>(v)].p[0]),
                      50.0)
                << "view " << v << " translation e0";
        }
    }
}

// ===================== robustness: residuals & grid knobs ===================

using zproj::crs::RpcBaMeasureResidual;
using zproj::crs::RpcBaRobustOptions;
using zproj::crs::RpcBaRobustPass;
using zproj::crs::RpcBaRobustReport;
using zproj::crs::solve_rpc_bundle_adjust_robust;

// The residual out-record: one entry per used measure, indices that
// address the caller's network, and a component RMS that matches the
// report's rms_after_px exactly.
TEST(SolveRpcBundleAdjustGridded, MeasureResidualsRecorded) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    Network data = MakeWaveNetwork(views, truth, 4.0, 3.0, 90, 3, 0.2, 61);
    std::vector<Pt> gcp_pts;
    for (std::size_t idx : ColStridedIndices(views, data.pts, 6)) {
        gcp_pts.push_back(data.pts[idx]);
    }
    const std::vector<RpcBaPoint> gcps =
        MakeWaveGcps(views, truth, 4.0, 3.0, gcp_pts, 0.0, 9);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());

    std::vector<RpcAffineGridded> net =
        FreshGriddedNet(3, 10, RpcAffineGridBasis::Linear);
    std::vector<RpcBaMeasureResidual> residuals;
    const RpcBaReport report = solve_rpc_bundle_adjust_gridded(
        views, data.points, net, {}, &residuals);

    ASSERT_TRUE(report.ok) << report.message;
    ASSERT_EQ(residuals.size(), static_cast<std::size_t>(report.num_measures));
    double sum = 0.0;
    for (const RpcBaMeasureResidual& r : residuals) {
        // The indices address the caller's network.
        ASSERT_LT(r.point, static_cast<int>(data.points.size()));
        const RpcBaPoint& pt = data.points[static_cast<std::size_t>(r.point)];
        ASSERT_LT(r.measure, static_cast<int>(pt.measures.size()));
        EXPECT_EQ(r.scene,
                  pt.measures[static_cast<std::size_t>(r.measure)].view);
        // Every scene carries cells, so every measure has a containing
        // cell.
        EXPECT_GE(r.cell, 0);
        sum += (r.res_col * r.res_col) + (r.res_row * r.res_row);
    }
    const double rms =
        std::sqrt(sum / (2.0 * static_cast<double>(residuals.size())));
    EXPECT_NEAR(rms, report.rms_after_px, 1e-9);
}

// The support floor: cells left below min_measures_per_cell (here: cells
// outside the covered columns, of a warm start that seeded them with a
// nonzero shift) fall back to the scene affine -- their shift returns
// exactly zero -- while covered cells keep fitting. Stage-2 semantics
// (affines held at the truth): under a FREE affine the data-less region's
// affine tilt vs shift-ramp gauge wanders freely (the documented
// constant-extension property), which is orthogonal to the floor.
TEST(SolveRpcBundleAdjustGridded, MinMeasuresPerCellPinsStarvedCells) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    Network data = MakeWaveNetwork(views, truth, 4.0, 3.0, 120, 2, 0.0, 71);
    std::vector<Pt> gcp_pts;
    for (std::size_t idx : ColStridedIndices(views, data.pts, 8)) {
        gcp_pts.push_back(data.pts[idx]);
    }
    const std::vector<RpcBaPoint> gcps =
        MakeWaveGcps(views, truth, 4.0, 3.0, gcp_pts, 0.0, 9);
    data.points.insert(data.points.end(), gcps.begin(), gcps.end());

    // Keep only measures in the LEFT half of the col span: the right-half
    // cells starve.
    constexpr double kSplit = 0.5 * (kColLo + kColHi);
    std::vector<RpcBaPoint> half;
    half.reserve(data.points.size());
    for (RpcBaPoint& pt : data.points) {
        RpcBaPoint keep = pt;
        keep.measures.clear();
        for (const RpcBaMeasure& m : pt.measures) {
            if (m.col < kSplit) {
                keep.measures.push_back(m);
            }
        }
        if (keep.measures.size() >= 2) {
            half.push_back(std::move(keep));
        }
    }
    ASSERT_GT(half.size(), std::size_t{20});

    // Warm start every cell with a nonzero shift: without the floor the
    // starved cells would carry the stale seed (nothing observes them).
    std::vector<RpcAffineGridded> net;
    for (int s = 0; s < 2; ++s) {
        net.push_back(
            *RpcAffineGridded::MakeUniform(truth[static_cast<std::size_t>(s)],
                                           kColLo,
                                           kColHi,
                                           10,
                                           RpcAffineGridBasis::Linear));
        for (int i = 0; i < net[static_cast<std::size_t>(s)].num_cells(); ++i) {
            net[static_cast<std::size_t>(s)].set_cell_shift(i, 5.0, -4.0);
        }
    }
    RpcBaGridOptions options;
    options.fix_scene_affines = true;
    options.min_measures_per_cell = 4;
    const RpcBaReport report =
        solve_rpc_bundle_adjust_gridded(views, half, net, options);

    ASSERT_TRUE(report.ok) << report.message;
    for (int s = 0; s < 2; ++s) {
        const RpcAffineGridded& obj = net[static_cast<std::size_t>(s)];
        for (int i = 0; i < obj.num_cells(); ++i) {
            if (obj.cell(i).col_lo >= kSplit && i + 1 < obj.num_cells()) {
                // Pinned free cells: exactly the fallback (zero).
                EXPECT_EQ(obj.cell(i).dx, 0.0)
                    << "scene " << s << " cell " << i;
                EXPECT_EQ(obj.cell(i).dy, 0.0)
                    << "scene " << s << " cell " << i;
            }
        }
        // The canonically last (derived) cell is not a parameter block
        // and never pins: its value is minus the sum of the free cells',
        // which for this scene's zero-mean-fitting wave is ~0 (tent
        // approximation residue), NOT a stale seed.
        const int last = obj.num_cells() - 1;
        EXPECT_NEAR(obj.cell(last).dx, 0.0, 1.0);
        EXPECT_NEAR(obj.cell(last).dy, 0.0, 1.0);
        // The covered (left) cells still absorb the wave.
        int fitted = 0;
        for (int i = 0; i < obj.num_cells(); ++i) {
            if (obj.cell(i).col_hi <= kSplit - 5000.0) {
                ++fitted;
            }
        }
        EXPECT_GE(fitted, 3);
    }
    EXPECT_LT(report.rms_after_px, 1.0);
}

// The median warm start selects the basin for a tight redescending
// loss. A GCP-only network (ground fixed, no triangulation) under the
// one-hot Constant basis isolates the mechanism exactly: scene 0's cells
// carry known +-8 px shifts, and from the ZERO start a 2 px Tukey sees
// every genuine shift as beyond-influence and leaves every cell stuck,
// while from the per-cell median start the residuals are already zero
// and the shifts survive -- including the cell where 40% of the measures
// are +25 px outliers (the median itself is the 50%-breakdown estimate).
// (The Linear tent basis re-activates across the whole wave through its
// shared bracket measures, and a free-ground network lets the common
// mode drain into the ground blocks -- both defeat the isolation.)
TEST(SolveRpcBundleAdjustGridded, MedianCellInitAnchorsBasin) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    constexpr int kBands = 10;
    constexpr int kCorruptBand = 4;
    // Alternating +-8 px, with the (canonically last) band derived to
    // keep the sum zero -- the solver's constrained family.
    std::vector<double> band_shift(kBands, 0.0);
    double sum = 0.0;
    for (int i = 0; i + 1 < kBands; ++i) {
        band_shift[static_cast<std::size_t>(i)] = ((i % 2 == 0) ? 8.0 : -8.0);
        sum += band_shift[static_cast<std::size_t>(i)];
    }
    band_shift[static_cast<std::size_t>(kBands - 1)] = -sum;
    for (int i = 0; i < kBands; ++i) {
        EXPECT_LE(std::fabs(band_shift[static_cast<std::size_t>(i)]),
                  8.0 + 1e-9);
    }

    // Every point is a GCP observed in both scenes; scene 0's pixels
    // carry the band shift, and 40% of the corrupted band's scene-0
    // measures take +25 px.
    const std::vector<Pt> pts = MakePoints(200, views[0]);
    std::mt19937 rng(91);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    const auto BandOf = [](double col) {
        return static_cast<int>((col - kColLo) / (kColHi - kColLo) * kBands);
    };
    std::vector<RpcBaPoint> points;
    points.reserve(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        RpcBaPoint pt;
        pt.ground_fixed = true;
        pt.lon = pts[i].lon;
        pt.lat = pts[i].lat;
        pt.height = pts[i].alt;
        for (int v = 0; v < 2; ++v) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              pt.lon,
                              pt.lat,
                              pt.height,
                              c,
                              r);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            if (v == 0) {
                const int band = BandOf(c);
                c += band_shift[static_cast<std::size_t>(band)];
                if (band == kCorruptBand && coin(rng) < 0.4) {
                    c += 25.0;
                }
            }
            pt.measures.push_back(RpcBaMeasure{v, c, r});
        }
        points.push_back(pt);
    }

    const auto Solve = [&](bool median_init) {
        std::vector<RpcAffineGridded> net(2);
        for (int s = 0; s < 2; ++s) {
            net[static_cast<std::size_t>(s)] = *RpcAffineGridded::MakeUniform(
                truth[static_cast<std::size_t>(s)],
                kColLo,
                kColHi,
                kBands,
                RpcAffineGridBasis::Constant);
        }
        RpcBaGridOptions options;
        options.fix_scene_affines = true;  // isolate the cell stage
        options.robust_threshold_px = 2.0;
        options.loss_kind = zproj::crs::RpcLossKind::Tukey;
        options.median_cell_init = median_init;
        const RpcBaReport rep =
            solve_rpc_bundle_adjust_gridded(views, points, net, options);
        EXPECT_TRUE(rep.ok) << rep.message;
        return net;
    };

    const std::vector<RpcAffineGridded> cold = Solve(false);
    const std::vector<RpcAffineGridded> warmed = Solve(true);

    double cold_worst = 0.0;
    double warm_worst = 0.0;
    for (int i = 0; i < kBands; ++i) {
        cold_worst =
            std::max(cold_worst,
                     std::fabs(cold[0].cell(i).dx -
                               band_shift[static_cast<std::size_t>(i)]));
        warm_worst =
            std::max(warm_worst,
                     std::fabs(warmed[0].cell(i).dx -
                               band_shift[static_cast<std::size_t>(i)]));
    }
    // Cold: every genuine shift sits beyond the Tukey threshold from a
    // zero start, so the cells never move.
    EXPECT_GT(cold_worst, 3.0);
    // Warm: the medians land the cells on their shifts, corrupted band
    // included, and the polish keeps them there.
    EXPECT_LT(warm_worst, 0.5);
}

// The smoothness prior: a thin cell carrying only outlier measures
// spikes its shift without the prior and stays with its neighbors under
// it.
TEST(SolveRpcBundleAdjustGridded, SmoothnessPriorSuppressesSpikes) {
    const std::vector<RpcInfo> views = MakeBaViews(2);
    const std::vector<RpcAffine> truth = MakeTruthAffines(2);
    constexpr double kColAmp = 3.0;
    constexpr double kRowAmp = 2.0;
    constexpr int kBands = 10;
    Network data =
        MakeWaveNetwork(views, truth, kColAmp, kRowAmp, 200, 2, 0.2, 97);

    // Band 4 of scene 0: every measure drops except two, which are pinned
    // to the band CENTER (tent weight exactly 1 on band 4, so their pull
    // cannot leak into the bracketing cells) and carry +15 px col
    // outliers; points that fall below 2 measures go entirely (keeping
    // their original measures would re-introduce clean band-4
    // observations that resist the spike).
    const auto BandOf = [](double col) {
        return static_cast<int>((col - kColLo) / (kColHi - kColLo) * kBands);
    };
    const double band_center = kColLo + 4.5 * (kColHi - kColLo) / kBands;
    std::vector<RpcBaPoint> points;
    points.reserve(data.points.size());
    int kept_in_band = 0;
    for (RpcBaPoint& pt : data.points) {
        RpcBaPoint keep = pt;
        keep.measures.clear();
        for (const RpcBaMeasure& m : pt.measures) {
            if (m.view == 0 && BandOf(m.col) == 4) {
                if (kept_in_band < 2) {
                    keep.measures.push_back(
                        RpcBaMeasure{m.view, band_center + 15.0, m.row});
                    ++kept_in_band;
                }
                continue;  // every other band-4 measure of scene 0 drops
            }
            keep.measures.push_back(m);
        }
        if (keep.measures.size() >= 2) {
            points.push_back(std::move(keep));
        }
    }
    // GCPs strided OUTSIDE band 4 only, so no clean observation anchors
    // the spiked band (the point of the test).
    std::vector<RpcBaPoint> gcps;
    {
        std::vector<Pt> gcp_pts;
        for (std::size_t idx : ColStridedIndices(views, data.pts, 8)) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[0],
                              data.pts[idx].lon,
                              data.pts[idx].lat,
                              data.pts[idx].alt,
                              c,
                              r);
            if (BandOf(c) != 4) {
                gcp_pts.push_back(data.pts[idx]);
            }
        }
        gcps = MakeWaveGcps(views, truth, kColAmp, kRowAmp, gcp_pts, 0.0, 9);
    }
    points.insert(points.end(), gcps.begin(), gcps.end());
    ASSERT_EQ(kept_in_band, 2);

    const auto Solve = [&](double smoothness) {
        std::vector<RpcAffineGridded> net(2);
        for (int s = 0; s < 2; ++s) {
            net[static_cast<std::size_t>(s)] = *RpcAffineGridded::MakeUniform(
                truth[static_cast<std::size_t>(s)],
                kColLo,
                kColHi,
                kBands,
                RpcAffineGridBasis::Linear);
        }
        RpcBaGridOptions options;
        options.fix_scene_affines = true;
        options.pixel_sigma = 0.2;
        options.cell_shift_smoothness_weight = smoothness;
        const RpcBaReport rep =
            solve_rpc_bundle_adjust_gridded(views, points, net, options);
        EXPECT_TRUE(rep.ok) << rep.message;
        return net;
    };

    const std::vector<RpcAffineGridded> plain = Solve(0.0);
    const std::vector<RpcAffineGridded> smooth = Solve(50.0);

    // Band 4's dx relative to its neighbors: a spike without the prior,
    // suppressed under it.
    const auto BandDx = [](const RpcAffineGridded& obj, int band) {
        return obj.cell(band).dx;
    };
    const double spike_plain =
        BandDx(plain[0], 4) - 0.5 * (BandDx(plain[0], 3) + BandDx(plain[0], 5));
    const double spike_smooth =
        BandDx(smooth[0], 4) -
        0.5 * (BandDx(smooth[0], 3) + BandDx(smooth[0], 5));
    EXPECT_GT(spike_plain, 6.0);
    EXPECT_LT(spike_smooth, 3.0);
}

// ===================== robust multi-pass driver =============================

// The full robust ladder over a noisy, outlier-corrupted 2D error field:
// the trimmed network's final solve beats the un-guarded single solve,
// the outliers leave as measures (their points survive on the clean
// measures), and the composite recovers the field.
TEST(SolveRpcBundleAdjustRobust, LadderTrimsOutliersAndRecovers) {
    constexpr double kColAmp = 3.0;
    constexpr double kRowAmp = 2.0;
    constexpr double kSigma = 0.3;
    constexpr int kNumCol = 6;
    constexpr int kNumRow = 4;
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);

    // The two-stage 2D-field network plus Gaussian noise and one gross
    // outlier measure per 9th point (+25 px col, first view).
    const std::vector<Pt> pts = MakePoints(300, views[0]);
    std::mt19937 rng(53);
    std::normal_distribution<double> noise(0.0, kSigma);
    std::vector<RpcBaPoint> points;
    points.reserve(pts.size() + 24);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        RpcBaPoint pt;
        for (int v = 0; v < 3; ++v) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              pts[i].lon,
                              pts[i].lat,
                              pts[i].alt,
                              c,
                              r);
            const double wc = Field2D(c, r, kColAmp, kRowAmp);
            const double wr = Field2D(c, r, 0.6 * kColAmp, 0.5 * kRowAmp);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            c += wc + noise(rng);
            r += wr + noise(rng);
            if (i % 9 == 0 && v == 0) {
                c += 25.0;  // the gross outlier
            }
            pt.measures.push_back(RpcBaMeasure{v, c, r});
        }
        points.push_back(pt);
    }
    std::vector<RpcAffineGridded> net_template;
    for (int s = 0; s < 3; ++s) {
        net_template.push_back(
            *RpcAffineGridded::MakeUniform2D(RpcAffine::Identity(),
                                             kColLo,
                                             kColHi,
                                             kRowLo,
                                             kRowHi,
                                             kNumCol,
                                             kNumRow,
                                             RpcAffineGridBasis::Linear));
    }
    // One clean GCP per cell region anchors the per-cell common modes.
    for (std::size_t idx : CellNearestIndices(views, pts, net_template[0])) {
        RpcBaPoint gcp = points[idx];
        gcp.ground_fixed = true;
        gcp.lon = pts[idx].lon;
        gcp.lat = pts[idx].lat;
        gcp.height = pts[idx].alt;
        points.push_back(gcp);
    }

    // The un-guarded reference: a single two-stage solve, no loss.
    RpcBaGridOptions plain_options;
    plain_options.identity_prior_px = 2.0;
    std::vector<RpcAffineGridded> plain_net = net_template;
    const RpcBaTwoStageReport plain_report =
        zproj::crs::solve_rpc_bundle_adjust_two_stage(
            views, points, plain_net, plain_options);
    ASSERT_TRUE(plain_report.affine_stage.ok)
        << plain_report.affine_stage.message;
    ASSERT_TRUE(plain_report.grid_stage.ok) << plain_report.grid_stage.message;

    // The robust ladder: loose -> tight, MAD trim between passes, median
    // warm start and a support floor riding along. Huber (the default)
    // bounds the outliers' influence while keeping it nonzero, so the
    // points' ground blocks migrate back towards their clean majority --
    // the trim then only needs to remove the outliers themselves (the
    // per-point rule spares the collateral clean measures to heal next
    // pass).
    RpcBaRobustOptions options;
    options.identity_prior_px = 2.0;
    options.pixel_sigma = kSigma;
    options.median_cell_init = true;
    options.min_measures_per_cell = 4;
    options.trim_floor_px = 1.0;
    options.passes = {
        RpcBaRobustPass{16.0, 8.0, 4.0},
        RpcBaRobustPass{8.0, 3.0, 3.0},
        RpcBaRobustPass{4.0, 2.0, 0.0},
    };
    std::vector<RpcAffineGridded> net = net_template;
    const RpcBaRobustReport report =
        solve_rpc_bundle_adjust_robust(views, points, net, options);

    ASSERT_TRUE(report.ok) << report.message;
    ASSERT_EQ(report.passes.size(), std::size_t{3});
    // Every 9th of the 300 tie points carries one outlier measure (34 in
    // total): the trim removes those; the collateral of the ground-block
    // contamination costs a measure from a few more points (spared by
    // the per-point rule to heal next pass), and a handful of points
    // that end at their 2-measure floor with a still-gross measure go
    // entirely -- the pipeline's honest accounting of an unrecoverable
    // minority.
    EXPECT_GE(report.num_measures_trimmed, 30);
    EXPECT_LE(report.num_measures_trimmed, 50);
    EXPECT_LE(report.num_points_dropped, 10);

    const double plain_rms = plain_report.grid_stage.rms_after_px;
    const double robust_rms = report.passes.back().grid_stage.rms_after_px;
    EXPECT_LT(robust_rms, 0.6);
    EXPECT_LT(robust_rms, 0.5 * plain_rms);

    // The composite recovers truth affine + field on the cell centers.
    for (int s = 0; s < 3; ++s) {
        const RpcAffineGridded& obj = net[static_cast<std::size_t>(s)];
        for (int i = 0; i < obj.num_cells(); ++i) {
            const double cc = 0.5 * (obj.cell(i).col_lo + obj.cell(i).col_hi);
            const double rc = 0.5 * (obj.cell(i).row_lo + obj.cell(i).row_hi);
            const RpcAffine eff = obj.EffectiveAffine(cc, rc);
            RpcAffine want = truth[static_cast<std::size_t>(s)];
            want.p[0] += Field2D(cc, rc, kColAmp, kRowAmp);
            want.p[3] += Field2D(cc, rc, 0.6 * kColAmp, 0.5 * kRowAmp);
            double ec = 0.0;
            double er = 0.0;
            double wc = 0.0;
            double wr = 0.0;
            eff.Apply(cc, rc, ec, er);
            want.Apply(cc, rc, wc, wr);
            EXPECT_NEAR(ec, wc, 1.0) << "scene " << s << " cell " << i;
            EXPECT_NEAR(er, wr, 1.0) << "scene " << s << " cell " << i;
        }
    }
}

// The empty ladder: exactly one pass, the base options' loss settings,
// nothing trimmed.
TEST(SolveRpcBundleAdjustRobust, EmptyLadderRunsOnePass) {
    const std::vector<RpcInfo> views = MakeBaViews(3);
    const std::vector<RpcAffine> truth = MakeTruthAffines(3);
    constexpr double kColAmp = 3.0;
    constexpr double kRowAmp = 2.0;
    const std::vector<Pt> pts = MakePoints(120, views[0]);
    std::vector<RpcBaPoint> points;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        RpcBaPoint pt;
        for (int v = 0; v < 3; ++v) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              pts[i].lon,
                              pts[i].lat,
                              pts[i].alt,
                              c,
                              r);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            c += Field2D(c, r, kColAmp, kRowAmp);
            r += Field2D(c, r, 0.6 * kColAmp, 0.5 * kRowAmp);
            pt.measures.push_back(RpcBaMeasure{v, c, r});
        }
        points.push_back(pt);
    }
    std::vector<RpcAffineGridded> net;
    for (int s = 0; s < 3; ++s) {
        net.push_back(
            *RpcAffineGridded::MakeUniform2D(RpcAffine::Identity(),
                                             kColLo,
                                             kColHi,
                                             kRowLo,
                                             kRowHi,
                                             6,
                                             4,
                                             RpcAffineGridBasis::Linear));
    }

    RpcBaRobustOptions options;
    options.identity_prior_px = 2.0;
    const RpcBaRobustReport report =
        solve_rpc_bundle_adjust_robust(views, points, net, options);

    ASSERT_TRUE(report.ok) << report.message;
    EXPECT_EQ(report.passes.size(), std::size_t{1});
    EXPECT_EQ(report.num_measures_trimmed, 0);
    EXPECT_EQ(report.num_points_dropped, 0);
    EXPECT_LT(report.passes.front().grid_stage.rms_after_px, 0.5);
}

}  // namespace

int main(int argc, char** argv) {
    zt::Logger::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
