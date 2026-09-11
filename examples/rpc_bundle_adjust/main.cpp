// Multi-view (N >= 2) RPC bundle adjustment demo (Ceres): the control-
// network generalization of the rpc_affine example, on a synthetic
// six-view satellite network.
//
// A known affine corruption is baked into every view's projections of a set
// of ground points (plus Gaussian pixel noise, a few gross outliers, and
// partial visibility -- each point is seen by only 3..6 of the views, like
// a real multi-view match pipeline). solve_rpc_bundle_adjust() recovers
// the per-view affines from the corrupted control network, using the
// GCP-anchored configuration: a handful of ground control points (observed
// in two views each) pin the network's common mode absolutely, per-point
// DEM heights pin the height gauge, and a Huber robust loss handles the
// outliers. The zero-mean gauge constraint is the no-GCP alternative (it
// fixes the common mode to "zero correction" by convention, recovering the
// DIFFERENTIAL corrections only -- covered by the tests).
//
// The demo then shows what the correction buys: N-view triangulating the
// matches with the RAW RPCs leaves 100 m-scale ground error, while
// triangulating through the corrected affines returns to the noise floor
// for the clean matches (outlier-corrupted ones stay bad; a real pipeline
// rejects them between passes, like ASP's multi-pass outlier removal).
//
// Usage:
//   ./build/default/bin/rpc_bundle_adjust [points]
//   e.g. ./build/default/bin/rpc_bundle_adjust 500

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_affine.hpp"
#include "zproj/crs/rpc_bundle_adjust.hpp"
#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/wgs84.hpp"

namespace {

using zproj::crs::Ecef;
using zproj::crs::from_ecef;
using zproj::crs::Geodetic;
using zproj::crs::kDegToRad;
using zproj::crs::kRadToDeg;
using zproj::crs::rpc_forward_point;
using zproj::crs::RpcAffine;
using zproj::crs::RpcBaMeasure;
using zproj::crs::RpcBaOptions;
using zproj::crs::RpcBaPoint;
using zproj::crs::RpcBaReport;
using zproj::crs::RpcInfo;
using zproj::crs::RpcModel;
using zproj::crs::RpcRay;
using zproj::crs::solve_rpc_bundle_adjust;
using zproj::crs::to_ecef;
using zproj::crs::triangulate_nview;

// The bundle-adjust demo views: the synthetic nadir base plus
// position-dependent height coupling and nonlinearity, varied per view
// (lean azimuth and sign, cross-term signs) -- the coupling that makes the
// per-view affines identifiable from tie points alone (see rpc_affine.hpp).
RpcInfo MakeBaViewInfo(int view) {
    RpcInfo info;
    info.line_off = 25000.0;
    info.samp_off = 50000.0;
    info.lat_off = 30.0;
    info.long_off = 100.0;
    info.height_off = 500.0;
    info.line_scale = 50000.0;
    info.samp_scale = 50000.0;
    info.lat_scale = 10.0;
    info.long_scale = 10.0;
    info.height_scale = 500.0;
    info.line_num_coeff[2] = 1.0;  // row = lat
    info.line_den_coeff[0] = 1.0;
    info.samp_num_coeff[1] = 1.0;  // col = lon
    info.samp_den_coeff[0] = 1.0;
    constexpr double kAzimuths[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    const double lean = 4e-3 * ((view % 2 == 0) ? 1.0 : -1.0);
    info.samp_num_coeff[3] = lean * kAzimuths[view % 4][0];
    info.line_num_coeff[3] = lean * kAzimuths[view % 4][1];
    info.samp_num_coeff[4] = 1e-2;                              // lon * lat
    info.line_num_coeff[4] = -1e-2;                             // lon * lat
    info.samp_num_coeff[5] = ((view % 3) - 1) * 2e-2;           // lon * height
    info.line_num_coeff[6] = (((view + 1) % 3) - 1) * -1.5e-2;  // lat * height
    info.samp_num_coeff[7] = 2e-2 + 1e-3 * view;                // lon^2
    info.line_num_coeff[8] = 2e-2 - 1e-3 * view;                // lat^2
    info.min_lon = 90.0;
    info.min_lat = 20.0;
    info.max_lon = 110.0;
    info.max_lat = 40.0;
    return info;
}

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

struct Pt {
    double lon;
    double lat;
    double alt;
};

std::vector<Pt> MakePoints(std::size_t n, const RpcInfo& info) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> lon(
        info.long_off - 0.5 * info.long_scale,
        info.long_off + 0.5 * info.long_scale);
    std::uniform_real_distribution<double> lat(
        info.lat_off - 0.5 * info.lat_scale,
        info.lat_off + 0.5 * info.lat_scale);
    std::uniform_real_distribution<double> alt(
        info.height_off - 0.5 * info.height_scale,
        info.height_off + 0.5 * info.height_scale);
    std::vector<Pt> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = Pt{lon(rng), lat(rng), alt(rng)};
    }
    return pts;
}

// Per-view affine deviation from the truth's zero-mean projection: the
// recovered differential quality, in the zero-mean convention out_v =
// identity + (truth_v - mean(truth)) + error.
double ParamDelta(const RpcAffine& a, const RpcAffine& b) {
    double worst = 0.0;
    for (int i = 0; i < 6; ++i) {
        worst = std::max(worst, std::fabs(a.p[i] - b.p[i]));
    }
    return worst;
}

// N-view triangulate every TIE point through the given affines (identity
// for the raw RPCs) and collect the ground error [m] against the truth,
// paired with the point index (failed triangulations omitted). Only the
// first num_ties points are visited -- the appended GCPs have no
// independent truth to check against.
std::vector<std::pair<std::size_t, double>> GroundErrors(
    const std::vector<RpcInfo>& views,
    const std::vector<RpcModel>& models,
    const std::vector<RpcAffine>& affines,
    const std::vector<Pt>& pts,
    const std::vector<RpcBaPoint>& points) {
    std::vector<std::pair<std::size_t, double>> errs;
    errs.reserve(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        std::vector<RpcRay> rays;
        rays.reserve(points[i].measures.size());
        for (const RpcBaMeasure& m : points[i].measures) {
            const RpcInfo& v = views[static_cast<std::size_t>(m.view)];
            const double span = std::min(0.9 * v.height_scale, 50.0);
            RpcRay ray;
            if (zproj::crs::rpc_ray_affine(
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
        Ecef p;
        double rms = 0.0;
        if (rays.size() < 2 ||
            !triangulate_nview(
                rays.data(), static_cast<int>(rays.size()), p, rms)) {
            continue;  // failed triangulation, skipped in the statistics
        }
        const Ecef truth = to_ecef(Geodetic{
            pts[i].lon * kDegToRad, pts[i].lat * kDegToRad, pts[i].alt});
        const double dx = p.x() - truth.x();
        const double dy = p.y() - truth.y();
        const double dz = p.z() - truth.z();
        errs.emplace_back(i, std::sqrt((dx * dx) + (dy * dy) + (dz * dz)));
    }
    return errs;
}

void PrintErrorSummary(const std::vector<std::pair<std::size_t, double>>& errs,
                       int outlier_every) {
    // The robust loss keeps the outlier MEASUREMENTS from biasing the
    // solve, but triangulating a corrupted match still yields a bad point
    // (a real pipeline rejects them between passes, like ASP's multi-pass
    // outlier removal) -- so the clean subset is the honest quality number.
    std::vector<double> all;
    std::vector<double> clean;
    all.reserve(errs.size());
    for (const auto& [i, e] : errs) {
        all.push_back(e);
        if (i % static_cast<std::size_t>(outlier_every) != 0) {
            clean.push_back(e);
        }
    }
    const auto Stats = [](std::vector<double> v, double& mean, double& p95) {
        double sum = 0.0;
        for (double e : v) {
            sum += e;
        }
        mean = sum / static_cast<double>(v.size());
        // 95th percentile: the (size/20)-th LARGEST.
        const std::size_t k = v.size() / 20;
        std::nth_element(
            v.begin(), v.end() - static_cast<std::ptrdiff_t>(k) - 1, v.end());
        p95 = v[v.size() - 1 - k];
    };
    double all_mean = 0.0;
    double all_p95 = 0.0;
    double clean_mean = 0.0;
    double clean_p95 = 0.0;
    Stats(all, all_mean, all_p95);
    Stats(clean, clean_mean, clean_p95);
    std::cout << "  all points:   mean " << std::setw(9) << all_mean
              << " m, p95 " << std::setw(9) << all_p95 << " m\n"
              << "  clean only:   mean " << std::setw(9) << clean_mean
              << " m, p95 " << std::setw(9) << clean_p95 << " m\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t num_points =
        (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 500;
    constexpr int kNumViews = 6;
    constexpr double kSigma = 0.3;     // pixel noise
    constexpr int kOutlierEvery = 10;  // every 10th point, +25 px outlier

    std::vector<RpcInfo> views;
    std::vector<RpcAffine> truth;
    for (int v = 0; v < kNumViews; ++v) {
        views.push_back(MakeBaViewInfo(v));
        truth.push_back(MakeTruthAffine(v));
    }
    std::vector<RpcAffine> raw_affines(static_cast<std::size_t>(kNumViews),
                                       RpcAffine::Identity());

    // Build the corrupted control network: rotating visibility windows of
    // 3..6 views, per-view truth affines, pixel noise, gross outliers. The
    // per-point heights come from a (perfect) DEM; the first 8 ground
    // points double as GCPs (observed in two views each) to anchor the
    // absolute datum.
    const std::vector<Pt> pts = MakePoints(num_points, views[0]);
    constexpr std::size_t kNumGcps = 8;
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, kSigma);
    std::vector<RpcBaPoint> points;
    points.reserve(num_points + kNumGcps);
    int total_measures = 0;
    for (std::size_t i = 0; i < num_points; ++i) {
        const int start =
            static_cast<int>(i % static_cast<std::size_t>(kNumViews));
        const int count =
            3 + static_cast<int>(i % static_cast<std::size_t>(kNumViews - 2));
        const bool outlier = (i % static_cast<std::size_t>(kOutlierEvery)) == 0;
        RpcBaPoint pt;
        pt.height = pts[i].alt;  // DEM height
        for (int k = 0; k < count; ++k) {
            const int v = (start + k) % kNumViews;
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(views[static_cast<std::size_t>(v)],
                              pts[i].lon,
                              pts[i].lat,
                              pts[i].alt,
                              c,
                              r);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            c += noise(rng);
            r += noise(rng);
            if (outlier && k == 0) {
                c += 25.0;
            }
            pt.measures.push_back(RpcBaMeasure{v, c, r});
            ++total_measures;
        }
        points.push_back(pt);
    }
    for (std::size_t g = 0; g < kNumGcps; ++g) {
        RpcBaPoint pt;
        pt.ground_fixed = true;
        pt.lon = pts[g].lon;
        pt.lat = pts[g].lat;
        pt.height = pts[g].alt;
        for (int k = 0; k < 2; ++k) {
            const int v = static_cast<int>((g + static_cast<std::size_t>(k)) %
                                           static_cast<std::size_t>(kNumViews));
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
            ++total_measures;
        }
        points.push_back(pt);
    }

    std::cout << "multi-view RPC bundle adjustment: " << kNumViews << " views, "
              << num_points << " tie points + " << kNumGcps << " GCPs, "
              << total_measures << " measures, sigma " << kSigma
              << " px, outlier every " << kOutlierEvery << " points\n\n";

    // The GCP-anchored configuration: GCPs pin the common mode, DEM heights
    // pin the height gauge, Huber handles the outliers. (No zero-mean
    // constraint: the GCPs anchor the common mode absolutely, which is
    // what the ground-accuracy comparison below needs.)
    RpcBaOptions options;
    options.pixel_sigma = kSigma;
    options.robust_threshold_px = 3.0 * kSigma;

    std::vector<RpcAffine> corrected(static_cast<std::size_t>(kNumViews),
                                     RpcAffine::Identity());
    const RpcBaReport report =
        solve_rpc_bundle_adjust(views, points, corrected, options);

    std::cout << std::fixed << std::setprecision(4);
    if (!report.ok) {
        std::cout << "solve FAILED: " << report.message << "\n";
        return 1;
    }
    std::cout << "solve: " << report.num_points << " points / "
              << report.num_measures << " measures used, " << report.num_gcps
              << " GCPs, " << report.num_points_skipped << " skipped\n"
              << "reprojection RMS: " << report.rms_before_px << " px -> "
              << report.rms_after_px << " px\n\n";

    // Absolute affine recovery against the corruption truth.
    std::cout << "view   affine dev from truth [px]\n";
    for (int v = 0; v < kNumViews; ++v) {
        std::cout << std::setw(4) << v << std::setw(12) << std::setprecision(3)
                  << ParamDelta(corrected[static_cast<std::size_t>(v)],
                                truth[static_cast<std::size_t>(v)])
                  << "\n";
    }

    // What the correction buys on the ground: N-view triangulation of the
    // matches, raw RPCs vs corrected affines.
    std::vector<RpcModel> models;
    models.reserve(views.size());
    for (const RpcInfo& v : views) {
        models.emplace_back(v);
    }
    std::cout << "\nN-view triangulation ground error vs truth:\n"
              << "raw RPCs:\n";
    PrintErrorSummary(GroundErrors(views, models, raw_affines, pts, points),
                      kOutlierEvery);
    std::cout << "corrected:\n";
    PrintErrorSummary(GroundErrors(views, models, corrected, pts, points),
                      kOutlierEvery);
    return 0;
}
