// Banded bundle adjustment demo (Ceres): the cross-track-error camera
// model -- one global affine per scene plus per-band translation shifts
// banded along the pixel COLUMN axis (tent basis) -- against a
// column-dependent systematic error pattern (error large at the scene's
// left, middle and right, small at the quarter columns).
//
// A known per-scene affine AND a single-period column wave are baked into
// three synthetic views' projections of a set of ground points (plus
// Gaussian pixel noise, outliers, and GCPs anchoring the absolute datum).
// solve_rpc_bundle_adjust_banded() recovers both: the scene affines land
// on the truth, the band shifts track the wave, and the residual-vs-col
// profile flattens from its W shape to the noise floor.
//
// Usage:
//   ./build/default/bin/rpc_bundle_adjust_banded [points]
//   e.g. ./build/default/bin/rpc_bundle_adjust_banded 500

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_affine.hpp"
#include "zproj/crs/rpc_bundle_adjust.hpp"
#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/wgs84.hpp"

namespace {

using zproj::crs::Ecef;
using zproj::crs::Geodetic;
using zproj::crs::kDegToRad;
using zproj::crs::rpc_forward_point;
using zproj::crs::RpcAffine;
using zproj::crs::RpcAffineBandBasis;
using zproj::crs::RpcAffineBanded;
using zproj::crs::RpcBaBandedOptions;
using zproj::crs::RpcBaMeasure;
using zproj::crs::RpcBaPoint;
using zproj::crs::RpcBaReport;
using zproj::crs::RpcInfo;
using zproj::crs::RpcModel;
using zproj::crs::RpcRay;
using zproj::crs::solve_rpc_bundle_adjust_banded;
using zproj::crs::to_ecef;
using zproj::crs::triangulate_nview;

// The synthetic scenes' col span: cols land in [25000, 75000]
// (samp_off -/+ 0.5*samp_scale, the points' lon window).
constexpr double kColLo = 25000.0;
constexpr double kColHi = 75000.0;

// The demo views (same family as the rpc_bundle_adjust example): nadir
// base plus per-view height coupling and nonlinearity.
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
    info.samp_num_coeff[4] = 1e-2;
    info.line_num_coeff[4] = -1e-2;
    info.samp_num_coeff[5] = ((view % 3) - 1) * 2e-2;
    info.line_num_coeff[6] = (((view + 1) % 3) - 1) * -1.5e-2;
    info.samp_num_coeff[7] = 2e-2 + 1e-3 * view;
    info.line_num_coeff[8] = 2e-2 - 1e-3 * view;
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

// One full-period column wave (the observed pattern: max at left/middle/
// right, zero at the quarter columns). Evaluated at the RAW RPC col.
double ColWave(double col, double amp) {
    constexpr double kTwoPi = 6.2831853071795865;
    return amp * std::cos(kTwoPi * (col - kColLo) / (kColHi - kColLo));
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

}  // namespace

int main(int argc, char** argv) {
    const std::size_t num_points =
        (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 500;
    constexpr int kNumScenes = 3;
    constexpr int kBandsPerScene = 12;
    constexpr double kSigma = 0.3;
    constexpr double kColAmp = 5.0;  // wave amplitudes [px]
    constexpr double kRowAmp = 4.0;
    constexpr int kOutlierEvery = 10;
    constexpr std::size_t kNumGcps = 20;

    std::vector<RpcInfo> scenes;
    std::vector<RpcAffine> truth;
    for (int v = 0; v < kNumScenes; ++v) {
        scenes.push_back(MakeBaViewInfo(v));
        truth.push_back(MakeTruthAffine(v));
    }

    // Corrupted control network: per-scene truth affine + the column wave
    // + noise + outliers. GCPs are picked from the tie points spread
    // UNIFORMLY across the columns: every band region needs its absolute
    // anchor (a col region without a GCP carries an unobservable common
    // mode, see rpc_bundle_adjust.hpp).
    const std::vector<Pt> pts = MakePoints(num_points, scenes[0]);
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, kSigma);
    std::vector<RpcBaPoint> points;
    points.reserve(num_points + kNumGcps);
    int total_measures = 0;
    for (std::size_t i = 0; i < num_points; ++i) {
        const bool outlier = (i % static_cast<std::size_t>(kOutlierEvery)) == 0;
        RpcBaPoint pt;
        pt.height = pts[i].alt;
        for (int v = 0; v < kNumScenes; ++v) {
            double c = 0.0;
            double r = 0.0;
            rpc_forward_point(scenes[static_cast<std::size_t>(v)],
                              pts[i].lon,
                              pts[i].lat,
                              pts[i].alt,
                              c,
                              r);
            const double wave_c = ColWave(c, kColAmp);
            const double wave_r = ColWave(c, kRowAmp);
            truth[static_cast<std::size_t>(v)].Apply(c, r, c, r);
            c += wave_c + noise(rng);
            r += wave_r + noise(rng);
            if (outlier && v == 0) {
                c += 25.0;
            }
            pt.measures.push_back(RpcBaMeasure{v, c, r});
            ++total_measures;
        }
        points.push_back(pt);
    }
    {
        std::vector<std::size_t> order(num_points);
        for (std::size_t i = 0; i < num_points; ++i) {
            order[i] = i;
        }
        std::sort(
            order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return points[a].measures.front().col <
                       points[b].measures.front().col;
            });
        for (std::size_t k = 0; k < kNumGcps; ++k) {
            const std::size_t idx = order[(k * num_points) / kNumGcps];
            RpcBaPoint gcp = points[idx];
            gcp.ground_fixed = true;
            gcp.lon = pts[idx].lon;
            gcp.lat = pts[idx].lat;
            gcp.height = pts[idx].alt;
            points.push_back(gcp);
            total_measures += static_cast<int>(gcp.measures.size());
        }
    }

    std::cout << "banded bundle adjustment: " << kNumScenes << " scenes x "
              << kBandsPerScene << " tent bands (col axis), " << num_points
              << " tie points + " << kNumGcps << " GCPs, " << total_measures
              << " measures\nwave: one period per scene, amplitudes " << kColAmp
              << "/" << kRowAmp << " px, sigma " << kSigma
              << " px, outlier every " << kOutlierEvery << " points\n\n";

    // The scene corrections travel as RpcAffineBanded values, in and out:
    // uniform zero-shift tent bands around identity affines for a fresh
    // solve; the solve overwrites them in place.
    std::vector<RpcAffineBanded> corrected;
    corrected.reserve(static_cast<std::size_t>(kNumScenes));
    for (int s = 0; s < kNumScenes; ++s) {
        auto scene = RpcAffineBanded::MakeUniform(RpcAffine::Identity(),
                                                  kColLo,
                                                  kColHi,
                                                  kBandsPerScene,
                                                  RpcAffineBandBasis::Linear);
        if (!scene) {
            std::cout << "scene " << s << ": bad band structure\n";
            return 1;
        }
        corrected.push_back(*scene);
    }

    RpcBaBandedOptions options;
    options.pixel_sigma = kSigma;
    options.robust_threshold_px = 3.0 * kSigma;
    // Col regions without a GCP carry their own common mode (all scenes'
    // bands there drift together, absorbed by the ground blocks); the
    // shift prior pins those regions to "no correction relative to the
    // scene affine" (see rpc_bundle_adjust.hpp's observability note).
    options.band_shift_prior_weight = 0.1;

    const RpcBaReport report =
        solve_rpc_bundle_adjust_banded(scenes, points, corrected, options);

    std::cout << std::fixed << std::setprecision(3);
    if (!report.ok) {
        std::cout << "solve FAILED: " << report.message << "\n";
        return 1;
    }
    std::cout << "solve: " << report.num_points << " points / "
              << report.num_measures << " measures, " << report.num_gcps
              << " GCPs, " << report.num_points_skipped << " skipped\n"
              << "reprojection RMS: " << report.rms_before_px << " px -> "
              << report.rms_after_px << " px\n\n";

    // Residual RMS by col tenth, raw vs corrected: the W shape must
    // flatten. (Clean points only; outlier-corrupted measures stay bad by
    // design -- the robust loss keeps them from biasing the SOLVE.)
    {
        constexpr int kBuckets = 10;
        const double col_step = (kColHi - kColLo) / kBuckets;
        std::vector<std::array<double, 2>> buck_sum(kBuckets, {0.0, 0.0});
        std::vector<std::array<int, 2>> buck_cnt(kBuckets, {0, 0});
        for (std::size_t i = 0; i < num_points; ++i) {
            if (i % static_cast<std::size_t>(kOutlierEvery) == 0) {
                continue;
            }
            for (const RpcBaMeasure& m : points[i].measures) {
                double c = 0.0;
                double r = 0.0;
                rpc_forward_point(scenes[static_cast<std::size_t>(m.view)],
                                  pts[i].lon,
                                  pts[i].lat,
                                  pts[i].alt,
                                  c,
                                  r);
                const int bucket =
                    std::min(kBuckets - 1,
                             static_cast<int>((m.col - kColLo) / col_step));
                const RpcAffine eff =
                    corrected[static_cast<std::size_t>(m.view)].EffectiveAffine(
                        m.col);
                const double raw_c = c - m.col;
                const double raw_r = r - m.row;
                const double cor_c =
                    (eff.p[0] + (eff.p[1] * c) + (eff.p[2] * r)) - m.col;
                const double cor_r =
                    (eff.p[3] + (eff.p[4] * c) + (eff.p[5] * r)) - m.row;
                auto& s = buck_sum[static_cast<std::size_t>(bucket)];
                auto& n = buck_cnt[static_cast<std::size_t>(bucket)];
                s[0] += (raw_c * raw_c) + (raw_r * raw_r);
                s[1] += (cor_c * cor_c) + (cor_r * cor_r);
                n[0] += 2;
                n[1] += 2;
            }
        }
        std::cout << "residual rms by col tenth [px]   (raw -> corrected)\n";
        for (int b = 0; b < kBuckets; ++b) {
            const auto& s = buck_sum[static_cast<std::size_t>(b)];
            const auto& n = buck_cnt[static_cast<std::size_t>(b)];
            std::cout << "  cols " << std::setw(5)
                      << static_cast<int>(kColLo + b * col_step) << "-"
                      << std::setw(5)
                      << static_cast<int>(kColLo + (b + 1) * col_step) << ":  "
                      << std::setw(7) << std::setprecision(2)
                      << std::sqrt(s[0] / n[0]) << " -> " << std::setw(7)
                      << std::sqrt(s[1] / n[1]) << "\n";
        }
    }

    // Worst-case recovery check: the max corrected residual component over
    // the clean measures (the row-bucket table above is its rms profile).
    // The raw scene affine alone is not the metric -- an affine tilt trades
    // exactly against a zero-mean linear ramp in the band shifts, so the
    // decomposition wobbles while the effective composite stays put.
    {
        double worst = 0.0;
        for (std::size_t i = 0; i < num_points; ++i) {
            if (i % static_cast<std::size_t>(kOutlierEvery) == 0) {
                continue;
            }
            for (const RpcBaMeasure& m : points[i].measures) {
                double c = 0.0;
                double r = 0.0;
                rpc_forward_point(scenes[static_cast<std::size_t>(m.view)],
                                  pts[i].lon,
                                  pts[i].lat,
                                  pts[i].alt,
                                  c,
                                  r);
                const RpcAffine eff =
                    corrected[static_cast<std::size_t>(m.view)].EffectiveAffine(
                        m.col);
                worst = std::max(
                    worst,
                    std::fabs((eff.p[0] + (eff.p[1] * c) + (eff.p[2] * r)) -
                              m.col));
                worst = std::max(
                    worst,
                    std::fabs((eff.p[3] + (eff.p[4] * c) + (eff.p[5] * r)) -
                              m.row));
            }
        }
        std::cout << "\nmax corrected residual over clean measures [px]: "
                  << std::setprecision(2) << worst << "\n";
    }

    std::vector<RpcModel> models;
    models.reserve(scenes.size());
    for (const RpcInfo& v : scenes) {
        models.emplace_back(v);
    }
    double raw_mean = 0.0;
    double cor_mean = 0.0;
    int used = 0;
    for (std::size_t i = 0; i < num_points; ++i) {
        if (i % static_cast<std::size_t>(kOutlierEvery) == 0) {
            continue;
        }
        std::array<double, 2> errs{0.0, 0.0};
        for (int which = 0; which < 2; ++which) {
            std::vector<RpcRay> rays;
            for (const RpcBaMeasure& m : points[i].measures) {
                const RpcInfo& v = scenes[static_cast<std::size_t>(m.view)];
                const double span = std::min(0.9 * v.height_scale, 50.0);
                const RpcAffine eff =
                    which == 1 ? corrected[static_cast<std::size_t>(m.view)]
                                     .EffectiveAffine(m.col)
                               : RpcAffine::Identity();
                RpcRay ray;
                if (zproj::crs::rpc_ray_affine(
                        v,
                        models[static_cast<std::size_t>(m.view)].inverse_init(),
                        eff,
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
                errs[0] = errs[1] = 0.0;
                break;
            }
            const Ecef truth_e = to_ecef(Geodetic{
                pts[i].lon * kDegToRad, pts[i].lat * kDegToRad, pts[i].alt});
            const double dx = p.x() - truth_e.x();
            const double dy = p.y() - truth_e.y();
            const double dz = p.z() - truth_e.z();
            errs[static_cast<std::size_t>(which)] =
                std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
        }
        raw_mean += errs[0];
        cor_mean += errs[1];
        ++used;
    }
    std::cout << "\nN-view triangulation ground error (clean points): raw mean "
              << std::setprecision(1) << raw_mean / used
              << " m -> corrected mean " << cor_mean / used << " m\n";
    return 0;
}
