// Two-stage gridded bundle adjustment demo (Ceres): the satellite-refine
// pipeline -- stage 1 solves one global affine per scene held NEAR
// IDENTITY by the pixel-unit prior (satellite RPC georeferencing is
// trusted to a few pixels; the affine must not drift or tilt whole
// scenes), stage 2 freezes those affines and lets a 2D (col x row) grid
// of translation shifts absorb the remaining few-pixel local structure
// (cross-track detector-array error, along-track drift).
//
// A known per-scene affine AND a genuinely 2D error field (separable
// column + row waves) are baked into three synthetic views' projections
// of a set of ground points (plus Gaussian pixel noise, outliers, and
// GCPs anchoring the absolute datum). solve_rpc_bundle_adjust_two_stage
// recovers both stages: the affines stay close to identity, the grid
// tracks the 2D field, and the residual profile flattens from its W
// shape to the noise floor along BOTH axes.
//
// Usage:
//   ./build/default/bin/rpc_bundle_adjust_gridded [points]
//   e.g. ./build/default/bin/rpc_bundle_adjust_gridded 500

#include <algorithm>
#include <array>
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
using zproj::crs::Geodetic;
using zproj::crs::kDegToRad;
using zproj::crs::rpc_forward_point;
using zproj::crs::RpcAffine;
using zproj::crs::RpcAffineGridBasis;
using zproj::crs::RpcAffineGridded;
using zproj::crs::RpcBaGridOptions;
using zproj::crs::RpcBaMeasure;
using zproj::crs::RpcBaPoint;
using zproj::crs::RpcBaReport;
using zproj::crs::RpcBaTwoStageReport;
using zproj::crs::RpcInfo;
using zproj::crs::RpcModel;
using zproj::crs::RpcRay;
using zproj::crs::solve_rpc_bundle_adjust_two_stage;
using zproj::crs::to_ecef;
using zproj::crs::triangulate_nview;

// The synthetic scenes' pixel spans: cols land in [25000, 75000], rows in
// [0, 50000] (off -/+ 0.5*scale, the points' lon/lat window).
constexpr double kColLo = 25000.0;
constexpr double kColHi = 75000.0;
constexpr double kRowLo = 0.0;
constexpr double kRowHi = 50000.0;

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

// The truth affines are deliberately SMALL (a few px, the satellite
// regime the two-stage pipeline is designed for): the identity prior of
// stage 1 keeps the solved affines near identity by construction.
RpcAffine MakeTruthAffine(int view) {
    RpcAffine a;
    a.p = {1.5 + 0.25 * view,
           1.0 + 1e-5 * ((view % 3) - 1),
           -1e-5 * ((view % 2) ? 1 : -1),
           -1.0 - 0.2 * view,
           5e-6 * ((view % 3) - 1),
           1.0 + 1e-5 * ((view % 2) ? 1 : 0)};
    return a;
}

// The 2D error field: one full period along each axis (max at the
// edges/middle, zero at the quarter points). Evaluated at the RAW RPC
// pixel.
double ColWave(double col, double amp) {
    constexpr double kTwoPi = 6.2831853071795865;
    return amp * std::cos(kTwoPi * (col - kColLo) / (kColHi - kColLo));
}
double RowWave(double row, double amp) {
    constexpr double kTwoPi = 6.2831853071795865;
    return amp * std::cos(kTwoPi * (row - kRowLo) / (kRowHi - kRowLo));
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
        (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 800;
    constexpr int kNumScenes = 3;
    constexpr int kNumCol = 8;
    constexpr int kNumRow = 6;
    constexpr double kSigma = 0.3;
    constexpr double kColAmp = 4.0;  // wave amplitudes [px]
    constexpr double kRowAmp = 2.0;
    constexpr int kOutlierEvery = 10;
    constexpr std::size_t kNumGcps =
        static_cast<std::size_t>(kNumCol * kNumRow);  // one per grid cell

    std::vector<RpcInfo> scenes;
    std::vector<RpcAffine> truth;
    for (int v = 0; v < kNumScenes; ++v) {
        scenes.push_back(MakeBaViewInfo(v));
        truth.push_back(MakeTruthAffine(v));
    }

    // Corrupted control network: per-scene truth affine + the 2D field +
    // noise + outliers. GCPs are picked from the tie points at 2D strides
    // (row-major order): every cell region needs its absolute anchor (a
    // region without a GCP carries an unobservable common mode, see
    // rpc_bundle_adjust.hpp).
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
            const double wave_c = ColWave(c, kColAmp) + RowWave(r, kRowAmp);
            const double wave_r =
                ColWave(c, 0.8 * kColAmp) + RowWave(r, 0.8 * kRowAmp);
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
        // One GCP per grid cell: the tie point whose view-0 pixel is
        // nearest the cell center. Every cell region needs its absolute
        // anchor -- a region without a GCP carries an unobservable common
        // mode that drifts by whole pixels (see rpc_bundle_adjust.hpp).
        std::vector<char> taken(num_points, 0);
        const double col_w = (kColHi - kColLo) / kNumCol;
        const double row_h = (kRowHi - kRowLo) / kNumRow;
        for (int r = 0; r < kNumRow; ++r) {
            for (int c = 0; c < kNumCol; ++c) {
                const double cc = kColLo + (c + 0.5) * col_w;
                const double rr = kRowLo + (r + 0.5) * row_h;
                double best = 1e300;
                std::size_t pick = num_points;
                for (std::size_t i = 0; i < num_points; ++i) {
                    if (taken[i]) {
                        continue;
                    }
                    const RpcBaMeasure& m = points[i].measures.front();
                    const double d = ((m.col - cc) * (m.col - cc)) +
                                     ((m.row - rr) * (m.row - rr));
                    if (d < best) {
                        best = d;
                        pick = i;
                    }
                }
                if (pick == num_points) {
                    continue;
                }
                taken[pick] = 1;
                RpcBaPoint gcp = points[pick];
                gcp.ground_fixed = true;
                gcp.lon = pts[pick].lon;
                gcp.lat = pts[pick].lat;
                gcp.height = pts[pick].alt;
                points.push_back(gcp);
                total_measures += static_cast<int>(gcp.measures.size());
            }
        }
    }

    std::cout << "two-stage gridded bundle adjustment: " << kNumScenes
              << " scenes, " << kNumCol << "x" << kNumRow
              << " tent grid (col x row), " << num_points << " tie points + "
              << kNumGcps << " GCPs, " << total_measures << " measures\n2D "
              << "field: one period per axis, amplitudes " << kColAmp << "/"
              << kRowAmp << " px, sigma " << kSigma << " px, outlier every "
              << kOutlierEvery << " points\n\n";

    // The scene corrections travel as RpcAffineGridded values, in and out:
    // identity affines with a uniform zero-shift grid for a fresh solve;
    // the stages overwrite them in place (the affine in stage 1, the cell
    // shifts in stage 2).
    std::vector<RpcAffineGridded> corrected;
    corrected.reserve(static_cast<std::size_t>(kNumScenes));
    for (int s = 0; s < kNumScenes; ++s) {
        auto scene =
            RpcAffineGridded::MakeUniform2D(RpcAffine::Identity(),
                                            kColLo,
                                            kColHi,
                                            kRowLo,
                                            kRowHi,
                                            kNumCol,
                                            kNumRow,
                                            RpcAffineGridBasis::Linear);
        if (!scene) {
            std::cout << "scene " << s << ": bad grid structure\n";
            return 1;
        }
        corrected.push_back(*scene);
    }

    RpcBaGridOptions options;
    options.pixel_sigma = kSigma;
    options.robust_threshold_px = 3.0 * kSigma;
    // Stage 1: the affines are held near identity -- the trusted absolute
    // accuracy of the (synthetic) RPC products.
    options.identity_prior_px = 2.0;
    // Cell regions without a GCP carry their own common mode (all scenes'
    // cells there drift together, absorbed by the ground blocks); the
    // shift prior pins those regions to "no correction relative to the
    // scene affine" (see rpc_bundle_adjust.hpp's observability note).
    options.cell_shift_prior_weight = 0.05;

    const RpcBaTwoStageReport report =
        solve_rpc_bundle_adjust_two_stage(scenes, points, corrected, options);

    std::cout << std::fixed << std::setprecision(3);
    if (!report.affine_stage.ok) {
        std::cout << "stage 1 (affine) FAILED: " << report.affine_stage.message
                  << "\n";
        return 1;
    }
    if (!report.grid_stage.ok) {
        std::cout << "stage 2 (grid) FAILED: " << report.grid_stage.message
                  << "\n";
        return 1;
    }
    std::cout << "stage 1 (global affine, identity prior "
              << std::setprecision(1) << options.identity_prior_px
              << " px): rms " << std::setprecision(3)
              << report.affine_stage.rms_before_px << " px -> "
              << report.affine_stage.rms_after_px << " px\n"
              << "stage 2 (2D grid, affine frozen):                   rms "
              << report.grid_stage.rms_before_px << " px -> "
              << report.grid_stage.rms_after_px << " px\n\n";

    // Residual rms by (row quarter x col tenth), raw vs corrected: the 2D
    // W shape must flatten along BOTH axes. (Clean points only;
    // outlier-corrupted measures stay bad by design -- the robust loss
    // keeps them from biasing the SOLVE.)
    {
        constexpr int kRowBuckets = 4;
        constexpr int kColBuckets = 10;
        const double row_step = (kRowHi - kRowLo) / kRowBuckets;
        const double col_step = (kColHi - kColLo) / kColBuckets;
        std::vector<std::array<double, 2>> sum(
            static_cast<std::size_t>(kRowBuckets * kColBuckets), {0.0, 0.0});
        std::vector<std::array<int, 2>> cnt(
            static_cast<std::size_t>(kRowBuckets * kColBuckets), {0, 0});
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
                const int rb =
                    std::clamp(static_cast<int>((m.row - kRowLo) / row_step),
                               0,
                               kRowBuckets - 1);
                const int cb =
                    std::clamp(static_cast<int>((m.col - kColLo) / col_step),
                               0,
                               kColBuckets - 1);
                const auto b = static_cast<std::size_t>(rb * kColBuckets + cb);
                const RpcAffine eff =
                    corrected[static_cast<std::size_t>(m.view)].EffectiveAffine(
                        m.col, m.row);
                const double raw_c = c - m.col;
                const double raw_r = r - m.row;
                const double cor_c =
                    (eff.p[0] + (eff.p[1] * c) + (eff.p[2] * r)) - m.col;
                const double cor_r =
                    (eff.p[3] + (eff.p[4] * c) + (eff.p[5] * r)) - m.row;
                sum[b][0] += (raw_c * raw_c) + (raw_r * raw_r);
                sum[b][1] += (cor_c * cor_c) + (cor_r * cor_r);
                cnt[b][0] += 2;
                cnt[b][1] += 2;
            }
        }
        std::cout << "residual rms by (row quarter x col tenth) [px]\n";
        std::cout << "             " << std::setprecision(0);
        for (int cb = 0; cb < kColBuckets; ++cb) {
            std::cout << std::setw(11)
                      << static_cast<int>(kColLo + cb * col_step);
        }
        std::cout << "\n";
        for (int rb = 0; rb < kRowBuckets; ++rb) {
            std::cout << "  rows " << std::setw(5)
                      << static_cast<int>(kRowLo + rb * row_step) << "-"
                      << std::setw(5)
                      << static_cast<int>(kRowLo + (rb + 1) * row_step) << ":";
            for (int cb = 0; cb < kColBuckets; ++cb) {
                const auto b = static_cast<std::size_t>(rb * kColBuckets + cb);
                std::cout << " " << std::setw(5) << std::setprecision(2)
                          << std::sqrt(sum[b][0] / cnt[b][0]) << "->"
                          << std::setw(5) << std::sqrt(sum[b][1] / cnt[b][1]);
            }
            std::cout << "\n";
        }
    }

    // Stage 1's affines vs the truth: the identity prior keeps them in
    // the few-pixel regime by construction (deliberately small truth
    // here; the composite below is the real correctness metric).
    {
        std::cout << "\nstage-1 affine deviation from truth [px at the "
                     "domain edge]:\n";
        for (int s = 0; s < kNumScenes; ++s) {
            const RpcAffine& a =
                corrected[static_cast<std::size_t>(s)].affine();
            const RpcAffine& t = truth[static_cast<std::size_t>(s)];
            const double scale[6] = {1.0,
                                     scenes[0].samp_scale,
                                     scenes[0].line_scale,
                                     1.0,
                                     scenes[0].samp_scale,
                                     scenes[0].line_scale};
            double worst = 0.0;
            for (int k = 0; k < 6; ++k) {
                worst = std::max(worst,
                                 std::fabs(a.p[static_cast<std::size_t>(k)] -
                                           t.p[static_cast<std::size_t>(k)]) *
                                     scale[static_cast<std::size_t>(k)]);
            }
            std::cout << "  scene " << s << ": " << std::setprecision(2)
                      << worst << " px\n";
        }
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
                                     .EffectiveAffine(m.col, m.row)
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
