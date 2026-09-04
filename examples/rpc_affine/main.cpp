// RPC affine bias-correction demo (Ceres): the companion to the
// triangulation example, on the same synthetic nadir + oblique pair.
//
// A known affine corruption is baked into both images' projections of a set
// of ground points (plus Gaussian pixel noise and a few gross outliers), and
// solve_rpc_affine() recovers the affines from the corrupted matches alone.
// The demo then shows what the correction buys: triangulating the matches
// with the RAW RPCs leaves metre-scale ground error (the corruption pushed
// the pixels), while triangulating through the corrected affines returns to
// the noise floor.
//
// Usage:
//   ./build/default/bin/rpc_affine [points]
//   e.g. ./build/default/bin/rpc_affine 500

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_affine.hpp"
#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/wgs84.hpp"

namespace {

using zproj::crs::Ecef;
using zproj::crs::from_ecef;
using zproj::crs::Geodetic;
using zproj::crs::kDegToRad;
using zproj::crs::rpc_forward_point;
using zproj::crs::RpcAffine;
using zproj::crs::RpcAffineOptions;
using zproj::crs::RpcGcp;
using zproj::crs::RpcInfo;
using zproj::crs::RpcInverseInit;
using zproj::crs::RpcMatch;
using zproj::crs::RpcModel;
using zproj::crs::RpcRay;
using zproj::crs::solve_rpc_affine;
using zproj::crs::to_ecef;
using zproj::crs::triangulate_pair;

// The affine-test pair from test_rpc_affine.cpp: the plain synthetic models
// plus realistic nonlinearity and height sensitivity (the H / LH / PH terms)
// -- the coupling that makes the affine identifiable from matches alone (see
// rpc_affine.hpp: weakly height-coupled pairs leave a flat gauge that LM
// crawls along).
RpcInfo MakeNadirInfo() {
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
    info.samp_num_coeff[3] = -5e-4;    // + height (opposite lean)
    info.samp_num_coeff[4] = 1e-2;     // + lon * lat
    info.samp_num_coeff[5] = 2e-2;     // + lon * height
    info.samp_num_coeff[7] = 2e-2;     // + lon^2
    info.line_num_coeff[4] = 1e-2;     // + lon * lat
    info.line_num_coeff[6] = -1.5e-2;  // + lat * height
    info.line_num_coeff[8] = 2e-2;     // + lat^2
    info.min_lon = 90.0;
    info.min_lat = 20.0;
    info.max_lon = 110.0;
    info.max_lat = 40.0;
    return info;
}

RpcInfo MakeObliqueInfo() {
    RpcInfo info = MakeNadirInfo();
    info.samp_num_coeff[3] = 5e-3;    // height (off-nadir lean)
    info.samp_num_coeff[5] = -2e-2;   // lon * height
    info.samp_num_coeff[7] = 3e-2;    // lon^2
    info.samp_num_coeff[9] = 1e-3;    // height^2
    info.line_num_coeff[6] = 1.5e-2;  // lat * height
    return info;
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

// Ground-truth corruption: translations and ~1e-4-scale linear terms, the
// magnitude a real RPC product's systematic error maps to in pixels.
RpcAffine MakeTruthLeft() {
    RpcAffine a;
    a.p = {3.5, 1.0003, -1e-4, -2.2, 8e-5, 0.9999};
    return a;
}

RpcAffine MakeTruthRight() {
    RpcAffine a;
    a.p = {-1.25, 0.9997, 1.5e-4, 4.0, -6e-5, 1.0002};
    return a;
}

void PrintAffine(const char* name, const RpcAffine& a) {
    std::cout << "  " << std::left << std::setw(6) << name << " e = ["
              << std::fixed << std::setprecision(6) << std::setw(10) << a.p[0]
              << std::setw(10) << a.p[1] << std::setw(10) << a.p[2]
              << "]  f = [" << std::setw(10) << a.p[3] << std::setw(10)
              << a.p[4] << std::setw(10) << a.p[5] << "]\n";
}

// Triangulate every match through the given affines and return the median
// and max ECEF distance to the true ground points.
struct GroundError {
    double median_m;
    double max_m;
};

GroundError GroundErrorVsTruth(const RpcInfo& left,
                               const RpcInfo& right,
                               const RpcAffine& left_affine,
                               const RpcAffine& right_affine,
                               const std::vector<RpcMatch>& matches,
                               const std::vector<Pt>& truth) {
    const RpcModel left_model(left);
    const RpcModel right_model(right);
    const double h_low = left.height_off - 50.0;
    const double h_high = left.height_off + 50.0;

    std::vector<double> dists;
    dists.reserve(matches.size());
    for (std::size_t i = 0; i < matches.size(); ++i) {
        RpcRay ray_l;
        RpcRay ray_r;
        Ecef p;
        double err = 0.0;
        if (!zproj::crs::rpc_ray_affine(left,
                                        left_model.inverse_init(),
                                        left_affine,
                                        matches[i].left_col,
                                        matches[i].left_row,
                                        h_low,
                                        h_high,
                                        ray_l) ||
            !zproj::crs::rpc_ray_affine(right,
                                        right_model.inverse_init(),
                                        right_affine,
                                        matches[i].right_col,
                                        matches[i].right_row,
                                        h_low,
                                        h_high,
                                        ray_r) ||
            !triangulate_pair(ray_l, ray_r, p, err)) {
            continue;
        }
        const Ecef true_p = to_ecef(Geodetic{
            truth[i].lon * kDegToRad, truth[i].lat * kDegToRad, truth[i].alt});
        double dx = p.x() - true_p.x();
        double dy = p.y() - true_p.y();
        double dz = p.z() - true_p.z();
        dists.push_back(std::sqrt((dx * dx) + (dy * dy) + (dz * dz)));
    }
    GroundError out{0.0, 0.0};
    if (dists.empty()) {
        return out;
    }
    std::sort(dists.begin(), dists.end());
    out.median_m = dists[dists.size() / 2];
    out.max_m = dists.back();
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 500;
    const double sigma = 0.2;  // px, matching noise

    const RpcInfo left = MakeNadirInfo();
    const RpcInfo right = MakeObliqueInfo();
    const RpcAffine truth_l = MakeTruthLeft();
    const RpcAffine truth_r = MakeTruthRight();

    // Observations: true projections corrupted by the affines, noise, and a
    // gross outlier on every 10th match.
    const std::vector<Pt> pts = MakePoints(n, left);
    std::vector<RpcMatch> matches;
    matches.reserve(n);
    {
        std::mt19937 rng(7);
        std::normal_distribution<double> noise(0.0, sigma);
        for (std::size_t i = 0; i < n; ++i) {
            double cl = 0.0;
            double rl = 0.0;
            double cr = 0.0;
            double rr = 0.0;
            rpc_forward_point(left, pts[i].lon, pts[i].lat, pts[i].alt, cl, rl);
            rpc_forward_point(
                right, pts[i].lon, pts[i].lat, pts[i].alt, cr, rr);
            truth_l.Apply(cl, rl, cl, rl);
            truth_r.Apply(cr, rr, cr, rr);
            cl += noise(rng);
            rl += noise(rng);
            cr += noise(rng);
            rr += noise(rng);
            if (i % 10 == 0) {
                cr += 25.0;  // outlier
            }
            matches.push_back(RpcMatch{cl, rl, cr, rr});
        }
    }

    std::cout << "RPC affine bias correction (Ceres), synthetic stereo pair\n";
    std::cout
        << "  matches: " << n << ", pixel noise sigma: " << sigma
        << " px, outliers: every 10th (+25 px), GCPs: 10 (both images)\n\n";
    std::cout << "  ground-truth corruption:\n";
    PrintAffine("left", truth_l);
    PrintAffine("right", truth_r);

    // A handful of ground control points (the first 10 truth points,
    // observed in both images) anchor the geodetic datum -- the production
    // configuration: matches alone leave the weakly observable
    // height-vs-translation direction free (see rpc_affine.hpp).
    std::vector<RpcGcp> left_gcps;
    std::vector<RpcGcp> right_gcps;
    {
        std::mt19937 grng(97);
        std::normal_distribution<double> gnoise(0.0, sigma);
        for (std::size_t i = 0; i < 10 && i < pts.size(); ++i) {
            double cl = 0.0;
            double rl = 0.0;
            double cr = 0.0;
            double rr = 0.0;
            rpc_forward_point(left, pts[i].lon, pts[i].lat, pts[i].alt, cl, rl);
            rpc_forward_point(
                right, pts[i].lon, pts[i].lat, pts[i].alt, cr, rr);
            truth_l.Apply(cl, rl, cl, rl);
            truth_r.Apply(cr, rr, cr, rr);
            left_gcps.push_back(RpcGcp{pts[i].lon,
                                       pts[i].lat,
                                       pts[i].alt,
                                       cl + gnoise(grng),
                                       rl + gnoise(grng)});
            right_gcps.push_back(RpcGcp{pts[i].lon,
                                        pts[i].lat,
                                        pts[i].alt,
                                        cr + gnoise(grng),
                                        rr + gnoise(grng)});
        }
    }
    // Solve: full 6-parameter affines, Huber robust loss at 3 sigma. GCPs
    // anchor the solution, so no prior is needed.
    RpcAffineOptions options;
    options.pixel_sigma = sigma;
    options.robust_threshold_px = 3.0 * sigma;
    RpcAffine out_l;
    RpcAffine out_r;
    const zproj::crs::RpcAffineReport report = solve_rpc_affine(
        left, right, matches, left_gcps, right_gcps, out_l, out_r, options);

    std::cout << "\n  solver: " << (report.ok ? "converged" : "FAILED") << "  ("
              << report.num_matches << " matches, "
              << report.num_matches_skipped << " skipped)\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  reprojection RMS: " << report.rms_before_px << " px -> "
              << report.rms_after_px << " px\n\n";

    std::cout << "  recovered affine:\n";
    PrintAffine("left", out_l);
    PrintAffine("right", out_r);

    // What the correction buys on the ground.
    const GroundError raw = GroundErrorVsTruth(left,
                                               right,
                                               RpcAffine::Identity(),
                                               RpcAffine::Identity(),
                                               matches,
                                               pts);
    const GroundError fixed =
        GroundErrorVsTruth(left, right, out_l, out_r, matches, pts);
    std::cout << "\n  triangulated ground error vs truth (ECEF):\n";
    std::cout << "    raw RPC       : median " << std::fixed
              << std::setprecision(2) << std::setw(10) << raw.median_m
              << " m, max " << raw.max_m << " m\n";
    std::cout << "    corrected RPC : median " << std::setw(10)
              << fixed.median_m << " m, max " << fixed.max_m << " m\n";

    return report.ok ? 0 : 1;
}
