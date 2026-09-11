// tests/synthetic_rpc.hpp
//
// Synthetic RPC models and ground-point factories shared by the RPC test
// binaries (test_triangulation, test_rpc_affine, ...). Header-only: quoted
// includes from the same directory resolve without any include-dir wiring.
//
// The models deliberately stay simple enough to reason about geometrically:
//   * nadir:  col = lon, row = lat (no height coupling) -- every ray is the
//     geodetic vertical.
//   * oblique: col = lon + k * h -- rays lean with height, so a nadir/oblique
//     pair has a genuine convergence angle and triangulates.
//   * "realistic" variants: the same 50000 px image over a 0.05 deg (~5 km)
//     window, i.e. a real-scene GSD.
#pragma once

#include <random>
#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/wgs84.hpp"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"

namespace zproj_test {

using zproj::crs::RpcInfo;

// Nadir-ish model: col = lon, row = lat (height-independent), so the ray
// through any pixel is the geodetic vertical line.
inline RpcInfo MakeNadirInfo() {
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

    // row = lat
    info.line_num_coeff[2] = 1.0;  // lat
    info.line_den_coeff[0] = 1.0;
    // col = lon
    info.samp_num_coeff[1] = 1.0;  // lon
    info.samp_den_coeff[0] = 1.0;

    info.min_lon = 90.0;
    info.min_lat = 20.0;
    info.max_lon = 110.0;
    info.max_lat = 40.0;
    return info;
}

// Oblique model: col = lon + k*h (height coupling), row = lat. The ray
// through a pixel leans as the back-projected longitude shifts with height,
// so its ray genuinely converges with the nadir model's vertical ray.
inline RpcInfo MakeObliqueInfo() {
    RpcInfo info = MakeNadirInfo();
    // k = 1e-4 in normalized space: ~10 px of col shift across the full
    // +/-height_scale range, and a ~11 deg lean vs the vertical ray.
    info.samp_num_coeff[3] = 1e-4;  // height
    return info;
}

// Realistic-footprint variants of the two models above: same 50000 px image,
// but a 0.05 deg (~5 km) ground window (GSD ~0.1 m) -- the scale of a real
// satellite scene, and the regime the float ENU path is designed for (its
// float grid error scales with the footprint). The oblique height coupling
// is rescaled to keep the same ~11 deg ray convergence.
inline RpcInfo MakeNadirInfoRealistic() {
    RpcInfo info = MakeNadirInfo();
    info.lat_scale = 0.05;
    info.long_scale = 0.05;
    info.min_lon = info.long_off - 0.05;
    info.max_lon = info.long_off + 0.05;
    info.min_lat = info.lat_off - 0.05;
    info.max_lat = info.lat_off + 0.05;
    return info;
}

inline RpcInfo MakeObliqueInfoRealistic() {
    RpcInfo info = MakeNadirInfoRealistic();
    // k * height_scale / ground window ~ tan(11 deg) over the height span.
    info.samp_num_coeff[3] = 0.02;
    return info;
}

// Oblique model with parametric height coupling: col = lon + samp_h * h,
// row = lat + line_h * h -- the ray leans in the given image directions as
// height changes. MakeObliqueInfo() is the special case (1e-4, 0).
inline RpcInfo MakeLeaningInfo(double samp_height_coupling,
                               double line_height_coupling) {
    RpcInfo info = MakeNadirInfo();
    info.samp_num_coeff[3] = samp_height_coupling;
    info.line_num_coeff[3] = line_height_coupling;
    return info;
}

inline RpcInfo MakeLeaningInfoRealistic(double samp_height_coupling,
                                        double line_height_coupling) {
    RpcInfo info = MakeNadirInfoRealistic();
    info.samp_num_coeff[3] = samp_height_coupling;
    info.line_num_coeff[3] = line_height_coupling;
    return info;
}

// A deterministic multi-view RPC set: view 0 is nadir, views 1.. lean in
// rotating azimuths (+col, +row, -col, -row, diagonals, ...) at the same
// ~11 deg convergence as MakeObliqueInfo, so every view pair with the nadir
// ray has a genuine convergence angle and the N-view normal equations are
// well conditioned. num_views >= 1.
inline std::vector<RpcInfo> MakeMultiViewInfos(int num_views) {
    constexpr double k = 1e-4;
    constexpr double kAzimuths[8][2] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
    std::vector<RpcInfo> infos;
    infos.reserve(static_cast<std::size_t>(num_views));
    infos.push_back(MakeNadirInfo());
    for (int i = 1; i < num_views; ++i) {
        const auto& az = kAzimuths[(i - 1) % 8];
        infos.push_back(MakeLeaningInfo(k * az[0], k * az[1]));
    }
    return infos;
}

// Realistic-footprint variant of MakeMultiViewInfos (0.05 deg window, the
// float-ENU regime; same lean as MakeObliqueInfoRealistic).
inline std::vector<RpcInfo> MakeMultiViewInfosRealistic(int num_views) {
    constexpr double k = 0.02;
    constexpr double kAzimuths[8][2] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
    std::vector<RpcInfo> infos;
    infos.reserve(static_cast<std::size_t>(num_views));
    infos.push_back(MakeNadirInfoRealistic());
    for (int i = 1; i < num_views; ++i) {
        const auto& az = kAzimuths[(i - 1) % 8];
        infos.push_back(MakeLeaningInfoRealistic(k * az[0], k * az[1]));
    }
    return infos;
}

// A ground point in the RPC models' geodetic units: degrees / metres.
struct Pt {
    double lon;
    double lat;
    double alt;
};

// Deterministic random lon/lat/alt points inside the models' validity bounds.
inline std::vector<Pt> MakePoints(std::size_t n, const RpcInfo& info) {
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

// Wrap a host vector of (lon, lat, alt) points as a non-owning [N, 3] double
// tensor. `pts` must outlive the returned tensor.
inline zt::Tensor PointTensor(std::vector<Pt>& pts) {
    return zt::from_blob(pts.data(),
                         {static_cast<int64_t>(pts.size()), 3},
                         zt::dtype(zt::kDouble));
}

// ECEF norm of a difference (host-only helper).
inline double EcefDist(const zproj::crs::Ecef& a, const zproj::crs::Ecef& b) {
    const double dx = a.x() - b.x();
    const double dy = a.y() - b.y();
    const double dz = a.z() - b.z();
    return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

}  // namespace zproj_test
