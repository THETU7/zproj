// tests/test_triangulation.cpp
//
// Unit tests for zproj::crs RPC ray construction and triangulation
// (rpc_ray.hpp / RpcStereo).
//
// Built against the system GoogleTest. The CPU path always runs; the CUDA
// path runs only when a CUDA-capable device is present and otherwise skips,
// matching the rest of the zproj test suite.
//
// The strongest test is the closed loop: a known ground point is projected
// into two synthetic RPC images (one nadir, one oblique), triangulated back,
// and compared to the original -- so the test exercises the full forward ->
// ray -> intersection -> geodetic pipeline against an independent geometric
// construction rather than re-running the code under test.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#ifdef BUILD_CUDA_MODULE
#include "ztensor/zt/cuda/Guard.h"
#endif  // BUILD_CUDA_MODULE

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/rpc_ray_float.hpp"
#include "zproj/crs/triangulation.hpp"
#include "zproj/crs/wgs84.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace {

using zproj::crs::Ecef;
using zproj::crs::Geodetic;
using zproj::crs::kDegToRad;
using zproj::crs::kRadToDeg;
using zproj::crs::rpc_ray;
using zproj::crs::RpcInfo;
using zproj::crs::RpcInverseInit;
using zproj::crs::RpcModel;
using zproj::crs::RpcRay;
using zproj::crs::RpcStereo;
using zproj::crs::StereoPrecision;
using zproj::crs::to_ecef;
using zproj::crs::triangulate_nview;
using zproj::crs::triangulate_pair;

// Closed-loop lon/lat tolerance in degrees: the analytic inverse converges to
// ~1e-9 px, and the synthetic models map ~5000 px per degree, so a ray end
// point is accurate to ~2e-13 deg; the intersection inherits that. 1e-9 deg
// keeps a large margin while still proving machine-level closed-loop
// accuracy in lon/lat.
//
// Height is looser (0.1 m): each viewing ray is the STRAIGHT chord between
// its two ECEF end points, while the RPC back-projection locus is slightly
// curved in ECEF. The chord misses the true ground point by roughly the
// tangent deviation, which scales with (h_g - height_off)^2 and is ~mm for
// this synthetic model -- far below the sub-metre budget of the plan.
constexpr double kTolClosedLoopDeg = 1e-9;
constexpr double kTolClosedLoopH = 0.1;
constexpr double kTolClosedLoopRmsM = 1e-3;
// CPU vs CUDA can differ by ~1 ulp in the transcendentals (sincos on device
// vs sin/cos on host); 1e-7 deg is far looser than the measured ~1e-14.
constexpr double kTolCudaDeg = 1e-7;
constexpr double kTolCudaH = 1e-4;
// Point-level ray math tolerances. The intersection POINT is accurate to
// ~2e-9 m, but the perpendicular-distance rms is noise-limited to ~1e-5 m by
// catastrophic cancellation in |P-o|^2 - (d.(P-o))^2 (both terms ~4e6 m^2
// cancel; VW's formula has the same behaviour), so the rms checks are looser.
constexpr double kTolPointM = 1e-6;
constexpr double kTolRmsM = 1e-3;
// Float (ENU) path tolerances, measured on the realistic 5 km footprint:
// the float Newton floors at ~1e-2 px (float evaluation noise), which lands
// ~1e-3 deg-scale horizontal error at 1e-8 deg and ~2e-2 m height error --
// both at or below the double path's chord-approximation error. The
// tolerances below keep an order-of-magnitude margin.
constexpr double kTolFloatDeg = 1e-6;
constexpr double kTolFloatH = 0.1;
constexpr double kTolFloatRmsM = 0.01;
// Float CPU vs CUDA: float rounding of the ENU quantities plus the usual
// transcendental differences; measured ~1e-8 deg.
constexpr double kTolFloatCudaDeg = 1e-5;
constexpr double kTolFloatCudaH = 1e-2;

// Ray end-point height span for the closed-loop tests. ASP's
// RPCModel::point_and_dir clamps the span to min(0.9 * height_scale, 50) m;
// a taller span would make the straight-line ray deviate measurably from the
// curved RPC back-projection locus (the error scales with span^2). +/-50 m is
// still far above the numerical-stability floor for the ray direction.
constexpr double kRaySpanM = 50.0;

// Nadir-ish model: col = lon, row = lat (height-independent), so the ray
// through any pixel is the geodetic vertical line.
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
RpcInfo MakeObliqueInfo() {
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
RpcInfo MakeNadirInfoRealistic() {
    RpcInfo info = MakeNadirInfo();
    info.lat_scale = 0.05;
    info.long_scale = 0.05;
    info.min_lon = info.long_off - 0.05;
    info.max_lon = info.long_off + 0.05;
    info.min_lat = info.lat_off - 0.05;
    info.max_lat = info.lat_off + 0.05;
    return info;
}

RpcInfo MakeObliqueInfoRealistic() {
    RpcInfo info = MakeNadirInfoRealistic();
    // k * height_scale / ground window ~ tan(11 deg) over the height span.
    info.samp_num_coeff[3] = 0.02;
    return info;
}

// Deterministic random lon/lat/alt points inside the models' validity bounds.
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

// Wrap a host vector of (lon, lat, alt) points as a non-owning [N, 3] double
// tensor. `pts` must outlive the returned tensor.
zt::Tensor PointTensor(std::vector<Pt>& pts) {
    return zt::from_blob(pts.data(),
                         {static_cast<int64_t>(pts.size()), 3},
                         zt::dtype(zt::kDouble));
}

// ECEF norm of a difference (host-only helper).
double EcefDist(const Ecef& a, const Ecef& b) {
    const double dx = a.x() - b.x();
    const double dy = a.y() - b.y();
    const double dz = a.z() - b.z();
    return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() { return zt::cuda::IsAvailable(); }
#endif  // BUILD_CUDA_MODULE

// ============================== CPU path ================================

TEST(RpcStereoCpu, ClosedLoopTriangulation) {
    // A known ground point, projected into two different RPC images and
    // triangulated back, must recover the original position to machine-level
    // accuracy.
    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcModel left(left_info);
    const RpcModel right(right_info);

    const std::vector<Pt> pts = MakePoints(64, left_info);

    // Forward the ground points into both images.
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    const double h_low = left_info.height_off - kRaySpanM;
    const double h_high = left_info.height_off + kRaySpanM;
    const RpcStereo stereo(left_info, right_info, h_low, h_high);

    zt::Tensor lonlath;
    zt::Tensor rms;
    stereo.triangulate(left_cr, right_cr, lonlath, rms);

    ASSERT_EQ(lonlath.size(0), static_cast<int64_t>(pts.size()));
    const double* ll = lonlath.data_ptr<double>();
    const double* rm = rms.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        SCOPED_TRACE("point " + std::to_string(i));
        EXPECT_NEAR(ll[3 * i + 0], pts[i].lon, kTolClosedLoopDeg);
        EXPECT_NEAR(ll[3 * i + 1], pts[i].lat, kTolClosedLoopDeg);
        EXPECT_NEAR(ll[3 * i + 2], pts[i].alt, kTolClosedLoopH);
        // A perfect intersection has zero ray-to-ray distance; the closed loop
        // is limited by the ray-curvature chord deviation (sub-mm here).
        EXPECT_LT(rm[i], kTolClosedLoopRmsM) << "rms for a closed-loop point";
    }
}

TEST(RpcStereoCpu, PairMatchesNview) {
    // For two rays, triangulate_nview() must agree with the closed-form
    // triangulate_pair() (VW delegates the n==2 case to the pair formula).
    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcModel left(left_info);
    const RpcModel right(right_info);
    const RpcInverseInit left_init = left.inverse_init();
    const RpcInverseInit right_init = right.inverse_init();
    const double h_low = left_info.height_off - kRaySpanM;
    const double h_high = left_info.height_off + kRaySpanM;

    const std::vector<Pt> pts = MakePoints(64, left_info);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        SCOPED_TRACE("point " + std::to_string(i));
        // Get the pixel in both images, then build the rays directly.
        double col_l = 0.0;
        double row_l = 0.0;
        zproj::crs::rpc_forward_point(
            left_info, pts[i].lon, pts[i].lat, pts[i].alt, col_l, row_l);
        double col_r = 0.0;
        double row_r = 0.0;
        zproj::crs::rpc_forward_point(
            right_info, pts[i].lon, pts[i].lat, pts[i].alt, col_r, row_r);

        RpcRay ray_l;
        RpcRay ray_r;
        ASSERT_TRUE(
            rpc_ray(left_info, left_init, col_l, row_l, h_low, h_high, ray_l));
        ASSERT_TRUE(rpc_ray(
            right_info, right_init, col_r, row_r, h_low, h_high, ray_r));

        Ecef p_pair;
        double err_pair = 0.0;
        ASSERT_TRUE(triangulate_pair(ray_l, ray_r, p_pair, err_pair));

        Ecef p_nview;
        double rms_nview = 0.0;
        const RpcRay rays[2] = {ray_l, ray_r};
        ASSERT_TRUE(triangulate_nview(rays, 2, p_nview, rms_nview));

        EXPECT_LT(EcefDist(p_pair, p_nview), kTolPointM);
        EXPECT_NEAR(rms_nview, err_pair, kTolPointM);
    }
}

TEST(RpcStereoCpu, NviewRecoversKnownPoint) {
    // Three synthetic rays through a known ECEF point must intersect back at
    // it (exercises the Slabaugh normal equations and solve3x3).
    const Ecef g =
        to_ecef(Geodetic{100.5 * kDegToRad, 30.5 * kDegToRad, 750.0});
    const std::array<Ecef, 3> origins = {
        Ecef{g.x() - 1000.0, g.y() - 500.0, g.z() - 2000.0},
        Ecef{g.x() + 800.0, g.y() + 900.0, g.z() - 1500.0},
        Ecef{g.x() - 300.0, g.y() + 1200.0, g.z() - 2500.0},
    };

    RpcRay rays[3];
    for (int i = 0; i < 3; ++i) {
        const double dx = g.x() - origins[i].x();
        const double dy = g.y() - origins[i].y();
        const double dz = g.z() - origins[i].z();
        const double len = std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
        rays[i].origin = origins[i];
        rays[i].dir = Ecef{dx / len, dy / len, dz / len};
    }

    Ecef p;
    double rms = 0.0;
    ASSERT_TRUE(triangulate_nview(rays, 3, p, rms));
    EXPECT_LT(EcefDist(p, g), kTolPointM);
    EXPECT_LT(rms, kTolRmsM);
}

TEST(RpcStereoCpu, ParallelRaysFail) {
    const RpcRay a{Ecef{0.0, 0.0, 0.0}, Ecef{1.0, 0.0, 0.0}};
    const RpcRay b{Ecef{100.0, 50.0, -20.0}, Ecef{1.0, 0.0, 0.0}};
    Ecef p;
    double err = 0.0;
    EXPECT_FALSE(triangulate_pair(a, b, p, err));
    const RpcRay rays[2] = {a, b};
    EXPECT_FALSE(triangulate_nview(rays, 2, p, err));
    // A coplanar/parallel bundle also fails the normal equations.
    const RpcRay rays3[3] = {
        a, b, RpcRay{Ecef{-50.0, 10.0, 5.0}, Ecef{1.0, 0.0, 0.0}}};
    EXPECT_FALSE(triangulate_nview(rays3, 3, p, err));
}

TEST(RpcStereoCpu, DirectionSignInvariant) {
    // Flipping a ray's direction must not move the intersection: the normal
    // equations and the closed form depend on the line, not its orientation.
    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcModel left(left_info);
    const RpcModel right(right_info);
    const RpcInverseInit left_init = left.inverse_init();
    const RpcInverseInit right_init = right.inverse_init();
    const double h_low = left_info.height_off - kRaySpanM;
    const double h_high = left_info.height_off + kRaySpanM;

    const Pt pt = MakePoints(1, left_info)[0];
    double col_l = 0.0;
    double row_l = 0.0;
    zproj::crs::rpc_forward_point(
        left_info, pt.lon, pt.lat, pt.alt, col_l, row_l);
    double col_r = 0.0;
    double row_r = 0.0;
    zproj::crs::rpc_forward_point(
        right_info, pt.lon, pt.lat, pt.alt, col_r, row_r);

    RpcRay ray_l;
    RpcRay ray_r;
    ASSERT_TRUE(
        rpc_ray(left_info, left_init, col_l, row_l, h_low, h_high, ray_l));
    ASSERT_TRUE(
        rpc_ray(right_info, right_init, col_r, row_r, h_low, h_high, ray_r));

    Ecef p0;
    double err0 = 0.0;
    ASSERT_TRUE(triangulate_pair(ray_l, ray_r, p0, err0));

    ray_r.dir = Ecef{-ray_r.dir.x(), -ray_r.dir.y(), -ray_r.dir.z()};
    Ecef p1;
    double err1 = 0.0;
    ASSERT_TRUE(triangulate_pair(ray_l, ray_r, p1, err1));
    EXPECT_LT(EcefDist(p0, p1), kTolPointM);

    const RpcRay rays[2] = {ray_l, ray_r};
    Ecef p2;
    double rms2 = 0.0;
    ASSERT_TRUE(triangulate_nview(rays, 2, p2, rms2));
    EXPECT_LT(EcefDist(p0, p2), kTolPointM);
    EXPECT_NEAR(rms2, err1, kTolPointM);
}

TEST(RpcStereoCpu, FailedInverseWritesHugeVal) {
    // A non-convergent pixel (NaN input) must produce HUGE_VAL outputs,
    // following the GDAL failure convention.
    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcModel left(left_info);
    const RpcModel right(right_info);
    const RpcStereo stereo(left_info,
                           right_info,
                           left_info.height_off - kRaySpanM,
                           left_info.height_off + kRaySpanM);

    // A real ground point, projected into both images, gives a valid pair.
    const Pt pt = MakePoints(1, left_info)[0];
    std::vector<Pt> one{pt};
    zt::Tensor in = PointTensor(one);
    zt::Tensor valid_left;
    zt::Tensor valid_right;
    left.lonlatalt_to_colrow(in, valid_left);
    right.lonlatalt_to_colrow(in, valid_right);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::array<double, 2>> left_cr = {
        {nan, nan},
        {valid_left.data_ptr<double>()[0], valid_left.data_ptr<double>()[1]}};
    const std::vector<std::array<double, 2>> right_cr = {
        {valid_right.data_ptr<double>()[0], valid_right.data_ptr<double>()[1]},
        {valid_right.data_ptr<double>()[0], valid_right.data_ptr<double>()[1]}};
    zt::Tensor left_t = zt::from_blob(
        const_cast<double*>(left_cr[0].data()), {2, 2}, zt::dtype(zt::kDouble));
    zt::Tensor right_t = zt::from_blob(const_cast<double*>(right_cr[0].data()),
                                       {2, 2},
                                       zt::dtype(zt::kDouble));
    zt::Tensor lonlath;
    zt::Tensor rms;
    stereo.triangulate(left_t, right_t, lonlath, rms);

    const double* ll = lonlath.data_ptr<double>();
    const double* rm = rms.data_ptr<double>();
    EXPECT_EQ(ll[0], HUGE_VAL);
    EXPECT_EQ(ll[1], HUGE_VAL);
    EXPECT_EQ(ll[2], HUGE_VAL);
    EXPECT_EQ(rm[0], HUGE_VAL);
    // The valid pixel pair still triangulates back to the original point.
    EXPECT_NEAR(ll[3], pt.lon, kTolClosedLoopDeg);
    EXPECT_NEAR(ll[4], pt.lat, kTolClosedLoopDeg);
    EXPECT_NEAR(ll[5], pt.alt, kTolClosedLoopH);
}

TEST(RpcStereoCpu, ReusesProvidedOutputs) {
    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcModel left(left_info);
    const RpcModel right(right_info);
    const RpcStereo stereo(left_info,
                           right_info,
                           left_info.height_off - kRaySpanM,
                           left_info.height_off + kRaySpanM);

    const std::vector<Pt> pts = MakePoints(8, left_info);
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    zt::Tensor lonlath = zt::zeros({8, 3}, zt::dtype(zt::kDouble));
    zt::Tensor rms = zt::zeros({8}, zt::dtype(zt::kDouble));
    stereo.triangulate(left_cr, right_cr, lonlath, rms);

    const double* ll = lonlath.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(ll[3 * i + 0], pts[i].lon, kTolClosedLoopDeg);
        EXPECT_NEAR(ll[3 * i + 1], pts[i].lat, kTolClosedLoopDeg);
        EXPECT_NEAR(ll[3 * i + 2], pts[i].alt, kTolClosedLoopH);
    }
}

// ============================ float (ENU) path ============================

// Closed loop on a realistic 5 km footprint: the float pipeline must recover
// the ground truth to its documented budget (mm horizontal, cm height).
TEST(RpcStereoFloatCpu, ClosedLoopRealisticFootprint) {
    const RpcInfo left_info = MakeNadirInfoRealistic();
    const RpcInfo right_info = MakeObliqueInfoRealistic();
    const RpcModel left(left_info);
    const RpcModel right(right_info);

    const std::vector<Pt> pts = MakePoints(256, left_info);
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    const RpcStereo stereo(left_info,
                           right_info,
                           left_info.height_off - kRaySpanM,
                           left_info.height_off + kRaySpanM,
                           StereoPrecision::FloatEnu);
    zt::Tensor lonlath;
    zt::Tensor rms;
    stereo.triangulate(left_cr, right_cr, lonlath, rms);

    const double* ll = lonlath.data_ptr<double>();
    const double* rm = rms.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        SCOPED_TRACE("point " + std::to_string(i));
        EXPECT_NEAR(ll[3 * i + 0], pts[i].lon, kTolFloatDeg);
        EXPECT_NEAR(ll[3 * i + 1], pts[i].lat, kTolFloatDeg);
        EXPECT_NEAR(ll[3 * i + 2], pts[i].alt, kTolFloatH);
        EXPECT_LT(rm[i], kTolFloatRmsM) << "rms for a closed-loop point";
    }
}

// The float path must stay within its documented error of the double
// reference on the same points (accuracy regression tripwire).
TEST(RpcStereoFloatCpu, MatchesDoublePath) {
    const RpcInfo left_info = MakeNadirInfoRealistic();
    const RpcInfo right_info = MakeObliqueInfoRealistic();
    const RpcModel left(left_info);
    const RpcModel right(right_info);

    const std::vector<Pt> pts = MakePoints(256, left_info);
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    const double h_low = left_info.height_off - kRaySpanM;
    const double h_high = left_info.height_off + kRaySpanM;
    const RpcStereo stereo_d(left_info, right_info, h_low, h_high);
    const RpcStereo stereo_f(
        left_info, right_info, h_low, h_high, StereoPrecision::FloatEnu);

    zt::Tensor ll_d;
    zt::Tensor rms_d;
    stereo_d.triangulate(left_cr, right_cr, ll_d, rms_d);
    zt::Tensor ll_f;
    zt::Tensor rms_f;
    stereo_f.triangulate(left_cr, right_cr, ll_f, rms_f);

    const double* a = ll_d.data_ptr<double>();
    const double* b = ll_f.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        SCOPED_TRACE("point " + std::to_string(i));
        EXPECT_NEAR(b[3 * i + 0], a[3 * i + 0], kTolFloatDeg);
        EXPECT_NEAR(b[3 * i + 1], a[3 * i + 1], kTolFloatDeg);
        EXPECT_NEAR(b[3 * i + 2], a[3 * i + 2], kTolFloatH);
    }
}

TEST(RpcStereoFloatCpu, FailedInverseWritesHugeVal) {
    const RpcInfo left_info = MakeNadirInfoRealistic();
    const RpcInfo right_info = MakeObliqueInfoRealistic();
    const RpcStereo stereo(left_info,
                           right_info,
                           left_info.height_off - kRaySpanM,
                           left_info.height_off + kRaySpanM,
                           StereoPrecision::FloatEnu);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::array<double, 2>> left_cr = {{nan, nan}};
    const std::vector<std::array<double, 2>> right_cr = {{1.0, 1.0}};
    zt::Tensor left_t = zt::from_blob(
        const_cast<double*>(left_cr[0].data()), {1, 2}, zt::dtype(zt::kDouble));
    zt::Tensor right_t = zt::from_blob(const_cast<double*>(right_cr[0].data()),
                                       {1, 2},
                                       zt::dtype(zt::kDouble));
    zt::Tensor lonlath;
    zt::Tensor rms;
    stereo.triangulate(left_t, right_t, lonlath, rms);

    const double* ll = lonlath.data_ptr<double>();
    EXPECT_EQ(ll[0], HUGE_VAL);
    EXPECT_EQ(ll[1], HUGE_VAL);
    EXPECT_EQ(ll[2], HUGE_VAL);
    EXPECT_EQ(rms.data_ptr<double>()[0], HUGE_VAL);
}

#ifdef BUILD_CUDA_MODULE

class RpcStereoCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!HasCudaDevice()) {
            GTEST_SKIP() << "no CUDA-capable device";
        }
    }
};

TEST_F(RpcStereoCudaTest, MatchesCpu) {
    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcModel left(left_info);
    const RpcModel right(right_info);
    const RpcStereo stereo(left_info,
                           right_info,
                           left_info.height_off - kRaySpanM,
                           left_info.height_off + kRaySpanM);

    const std::vector<Pt> pts = MakePoints(10000, left_info);
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    zt::Tensor cpu_ll;
    zt::Tensor cpu_rms;
    stereo.triangulate(left_cr, right_cr, cpu_ll, cpu_rms);

    zt::Tensor gpu_ll;
    zt::Tensor gpu_rms;
    stereo.triangulate(left_cr.cuda(), right_cr.cuda(), gpu_ll, gpu_rms);

    ASSERT_TRUE(gpu_ll.is_cuda());
    const zt::Tensor gpu_ll_cpu = gpu_ll.cpu();
    const zt::Tensor gpu_rms_cpu = gpu_rms.cpu();

    const double* a = cpu_ll.data_ptr<double>();
    const double* b = gpu_ll_cpu.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(b[3 * i + 0], a[3 * i + 0], kTolCudaDeg)
            << "point " << i << " lon";
        EXPECT_NEAR(b[3 * i + 1], a[3 * i + 1], kTolCudaDeg)
            << "point " << i << " lat";
        EXPECT_NEAR(b[3 * i + 2], a[3 * i + 2], kTolCudaH)
            << "point " << i << " h";
    }
    const double* ra = cpu_rms.data_ptr<double>();
    const double* rb = gpu_rms_cpu.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(rb[i], ra[i], kTolCudaH) << "point " << i << " rms";
    }
}

TEST_F(RpcStereoCudaTest, MixedCpuInCudaOutThrows) {
    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcStereo stereo(left_info,
                           right_info,
                           left_info.height_off - kRaySpanM,
                           left_info.height_off + kRaySpanM);

    const std::vector<Pt> pts = MakePoints(4, left_info);
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    RpcModel(left_info).lonlatalt_to_colrow(in, left_cr);
    RpcModel(right_info).lonlatalt_to_colrow(in, right_cr);

    zt::Tensor lonlath = zt::zeros({4, 3}, zt::dtype(zt::kDouble)).cuda();
    zt::Tensor rms = zt::zeros({4}, zt::dtype(zt::kDouble)).cuda();
    EXPECT_THROW(stereo.triangulate(left_cr, right_cr, lonlath, rms),
                 std::runtime_error);
}

TEST_F(RpcStereoCudaTest, FloatEnuMatchesCpu) {
    const RpcInfo left_info = MakeNadirInfoRealistic();
    const RpcInfo right_info = MakeObliqueInfoRealistic();
    const RpcModel left(left_info);
    const RpcModel right(right_info);
    const RpcStereo stereo(left_info,
                           right_info,
                           left_info.height_off - kRaySpanM,
                           left_info.height_off + kRaySpanM,
                           StereoPrecision::FloatEnu);

    const std::vector<Pt> pts = MakePoints(10000, left_info);
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    zt::Tensor cpu_ll;
    zt::Tensor cpu_rms;
    stereo.triangulate(left_cr, right_cr, cpu_ll, cpu_rms);

    zt::Tensor gpu_ll;
    zt::Tensor gpu_rms;
    stereo.triangulate(left_cr.cuda(), right_cr.cuda(), gpu_ll, gpu_rms);

    ASSERT_TRUE(gpu_ll.is_cuda());
    const zt::Tensor gpu_ll_cpu = gpu_ll.cpu();
    const zt::Tensor gpu_rms_cpu = gpu_rms.cpu();

    const double* a = cpu_ll.data_ptr<double>();
    const double* b = gpu_ll_cpu.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(b[3 * i + 0], a[3 * i + 0], kTolFloatCudaDeg)
            << "point " << i << " lon";
        EXPECT_NEAR(b[3 * i + 1], a[3 * i + 1], kTolFloatCudaDeg)
            << "point " << i << " lat";
        EXPECT_NEAR(b[3 * i + 2], a[3 * i + 2], kTolFloatCudaH)
            << "point " << i << " h";
    }
    const double* ra = cpu_rms.data_ptr<double>();
    const double* rb = gpu_rms_cpu.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(rb[i], ra[i], kTolFloatCudaH) << "point " << i << " rms";
    }
}

#endif  // BUILD_CUDA_MODULE

}  // namespace

int main(int argc, char** argv) {
    zt::Logger::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
