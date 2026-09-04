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
#include "ztensor/zt/utility/Log.h"

#include "synthetic_rpc.hpp"

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
using zproj_test::EcefDist;
using zproj_test::MakeNadirInfo;
using zproj_test::MakeNadirInfoRealistic;
using zproj_test::MakeObliqueInfo;
using zproj_test::MakeObliqueInfoRealistic;
using zproj_test::MakePoints;
using zproj_test::PointTensor;
using zproj_test::Pt;

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
// both at or below the double path's chord-approximation error.
constexpr double kTolFloatDeg = 1e-6;
constexpr double kTolFloatH = 0.1;
constexpr double kTolFloatRmsM = 0.01;
// Tight float-vs-double drift tripwire (CPU only, so deterministic): the
// measured max deviation on the realistic footprint is ~1e-8 deg / ~2e-2 m;
// these leave one order of magnitude of margin while staying tight enough
// to catch solver drift between the shared-template instantiations.
constexpr double kTolFloatVsDoubleDeg = 1e-7;
constexpr double kTolFloatVsDoubleH = 0.05;
constexpr double kTolFloatVsDoubleRmsM = 1e-2;
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
// reference on the same points. This is the drift tripwire between the two
// instantiations of the shared solver templates (rpc_compute_terms /
// detail::rpc_jac_coeffs / detail::rpc_newton_inverse_core /
// detail::triangulate_pair_impl), so its tolerances are as tight as the
// float noise floor allows.
TEST(RpcStereoFloatCpu, MatchesDoublePath) {
    const RpcInfo left_info = MakeNadirInfoRealistic();
    const RpcInfo right_info = MakeObliqueInfoRealistic();
    const RpcModel left(left_info);
    const RpcModel right(right_info);

    const std::vector<Pt> pts = MakePoints(1024, left_info);
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
    const double* ra = rms_d.data_ptr<double>();
    const double* rb = rms_f.data_ptr<double>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        SCOPED_TRACE("point " + std::to_string(i));
        EXPECT_NEAR(b[3 * i + 0], a[3 * i + 0], kTolFloatVsDoubleDeg);
        EXPECT_NEAR(b[3 * i + 1], a[3 * i + 1], kTolFloatVsDoubleDeg);
        EXPECT_NEAR(b[3 * i + 2], a[3 * i + 2], kTolFloatVsDoubleH);
        EXPECT_NEAR(rb[i], ra[i], kTolFloatVsDoubleRmsM);
    }
}

// Dateline parity: the float inverse must wrap a seed longitude that lands
// on the far side of the globe, exactly like the double analytic inverse.
// Shift the affine seed by -360 deg and require both solvers to still
// converge to the unshifted answer.
TEST(RpcStereoFloatCpu, SeedWrapMatchesDouble) {
    const RpcInfo left_info = MakeNadirInfoRealistic();
    const RpcModel left(left_info);
    const RpcInverseInit init = left.inverse_init();

    // A seed shifted by -360 deg: without the +/-270 deg wrap the Newton
    // iteration would start a full revolution away and fail or diverge.
    RpcInverseInit shifted = init;
    shifted.lon_c0 -= 360.0;
    const zproj::crs::RpcInverseInitFloat init_f =
        zproj::crs::MakeRpcInverseInitFloat(left_info, shifted);
    const zproj::crs::RpcInfoFloat info_f =
        zproj::crs::MakeRpcInfoFloat(left_info);

    const std::vector<Pt> pts = MakePoints(64, left_info);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        SCOPED_TRACE("point " + std::to_string(i));
        double col = 0.0;
        double row = 0.0;
        zproj::crs::rpc_forward_point(
            left_info, pts[i].lon, pts[i].lat, pts[i].alt, col, row);

        double lon_d = 0.0;
        double lat_d = 0.0;
        ASSERT_TRUE(zproj::crs::rpc_inverse_point_analytic(
            left_info, shifted, col, row, pts[i].alt, lon_d, lat_d, 1e-9, 20));

        double lon_f = 0.0;
        double lat_f = 0.0;
        ASSERT_TRUE(zproj::crs::rpc_inverse_point_analytic_float(
            info_f, init_f, col, row, pts[i].alt, lon_f, lat_f, 0.1f, 20));

        // The double wrap recovers machine precision; the float wrap recovers
        // the float floor. Both must agree with the ground truth.
        EXPECT_NEAR(lon_d, pts[i].lon, 1e-9);
        EXPECT_NEAR(lat_d, pts[i].lat, 1e-9);
        EXPECT_NEAR(lon_f, pts[i].lon, 1e-6);
        EXPECT_NEAR(lat_f, pts[i].lat, 1e-6);
    }
}

// End-to-end dateline scene: a stereo pair straddling the 180 deg meridian
// (points expressed with lon on both sides), triangulated in FloatEnu.
TEST(RpcStereoFloatCpu, DatelineScene) {
    RpcInfo left_info = MakeNadirInfoRealistic();
    // Centre the window on the dateline: 179.98 +/- 0.025 deg.
    left_info.long_off = 179.98;
    left_info.min_lon = 179.955;
    left_info.max_lon = 180.005;
    RpcInfo right_info = MakeObliqueInfoRealistic();
    right_info.long_off = 179.98;
    right_info.min_lon = 179.955;
    right_info.max_lon = 180.005;
    const RpcModel left(left_info);
    const RpcModel right(right_info);

    // Ground points across the meridian; lon > 180 is expressed as its
    // -360 deg equivalent (the convention geodetic coordinates use).
    const std::size_t n = 128;
    std::vector<Pt> pts(n);
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> lon(179.955, 180.005);
        std::uniform_real_distribution<double> lat(left_info.lat_off - 0.02,
                                                   left_info.lat_off + 0.02);
        std::uniform_real_distribution<double> alt(left_info.height_off - 250,
                                                   left_info.height_off + 250);
        for (std::size_t i = 0; i < n; ++i) {
            double lon_deg = lon(rng);
            if (lon_deg > 180.0) {
                lon_deg -= 360.0;
            }
            pts[i] = Pt{lon_deg, lat(rng), alt(rng)};
        }
    }
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    const RpcStereo stereo_d(left_info,
                             right_info,
                             left_info.height_off - kRaySpanM,
                             left_info.height_off + kRaySpanM);
    const RpcStereo stereo_f(left_info,
                             right_info,
                             left_info.height_off - kRaySpanM,
                             left_info.height_off + kRaySpanM,
                             StereoPrecision::FloatEnu);
    zt::Tensor ll_d;
    zt::Tensor rms_d;
    stereo_d.triangulate(left_cr, right_cr, ll_d, rms_d);
    zt::Tensor ll_f;
    zt::Tensor rms_f;
    stereo_f.triangulate(left_cr, right_cr, ll_f, rms_f);

    // from_ecef returns lon on the (-180, 180] branch (atan2), so compare
    // longitudes through the branch-agnostic wrapped difference.
    const auto LonDelta = [](double out_lon, double truth_lon) {
        double d = out_lon - truth_lon;
        if (d > 180.0) {
            d -= 360.0;
        } else if (d < -180.0) {
            d += 360.0;
        }
        return d;
    };
    const double* a = ll_d.data_ptr<double>();
    const double* b = ll_f.data_ptr<double>();
    for (std::size_t i = 0; i < n; ++i) {
        SCOPED_TRACE("point " + std::to_string(i));
        EXPECT_NEAR(LonDelta(a[3 * i + 0], pts[i].lon), 0.0, kTolClosedLoopDeg);
        EXPECT_NEAR(a[3 * i + 1], pts[i].lat, kTolClosedLoopDeg);
        EXPECT_NEAR(a[3 * i + 2], pts[i].alt, kTolClosedLoopH);
        EXPECT_NEAR(LonDelta(b[3 * i + 0], pts[i].lon), 0.0, kTolFloatDeg);
        EXPECT_NEAR(b[3 * i + 1], pts[i].lat, kTolFloatDeg);
        EXPECT_NEAR(b[3 * i + 2], pts[i].alt, kTolFloatH);
        // Both outputs take the same atan2 branch, so their difference needs
        // no wrapping.
        EXPECT_NEAR(b[3 * i + 0] - a[3 * i + 0], 0.0, kTolFloatVsDoubleDeg);
        EXPECT_NEAR(b[3 * i + 1] - a[3 * i + 1], 0.0, kTolFloatVsDoubleDeg);
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
