// WGS84 reference ellipsoid constants and the geodetic <-> ECEF transforms.
//
// Points in both coordinate spaces are represented by the same vector type,
// zt::eigen::Vec3d (= Eigen::Vector3d), so the geometry flows directly
// through ztensor's Eigen interop (zt::eigen::from_vector / to_vector):
//   * Geodetic: (lon, lat, h) -- x = lon [rad], y = lat [rad], z = h [m]
//   * ECEF:     (x, y, z)     -- metres
//
// Everything here is usable from both host (plain C++) and device (CUDA)
// translation units. The ZT_HOST_DEVICE macro expands to __host__ __device__
// under nvcc and to nothing otherwise, so the identical math is shared between
// the GPU kernel and any host-side reference implementation.
#pragma once

#include <cmath>

#include "ztensor/zt/eigen/EigenConvert.h"
#include "ztensor/zt/Macros.h"

namespace zproj::crs {

// WGS84 reference ellipsoid (EPSG::7030 / WGS84).
namespace wgs84 {
inline constexpr double kSemiMajorAxis = 6378137.0;              // a      [m]
inline constexpr double kInverseFlattening = 298.257223563;      // 1 / f
inline constexpr double kFlattening = 1.0 / kInverseFlattening;  // f
inline constexpr double kEccentricitySquared =
    kFlattening * (2.0 - kFlattening);  // e^2
inline constexpr double kSemiMinorAxis =
    kSemiMajorAxis * (1.0 - kFlattening);  // b
}  // namespace wgs84

// Geodetic coordinates as a Vec3d: x = lon [rad], y = lat [rad], z = h [m].
// (Doubles only -- single precision is too coarse for positions near the
// 6.4e6 m ellipsoid radius.)
using Geodetic = zt::eigen::Vec3d;

// Earth-Centered, Earth-Fixed cartesian coordinates as a Vec3d [m].
using Ecef = zt::eigen::Vec3d;

// Convert a single geodetic point (lon, lat, h) to ECEF. Identical math on
// host and device.
ZT_HOST_DEVICE inline Ecef to_ecef(const Geodetic& g) noexcept {
    using namespace wgs84;
#ifdef __CUDACC__
    // Device: sincos() computes each (sin, cos) pair with one range reduction
    // and a shared polynomial instead of two independent transcendentals,
    // which dominate the kernel's runtime (double-precision math is the
    // bottleneck on consumer GeForce parts).
    double sin_lat = 0.0;
    double cos_lat = 0.0;
    double sin_lon = 0.0;
    double cos_lon = 0.0;
    sincos(g.y(), &sin_lat, &cos_lat);
    sincos(g.x(), &sin_lon, &cos_lon);
#else
    const double sin_lat = sin(g.y());
    const double cos_lat = cos(g.y());
    const double sin_lon = sin(g.x());
    const double cos_lon = cos(g.x());
#endif  // __CUDACC__
    // Prime-vertical radius of curvature.
    const double N =
        kSemiMajorAxis / sqrt(1.0 - kEccentricitySquared * sin_lat * sin_lat);
    return Ecef{(N + g.z()) * cos_lat * cos_lon,
                (N + g.z()) * cos_lat * sin_lon,
                (N * (1.0 - kEccentricitySquared) + g.z()) * sin_lat};
}

// Convert a single ECEF point to geodetic (WGS84) coordinates (lon, lat, h).
// Identical math on host and device.
//
// Bowring's method (B. R. Bowring, "Transformation from spatial to
// geographical coordinates", Survey Review 23(181), 1976) -- the same
// algorithm PROJ's `cart` operation uses for the inverse, so results match the
// GDAL/PROJ reference to round-off. theta is a first approximation of the
// reduced latitude; the atan2() correction yields the geodetic latitude. The
// height is computed by surface-normal projection, which stays accurate
// through the polar cap (see the comment at the height formula below).
ZT_HOST_DEVICE inline Geodetic from_ecef(const Ecef& e) noexcept {
    using namespace wgs84;
    // Perpendicular distance from the point to the Z axis (HM eq. 5-28).
    const double p = hypot(e.x(), e.y());

    // Ancillary ellipsoidal parameters. The second eccentricity is computed
    // as (a-b)(a+b)/b^2 (PROJ's second_eccentricity_squared) instead of
    // e^2/(1-e^2) for better numerical precision.
    const double b = kSemiMinorAxis;
    const double second_e2 =
        (kSemiMajorAxis - b) * (kSemiMajorAxis + b) / (b * b);

    const double theta = atan2(e.z() * kSemiMajorAxis, p * b);
#ifdef __CUDACC__
    double sin_theta = 0.0;
    double cos_theta = 0.0;
    sincos(theta, &sin_theta, &cos_theta);
#else
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
#endif  // __CUDACC__

    // Geodetic latitude (Bowring, 1976) and longitude.
    const double lat =
        atan2(e.z() + second_e2 * b * sin_theta * sin_theta * sin_theta,
              p - kEccentricitySquared * kSemiMajorAxis * cos_theta *
                      cos_theta * cos_theta);
    const double lon = atan2(e.y(), e.x());

    // Prime-vertical radius at the computed latitude, then the height.
#ifdef __CUDACC__
    double sin_lat = 0.0;
    double cos_lat = 0.0;
    sincos(lat, &sin_lat, &cos_lat);
#else
    const double sin_lat = sin(lat);
    const double cos_lat = cos(lat);
#endif  // __CUDACC__
    // Height as the signed distance along the surface normal: since
    // p = (N + h) cos(lat) and z = (N (1 - e^2) + h) sin(lat), the projection
    // of the point onto the normal direction reduces to
    //   p cos(lat) + z sin(lat) = a W + h,   W = sqrt(1 - e^2 sin^2 lat),
    // so h = p cos(lat) + z sin(lat) - a W. This form has no division (the
    // poles need no special case, unlike p / cos(lat) - N) and its partial
    // derivative w.r.t. lat is zero, so the height does not inherit latitude
    // error amplified by tan(lat) near the poles. It matches the GDAL/PROJ
    // reference to round-off over the full globe, polar cap included.
    const double w = sqrt(1.0 - kEccentricitySquared * sin_lat * sin_lat);
    const double h = p * cos_lat + e.z() * sin_lat - kSemiMajorAxis * w;
    // x = lon, y = lat, z = h.
    return Geodetic{lon, lat, h};
}

}  // namespace zproj::crs
