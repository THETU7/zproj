// WGS84 reference ellipsoid constants and the geodetic -> ECEF transform.
//
// Everything here is usable from both host (plain C++) and device (CUDA)
// translation units. The ZPROJ_HD macro expands to __host__ __device__ under
// nvcc and to nothing otherwise, so the identical math is shared between the
// GPU kernel and any host-side reference implementation.
#pragma once

#include <cmath>

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

// Geodetic coordinates. Angles are in radians.
struct Geodetic {
    double lat = 0.0;  // latitude  [rad]
    double lon = 0.0;  // longitude [rad]
    double h = 0.0;    // ellipsoidal height [m]
};

// Earth-Centered, Earth-Fixed cartesian coordinates [m].
struct Ecef {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// Convert a single geodetic point to ECEF. Identical math on host and device.
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
    sincos(g.lat, &sin_lat, &cos_lat);
    sincos(g.lon, &sin_lon, &cos_lon);
#else
    const double sin_lat = sin(g.lat);
    const double cos_lat = cos(g.lat);
    const double sin_lon = sin(g.lon);
    const double cos_lon = cos(g.lon);
#endif  // __CUDACC__
    // Prime-vertical radius of curvature.
    const double N =
        kSemiMajorAxis / sqrt(1.0 - kEccentricitySquared * sin_lat * sin_lat);
    return Ecef{
        (N + g.h) * cos_lat * cos_lon,
        (N + g.h) * cos_lat * sin_lon,
        (N * (1.0 - kEccentricitySquared) + g.h) * sin_lat,
    };
}

}  // namespace zproj::crs
