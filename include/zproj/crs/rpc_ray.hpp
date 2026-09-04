// Viewing rays through RPC pixels and their triangulation (intersection).
//
// The math mirrors ASP's RPCModel::point_and_dir (ray construction) and
// VisionWorkbench's StereoModel::triangulate_pair / triangulate_point
// (2-view closed form / N-view Slabaugh normal equations), so results line up
// with the ASP / VisionWorkbench reference.
//
// Everything here is host/device shared (ZT_HOST_DEVICE inline): the same
// per-point math runs inside the CUDA kernel and the CPU (OpenMP) loop, so
// the two implementations cannot drift apart.
#pragma once

#include <array>
#include <cmath>
#include <numbers>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/wgs84.hpp"

namespace zproj::crs {

// Degrees <-> radians. The RPC inverse returns lon/lat in degrees while
// to_ecef / from_ecef work in radians (wgs84.hpp), so every ray and
// triangulation path converts explicitly.
inline constexpr double kDegToRad = std::numbers::pi / 180.0;
inline constexpr double kRadToDeg = 180.0 / std::numbers::pi;

// A viewing ray: a point on the ray (origin, ECEF metres) and its unit
// direction (ECEF, unit length).
struct RpcRay {
    Ecef origin;
    Ecef dir;
};

namespace detail {

// Parallelism threshold on dot(v2, dir_a) = -sin^2 of the convergence angle
// (unit directions): the double reference uses 1e-12; float evaluation noise
// keeps the float instantiation at 1e-10. Both are far below any physical
// convergence angle -- the guard only catches truly parallel/coplanar rays.
template<typename T>
ZT_HOST_DEVICE inline constexpr T kParallelDenTol() noexcept {
    if constexpr (sizeof(T) == 4) {
        return T(1e-10);
    } else {
        return T(1e-12);
    }
}

// Closed-form two-ray intersection, generic over the ray/vector scalar type:
// ONE implementation of the geometry, instantiated for Ecef/double (below)
// and Enu/float (rpc_ray_float.hpp). The vector type only needs arithmetic
// .x()/.y()/.z() components.
template<typename RayT, typename Vec3, typename T>
ZT_HOST_DEVICE inline bool triangulate_pair_impl(const RayT& a,
                                                 const RayT& b,
                                                 Vec3& p,
                                                 T& err) noexcept {
    // v12 = cross(dir_a, dir_b); v1 = cross(v12, dir_a); v2 = cross(v12,
    // dir_b).
    const T v12x = (a.dir.y() * b.dir.z()) - (a.dir.z() * b.dir.y());
    const T v12y = (a.dir.z() * b.dir.x()) - (a.dir.x() * b.dir.z());
    const T v12z = (a.dir.x() * b.dir.y()) - (a.dir.y() * b.dir.x());
    const T v1x = (v12y * a.dir.z()) - (v12z * a.dir.y());
    const T v1y = (v12z * a.dir.x()) - (v12x * a.dir.z());
    const T v1z = (v12x * a.dir.y()) - (v12y * a.dir.x());
    const T v2x = (v12y * b.dir.z()) - (v12z * b.dir.y());
    const T v2y = (v12z * b.dir.x()) - (v12x * b.dir.z());
    const T v2z = (v12x * b.dir.y()) - (v12y * b.dir.x());

    // For unit directions, dot(v2, dir_a) = (dir_a . dir_b)^2 - 1 = -sin^2
    // of the convergence angle: zero exactly when the rays are parallel (the
    // sign is irrelevant -- the threshold below uses fabs).
    const T den_a = (v2x * a.dir.x()) + (v2y * a.dir.y()) + (v2z * a.dir.z());
    const T den_b = (v1x * b.dir.x()) + (v1y * b.dir.y()) + (v1z * b.dir.z());
    constexpr T kTol = kParallelDenTol<T>();
    if (fabs(den_a) <= kTol || fabs(den_b) <= kTol) {
        return false;
    }

    const T wx = b.origin.x() - a.origin.x();
    const T wy = b.origin.y() - a.origin.y();
    const T wz = b.origin.z() - a.origin.z();
    const T ta = ((v2x * wx) + (v2y * wy) + (v2z * wz)) / den_a;
    const T ux = a.origin.x() - b.origin.x();
    const T uy = a.origin.y() - b.origin.y();
    const T uz = a.origin.z() - b.origin.z();
    const T tb = ((v1x * ux) + (v1y * uy) + (v1z * uz)) / den_b;

    const T cax = a.origin.x() + (a.dir.x() * ta);
    const T cay = a.origin.y() + (a.dir.y() * ta);
    const T caz = a.origin.z() + (a.dir.z() * ta);
    const T cbx = b.origin.x() + (b.dir.x() * tb);
    const T cby = b.origin.y() + (b.dir.y() * tb);
    const T cbz = b.origin.z() + (b.dir.z() * tb);

    p = Vec3{T(0.5) * (cax + cbx), T(0.5) * (cay + cby), T(0.5) * (caz + cbz)};
    err = sqrt(((cax - cbx) * (cax - cbx)) + ((cay - cby) * (cay - cby)) +
               ((caz - cbz) * (caz - cbz)));
    return true;
}

}  // namespace detail

// Build the viewing ray through (col,row) of an RPC image by back-projecting
// at two heights and converting both ECEF endpoints. Uses the analytic
// inverse for sub-pixel accuracy (rays feed triangulation, where errors
// compound). Heights are metres above the ellipsoid.
//   h_low, h_high: e.g. info.height_off +/- info.height_scale.
// Returns false if either inverse fails (non-convergence / singular).
ZT_HOST_DEVICE inline bool rpc_ray(const RpcInfo& info,
                                   const RpcInverseInit& init,
                                   double col,
                                   double row,
                                   double h_low,
                                   double h_high,
                                   RpcRay& ray) noexcept {
    // The analytic inverse reaches machine precision; the default affine
    // solver's ~0.1 px error would bias the ray direction and accumulate in
    // the intersection point.
    constexpr double kThreshold = 1e-9;
    constexpr int kMaxIterations = 20;

    double lon_low = 0.0;
    double lat_low = 0.0;
    if (!rpc_inverse_point_analytic(info,
                                    init,
                                    col,
                                    row,
                                    h_low,
                                    lon_low,
                                    lat_low,
                                    kThreshold,
                                    kMaxIterations)) {
        return false;
    }
    double lon_high = 0.0;
    double lat_high = 0.0;
    if (!rpc_inverse_point_analytic(info,
                                    init,
                                    col,
                                    row,
                                    h_high,
                                    lon_high,
                                    lat_high,
                                    kThreshold,
                                    kMaxIterations)) {
        return false;
    }

    // The inverse returns degrees; to_ecef expects radians.
    const Ecef c_low =
        to_ecef(Geodetic{lon_low * kDegToRad, lat_low * kDegToRad, h_low});
    const Ecef c_high =
        to_ecef(Geodetic{lon_high * kDegToRad, lat_high * kDegToRad, h_high});

    const double dx = c_high.x() - c_low.x();
    const double dy = c_high.y() - c_low.y();
    const double dz = c_high.z() - c_low.z();
    const double len = sqrt((dx * dx) + (dy * dy) + (dz * dz));
    if (!(len > 0.0) || !std::isfinite(len)) {
        return false;
    }

    ray.origin = c_low;
    ray.dir = Ecef{dx / len, dy / len, dz / len};
    return true;
}

// Two-view intersection (closed form, cross-product): midpoint of the closest
// points on the two rays. Sets `err` to the distance between the closest
// points [m] (0 = perfect intersection). Returns false if rays are parallel.
// Double wrapper over the precision-generic detail::triangulate_pair_impl,
// which is shared with the float ENU path.
ZT_HOST_DEVICE inline bool triangulate_pair(const RpcRay& a,
                                            const RpcRay& b,
                                            Ecef& p,
                                            double& err) noexcept {
    return detail::triangulate_pair_impl(a, b, p, err);
}

namespace detail {

// Solve the symmetric 3x3 system M [x y z]^T = [r0 r1 r2]^T by Gaussian
// elimination with partial pivoting (no BLAS needed for 3x3). Returns false
// when M is numerically singular (rays coplanar / parallel). Identical math
// on host and device.
ZT_HOST_DEVICE inline bool solve3x3(double m00,
                                    double m01,
                                    double m02,
                                    double m11,
                                    double m12,
                                    double m22,
                                    double r0,
                                    double r1,
                                    double r2,
                                    double& x,
                                    double& y,
                                    double& z) noexcept {
    std::array<std::array<double, 3>, 3> a = {
        std::array<double, 3>{m00, m01, m02},
        std::array<double, 3>{m01, m11, m12},
        std::array<double, 3>{m02, m12, m22}};
    std::array<double, 3> b = {r0, r1, r2};

    // Scale of the matrix, used for a relative singularity threshold.
    double scale = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            scale = fmax(scale, fabs(a[i][j]));
        }
    }
    const double pivot_tol = 1e-14 * fmax(scale, 1e-300);

    for (int col = 0; col < 3; ++col) {
        // Partial pivoting: swap in the row with the largest |entry| in this
        // column.
        int pivot = col;
        double max_abs = fabs(a[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            const double v = fabs(a[row][col]);
            if (v > max_abs) {
                max_abs = v;
                pivot = row;
            }
        }
        if (max_abs <= pivot_tol) {
            return false;
        }
        if (pivot != col) {
            for (int j = 0; j < 3; ++j) {
                const double tmp = a[col][j];
                a[col][j] = a[pivot][j];
                a[pivot][j] = tmp;
            }
            const double tmp = b[col];
            b[col] = b[pivot];
            b[pivot] = tmp;
        }

        // Eliminate below the pivot.
        for (int row = col + 1; row < 3; ++row) {
            const double f = a[row][col] / a[col][col];
            for (int j = col; j < 3; ++j) {
                a[row][j] -= f * a[col][j];
            }
            b[row] -= f * b[col];
        }
    }

    // Back substitution.
    z = b[2] / a[2][2];
    y = (b[1] - (a[1][2] * z)) / a[1][1];
    x = (b[0] - (a[0][1] * y) - (a[0][2] * z)) / a[0][0];
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

}  // namespace detail

// N-view least-squares intersection (Slabaugh normal equations):
//   M = sum(I - d_i d_i^T),  R = sum(I - d_i d_i^T) . origin_i,  P = M^-1 R.
// Sets `rms` to sqrt(mean perpendicular-distance^2). n >= 2. Returns false
// if M is singular (rays coplanar / parallel).
ZT_HOST_DEVICE inline bool triangulate_nview(const RpcRay* rays,
                                             int n,
                                             Ecef& p,
                                             double& rms) noexcept {
    if (n < 2) {
        return false;
    }
    // VW switches the two-ray case to the closed form; the normal equations
    // agree with it, but the closed form is cheaper and more direct.
    if (n == 2) {
        double err = 0.0;
        const bool ok = triangulate_pair(rays[0], rays[1], p, err);
        if (ok) {
            rms = err;
        }
        return ok;
    }

    // Accumulate the symmetric 3x3 normal matrix M and rhs R.
    double m00 = 0.0;
    double m01 = 0.0;
    double m02 = 0.0;
    double m11 = 0.0;
    double m12 = 0.0;
    double m22 = 0.0;
    double r0 = 0.0;
    double r1 = 0.0;
    double r2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double a = rays[i].dir.x();
        const double b = rays[i].dir.y();
        const double c = rays[i].dir.z();
        const double x = rays[i].origin.x();
        const double y = rays[i].origin.y();
        const double z = rays[i].origin.z();
        m00 += 1.0 - (a * a);
        m01 += -(a * b);
        m02 += -(a * c);
        m11 += 1.0 - (b * b);
        m12 += -(b * c);
        m22 += 1.0 - (c * c);
        r0 += ((1.0 - (a * a)) * x) - (a * b * y) - (a * c * z);
        r1 += -(a * b * x) + ((1.0 - (b * b)) * y) - (b * c * z);
        r2 += -(a * c * x) - (b * c * y) + ((1.0 - (c * c)) * z);
    }

    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    if (!detail::solve3x3(
            m00, m01, m02, m11, m12, m22, r0, r1, r2, px, py, pz)) {
        return false;
    }
    p = Ecef{px, py, pz};

    // RMS of the perpendicular residuals: perp_i = (I - d_i d_i^T)(P - o_i),
    // |perp_i|^2 = |P - o_i|^2 - (d_i . (P - o_i))^2.
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dx = p.x() - rays[i].origin.x();
        const double dy = p.y() - rays[i].origin.y();
        const double dz = p.z() - rays[i].origin.z();
        const double v = (rays[i].dir.x() * dx) + (rays[i].dir.y() * dy) +
                         (rays[i].dir.z() * dz);
        const double dist_sq = ((dx * dx) + (dy * dy) + (dz * dz)) - (v * v);
        sum_sq += (dist_sq > 0.0) ? dist_sq : 0.0;
    }
    rms = sqrt(sum_sq / static_cast<double>(n));
    return true;
}

}  // namespace zproj::crs
