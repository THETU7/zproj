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

ZT_HOST_DEVICE inline double vec_dot(const Ecef& a, const Ecef& b) noexcept {
    return (a.x() * b.x()) + (a.y() * b.y()) + (a.z() * b.z());
}

ZT_HOST_DEVICE inline Ecef vec_cross(const Ecef& a, const Ecef& b) noexcept {
    return Ecef{(a.y() * b.z()) - (a.z() * b.y()),
                (a.z() * b.x()) - (a.x() * b.z()),
                (a.x() * b.y()) - (a.y() * b.x())};
}

ZT_HOST_DEVICE inline Ecef vec_sub(const Ecef& a, const Ecef& b) noexcept {
    return Ecef{a.x() - b.x(), a.y() - b.y(), a.z() - b.z()};
}

ZT_HOST_DEVICE inline Ecef vec_add(const Ecef& a, const Ecef& b) noexcept {
    return Ecef{a.x() + b.x(), a.y() + b.y(), a.z() + b.z()};
}

ZT_HOST_DEVICE inline Ecef vec_scale(const Ecef& a, double s) noexcept {
    return Ecef{a.x() * s, a.y() * s, a.z() * s};
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
ZT_HOST_DEVICE inline bool triangulate_pair(const RpcRay& a,
                                            const RpcRay& b,
                                            Ecef& p,
                                            double& err) noexcept {
    using detail::vec_add;
    using detail::vec_cross;
    using detail::vec_dot;
    using detail::vec_scale;
    using detail::vec_sub;

    // v12 = cross(dir_a, dir_b); v1 = cross(v12, dir_a); v2 = cross(v12,
    // dir_b).
    const Ecef v12 = vec_cross(a.dir, b.dir);
    const Ecef v1 = vec_cross(v12, a.dir);
    const Ecef v2 = vec_cross(v12, b.dir);

    // For unit directions, dot(v2, dir_a) = 1 - (dir_a . dir_b)^2 = sin^2 of
    // the convergence angle: zero exactly when the rays are parallel.
    const double den_a = vec_dot(v2, a.dir);
    const double den_b = vec_dot(v1, b.dir);
    if (fabs(den_a) <= 1e-12 || fabs(den_b) <= 1e-12) {
        return false;
    }

    const Ecef closest_a = vec_add(
        a.origin,
        vec_scale(a.dir, vec_dot(v2, vec_sub(b.origin, a.origin)) / den_a));
    const Ecef closest_b = vec_add(
        b.origin,
        vec_scale(b.dir, vec_dot(v1, vec_sub(a.origin, b.origin)) / den_b));

    p = Ecef{0.5 * (closest_a.x() + closest_b.x()),
             0.5 * (closest_a.y() + closest_b.y()),
             0.5 * (closest_a.z() + closest_b.z())};
    err =
        sqrt((closest_a.x() - closest_b.x()) * (closest_a.x() - closest_b.x()) +
             (closest_a.y() - closest_b.y()) * (closest_a.y() - closest_b.y()) +
             (closest_a.z() - closest_b.z()) * (closest_a.z() - closest_b.z()));
    return true;
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
