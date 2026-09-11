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

// Relative pivot-singularity threshold for the 3x3 normal-equation solve:
// double keeps the tight 1e-14; float elimination noise (~1e-7 relative per
// operation, amplified by pivoting) needs the looser 1e-5 to reject the same
// degenerate bundles the double solve rejects.
template<typename T>
ZT_HOST_DEVICE inline constexpr T kSingularTol() noexcept {
    if constexpr (sizeof(T) == 4) {
        return T(1e-5f);
    } else {
        return T(1e-14);
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
// when M is numerically singular (rays coplanar / parallel). Generic over the
// scalar type: ONE implementation for the double ECEF and float ENU paths.
// Identical math on host and device.
template<typename T>
ZT_HOST_DEVICE inline bool solve3x3(T m00,
                                    T m01,
                                    T m02,
                                    T m11,
                                    T m12,
                                    T m22,
                                    T r0,
                                    T r1,
                                    T r2,
                                    T& x,
                                    T& y,
                                    T& z) noexcept {
    std::array<std::array<T, 3>, 3> a = {std::array<T, 3>{m00, m01, m02},
                                         std::array<T, 3>{m01, m11, m12},
                                         std::array<T, 3>{m02, m12, m22}};
    std::array<T, 3> b = {r0, r1, r2};

    // Scale of the matrix, used for a relative singularity threshold.
    T scale = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            scale = fmax(scale, fabs(a[i][j]));
        }
    }
    const T pivot_tol = kSingularTol<T>() * fmax(scale, T(1e-30));

    for (int col = 0; col < 3; ++col) {
        // Partial pivoting: swap in the row with the largest |entry| in this
        // column.
        int pivot = col;
        T max_abs = fabs(a[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            const T v = fabs(a[row][col]);
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
                const T tmp = a[col][j];
                a[col][j] = a[pivot][j];
                a[pivot][j] = tmp;
            }
            const T tmp = b[col];
            b[col] = b[pivot];
            b[pivot] = tmp;
        }

        // Eliminate below the pivot.
        for (int row = col + 1; row < 3; ++row) {
            const T f = a[row][col] / a[col][col];
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

// Precision-generic N-view core, mirroring triangulate_pair_impl: ONE
// implementation of the Slabaugh normal equations, instantiated for
// Ecef/double (triangulate_nview above) and Enu/float (rpc_ray_float.hpp).
// The vector type only needs arithmetic .x()/.y()/.z() components.
template<typename RayT, typename Vec3, typename T>
ZT_HOST_DEVICE inline bool triangulate_nview_impl(const RayT* rays,
                                                  int n,
                                                  Vec3& p,
                                                  T& rms) noexcept {
    if (n < 2) {
        return false;
    }
    // VW switches the two-ray case to the closed form; the normal equations
    // agree with it, but the closed form is cheaper and more direct.
    if (n == 2) {
        T err = 0;
        const bool ok =
            triangulate_pair_impl<RayT, Vec3, T>(rays[0], rays[1], p, err);
        if (ok) {
            rms = err;
        }
        return ok;
    }

    // Accumulate the symmetric 3x3 normal matrix M and rhs R. The rhs is
    // formed about a reference point c (the first ray's origin): solving for
    // P - c keeps its entries on the bundle scale (metres) instead of the
    // origin magnitudes (6.4e6 m ECEF / km-scale ENU), so the float solve is
    // not amplified by cond(M) * |origin| into centimetres. The 3x3 matrix
    // itself depends only on the directions and needs no centring.
    const Vec3 c = rays[0].origin;
    T m00 = 0;
    T m01 = 0;
    T m02 = 0;
    T m11 = 0;
    T m12 = 0;
    T m22 = 0;
    T r0 = 0;
    T r1 = 0;
    T r2 = 0;
    for (int i = 0; i < n; ++i) {
        const T a = rays[i].dir.x();
        const T b = rays[i].dir.y();
        const T cc = rays[i].dir.z();
        const T x = rays[i].origin.x() - c.x();
        const T y = rays[i].origin.y() - c.y();
        const T z = rays[i].origin.z() - c.z();
        m00 += 1 - (a * a);
        m01 += -(a * b);
        m02 += -(a * cc);
        m11 += 1 - (b * b);
        m12 += -(b * cc);
        m22 += 1 - (cc * cc);
        r0 += ((1 - (a * a)) * x) - (a * b * y) - (a * cc * z);
        r1 += -(a * b * x) + ((1 - (b * b)) * y) - (b * cc * z);
        r2 += -(a * cc * x) - (b * cc * y) + ((1 - (cc * cc)) * z);
    }

    T dx = 0;
    T dy = 0;
    T dz = 0;
    if (!solve3x3(m00, m01, m02, m11, m12, m22, r0, r1, r2, dx, dy, dz)) {
        return false;
    }
    p = Vec3{c.x() + dx, c.y() + dy, c.z() + dz};

    // RMS of the perpendicular residuals: perp_i = (I - d_i d_i^T)(P - o_i),
    // |perp_i|^2 = |P - o_i|^2 - (d_i . (P - o_i))^2. Two accuracy guards,
    // both invisible on the double path (already exact there):
    //   * the directions are renormalized in double: the float path's unit
    //     directions carry a ~6e-8 norm rounding, and (d.u)^2 turns that
    //     into 2*eps*|u|^2 ~ 4e-3 m^2 of fake residual at scene-scale |u|
    //     (hundreds of metres) -- a phantom ~6 cm miss;
    //   * the accumulation itself runs in double: |u|^2 and (d.u)^2 cancel
    //     heavily for near-intersecting bundles, which float would quantize
    //     at the centimetre level. (The float two-ray miss distance is a
    //     direct closest-point difference and needs no such guards.)
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ux = static_cast<double>(p.x()) -
                          static_cast<double>(rays[i].origin.x());
        const double uy = static_cast<double>(p.y()) -
                          static_cast<double>(rays[i].origin.y());
        const double uz = static_cast<double>(p.z()) -
                          static_cast<double>(rays[i].origin.z());
        const double dx = static_cast<double>(rays[i].dir.x());
        const double dy = static_cast<double>(rays[i].dir.y());
        const double dz = static_cast<double>(rays[i].dir.z());
        const double dn = sqrt((dx * dx) + (dy * dy) + (dz * dz));
        const double v = ((dx * ux) + (dy * uy) + (dz * uz)) / dn;
        const double dist_sq = ((ux * ux) + (uy * uy) + (uz * uz)) - (v * v);
        sum_sq += (dist_sq > 0.0) ? dist_sq : 0.0;
    }
    rms = static_cast<T>(sqrt(sum_sq / static_cast<double>(n)));
    return true;
}

}  // namespace detail

// N-view least-squares intersection (Slabaugh normal equations):
//   M = sum(I - d_i d_i^T),  R = sum(I - d_i d_i^T) . origin_i,  P = M^-1 R.
// Sets `rms` to sqrt(mean perpendicular-distance^2). n >= 2. Returns false
// if M is singular (rays coplanar / parallel). Double wrapper over the
// precision-generic detail::triangulate_nview_impl, which is shared with the
// float ENU path.
ZT_HOST_DEVICE inline bool triangulate_nview(const RpcRay* rays,
                                             int n,
                                             Ecef& p,
                                             double& rms) noexcept {
    return detail::triangulate_nview_impl<RpcRay, Ecef, double>(
        rays, n, p, rms);
}

}  // namespace zproj::crs
