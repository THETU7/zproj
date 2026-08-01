// Example: WGS84 geodetic -> ECEF.
//
// Runs the same set of points through two paths and compares them:
//   1. GDAL/PROJ reference (EPSG:4326 geodetic -> EPSG:4978 geocentric), on CPU
//   2. zproj CUDA kernel, on GPU
//
// The zproj math works in radians; PROJ expects degrees, so the reference
// path converts. Eigen is used on the host to compute the error norm.
#include <ogr_spatialref.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "zproj/crs/wgs84.hpp"
#include "zproj/crs/wgs84_to_ecef.hpp"

namespace {

using namespace zproj::crs;

// CPU reference via GDAL/PROJ: EPSG:4326 (lon/lat, degrees) -> EPSG:4978
// (ECEF).
std::vector<Ecef> wgs84_to_ecef_gdal(const std::vector<Geodetic>& geo) {
    OGRSpatialReference src;
    src.importFromEPSG(4326);
    src.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);  // x=lon, y=lat

    OGRSpatialReference dst;
    dst.importFromEPSG(4978);
    dst.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    std::unique_ptr<OGRCoordinateTransformation> ct(
        OGRCreateCoordinateTransformation(&src, &dst));
    if (ct == nullptr) {
        std::cerr << "Failed to build GDAL transformation 4326 -> 4978\n";
        std::exit(EXIT_FAILURE);
    }

    constexpr double kRad2Deg = 180.0 / std::numbers::pi;
    std::vector<Ecef> out(geo.size());
    for (std::size_t i = 0; i < geo.size(); ++i) {
        double x = geo[i].lon * kRad2Deg;
        double y = geo[i].lat * kRad2Deg;
        double z = geo[i].h;
        if (ct->Transform(1, &x, &y, &z) == 0) {
            std::cerr << "GDAL transform failed at index " << i << "\n";
            std::exit(EXIT_FAILURE);
        }
        out[i] = Ecef{x, y, z};
    }
    return out;
}

std::vector<Geodetic> make_points(std::size_t n) {
    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<double> lat_deg(-89.0, 89.0);
    std::uniform_real_distribution<double> lon_deg(-180.0, 180.0);
    std::uniform_real_distribution<double> height(0.0, 1000.0);

    constexpr double kDeg2Rad = std::numbers::pi / 180.0;
    std::vector<Geodetic> geo(n);
    for (std::size_t i = 0; i < n; ++i) {
        geo[i] = Geodetic{
            lat_deg(rng) * kDeg2Rad, lon_deg(rng) * kDeg2Rad, height(rng)};
    }
    return geo;
}

double max_error(const std::vector<Ecef>& a, const std::vector<Ecef>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Eigen::Vector3d va(a[i].x, a[i].y, a[i].z);
        const Eigen::Vector3d vb(b[i].x, b[i].y, b[i].z);
        worst = std::max(worst, (va - vb).norm());
    }
    return worst;
}

}  // namespace

int main() {
    // constexpr std::size_t n = 1'000'000;
    //
    // const auto geo = make_points(n);
    // std::cout << "WGS84 geodetic -> ECEF, n = " << n << " points\n";
    //
    // const auto t0 = std::chrono::steady_clock::now();
    // const auto ref = wgs84_to_ecef_gdal(geo);
    // const auto t1 = std::chrono::steady_clock::now();
    // const auto gpu = wgs84_to_ecef(geo);
    // const auto t2 = std::chrono::steady_clock::now();
    //
    // const double err = max_error(ref, gpu);
    // const double gdal_ms =
    //     std::chrono::duration<double, std::milli>(t1 - t0).count();
    // const double cuda_ms =
    //     std::chrono::duration<double, std::milli>(t2 - t1).count();
    //
    // std::cout << "max |GDAL - CUDA|   = " << err << " m\n";
    // std::cout << "GDAL/PROJ (CPU)     = " << gdal_ms << " ms\n";
    // std::cout << "zproj CUDA (GPU)    = " << cuda_ms
    //           << " ms  (incl. H2D/D2H + sync)\n";
    // if (cuda_ms > 0.0) {
    //     std::cout << "speedup             = " << (gdal_ms / cuda_ms) << "x\n";
    // }

    return 0;
}
