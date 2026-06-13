// Smoke tests for the CUDA WGS84 -> ECEF transform against analytic answers.
#include "zproj/crs/wgs84.hpp"
#include "zproj/crs/wgs84_to_ecef.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

int main()
{
    using namespace zproj::crs;
    constexpr double pi = std::numbers::pi;
    // ~0.1 mm tolerance is more than enough for the analytic checks below.
    const double eps = 1e-4;

    // (lat=0, lon=0, h=0) -> (a, 0, 0)
    {
        const std::vector<Geodetic> geo{{0.0, 0.0, 0.0}};
        const auto out = wgs84_to_ecef(geo);
        assert(std::abs(out[0].x - wgs84::kSemiMajorAxis) < eps);
        assert(std::abs(out[0].y) < eps);
        assert(std::abs(out[0].z) < eps);
    }

    // North pole (lat=90, lon=0, h=0) -> (0, 0, b)
    {
        const double b = wgs84::kSemiMinorAxis;
        const std::vector<Geodetic> geo{{pi / 2.0, 0.0, 0.0}};
        const auto out = wgs84_to_ecef(geo);
        assert(std::abs(out[0].x) < eps);
        assert(std::abs(out[0].y) < eps);
        assert(std::abs(out[0].z - b) < eps);
    }

    // Equator at lon=90, h=0 -> (0, a, 0)
    {
        const std::vector<Geodetic> geo{{0.0, pi / 2.0, 0.0}};
        const auto out = wgs84_to_ecef(geo);
        assert(std::abs(out[0].x) < eps);
        assert(std::abs(out[0].y - wgs84::kSemiMajorAxis) < eps);
        assert(std::abs(out[0].z) < eps);
    }

    std::cout << "wgs84_to_ecef tests passed\n";
    return 0;
}
