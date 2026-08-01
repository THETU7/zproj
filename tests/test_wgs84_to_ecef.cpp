// Smoke tests for the CUDA WGS84 -> ECEF transform against analytic answers.
//
// Always-on checks (no assert): Release builds compile with NDEBUG, so plain
// asserts would silently vanish. A failed check prints and fails the test.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>

#include "zproj/crs/wgs84.hpp"
#include "zproj/crs/wgs84_to_ecef.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

#define CHECK_NEAR(actual, expected, eps, what) \
    check(std::abs((actual) - (expected)) < (eps), what)

}  // namespace

int main() {
    using namespace zproj::crs;
    constexpr double pi = std::numbers::pi;
    constexpr double kEps = 1e-4;  // ~0.1 mm tolerance for the analytic checks.

    // (lat=0, lon=0, h=0) -> (a, 0, 0)
    {
        const std::vector<Geodetic> geo{{0.0, 0.0, 0.0}};
        const auto out = wgs84_to_ecef(geo);
        CHECK_NEAR(out[0].x, wgs84::kSemiMajorAxis, kEps, "origin x");
        CHECK_NEAR(out[0].y, 0.0, kEps, "origin y");
        CHECK_NEAR(out[0].z, 0.0, kEps, "origin z");
    }

    // North pole (lat=90, lon=0, h=0) -> (0, 0, b)
    {
        const double b = wgs84::kSemiMinorAxis;
        const std::vector<Geodetic> geo{{pi / 2.0, 0.0, 0.0}};
        const auto out = wgs84_to_ecef(geo);
        CHECK_NEAR(out[0].x, 0.0, kEps, "north pole x");
        CHECK_NEAR(out[0].y, 0.0, kEps, "north pole y");
        CHECK_NEAR(out[0].z, b, kEps, "north pole z");
    }

    // Equator at lon=90, h=0 -> (0, a, 0)
    {
        const std::vector<Geodetic> geo{{0.0, pi / 2.0, 0.0}};
        const auto out = wgs84_to_ecef(geo);
        CHECK_NEAR(out[0].x, 0.0, kEps, "equator x");
        CHECK_NEAR(out[0].y, wgs84::kSemiMajorAxis, kEps, "equator y");
        CHECK_NEAR(out[0].z, 0.0, kEps, "equator z");
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "wgs84_to_ecef tests passed\n";
    return EXIT_SUCCESS;
}
