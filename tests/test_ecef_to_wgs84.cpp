// tests/test_ecef_to_wgs84.cpp
//
// Unit tests for zproj::crs::ecef_to_wgs84 (the ztensor-backed host API).
//
// Built against the system GoogleTest (find_package(GTest CONFIG), never
// vendored). The CPU path always runs; the CUDA path runs only when a
// CUDA-capable device is present and otherwise skips, matching the rest of
// the zproj test suite.
//
// Expected values come from an independent host-side implementation of the
// Bowring ECEF -> geodetic formulas plus closed-form analytic points, so the
// tests do not merely re-run the code under test.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#ifdef BUILD_CUDA_MODULE
#include <cuda_runtime.h>
#endif  // BUILD_CUDA_MODULE

#include "zproj/crs/ecef_to_wgs84.hpp"
#include "zproj/crs/wgs84.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace {

namespace wgs84 = zproj::crs::wgs84;
using zproj::crs::Ecef;
using zproj::crs::ecef_to_wgs84;
using zproj::crs::Geodetic;
using zproj::crs::to_ecef;

// Tolerances. The Bowring inverse round-trips geodetic points in the
// tested range with errors below ~1e-13 rad in latitude and ~1e-6 m in
// height, so these give comfortable headroom for both the reference match
// and the CUDA-vs-CPU comparison.
constexpr double kTolAng = 1e-9;  // rad
constexpr double kTolH = 1e-4;    // m
constexpr double kPi = std::numbers::pi;
constexpr double kDeg2Rad = kPi / 180.0;

// Independent host-side reference for a single ECEF point (Bowring, 1976,
// same method as PROJ's `cart` inverse).
Geodetic ReferenceGeodetic(const Ecef& e) {
    const double a = wgs84::kSemiMajorAxis;
    const double b = wgs84::kSemiMinorAxis;
    const double es = wgs84::kEccentricitySquared;
    const double ep2 = (a - b) * (a + b) / (b * b);

    const double p = std::hypot(e.x, e.y);
    const double theta = std::atan2(e.z * a, p * b);
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);

    const double lat =
        std::atan2(e.z + ep2 * b * sin_theta * sin_theta * sin_theta,
                   p - es * a * cos_theta * cos_theta * cos_theta);
    const double lon = std::atan2(e.y, e.x);

    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double n = a / std::sqrt(1.0 - es * sin_lat * sin_lat);

    double h = 0.0;
    if (std::fabs(cos_lat) < 1e-3) {
        h = e.z - (e.z > 0.0 ? b : -b);
    } else {
        h = p / cos_lat - n;
    }
    return Geodetic{lat, lon, h};
}

// Deterministic pseudo-random geodetic points with a fixed seed.
std::vector<Geodetic> MakePoints(std::size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> lat_deg(-89.0, 89.0);
    std::uniform_real_distribution<double> lon_deg(-180.0, 180.0);
    std::uniform_real_distribution<double> height(0.0, 1000.0);

    std::vector<Geodetic> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = Geodetic{
            lat_deg(rng) * kDeg2Rad, lon_deg(rng) * kDeg2Rad, height(rng)};
    }
    return pts;
}

// Deterministic pseudo-random ECEF points with a fixed seed (heights up to
// 10 km, so the round-trip stresses the inverse beyond surface points).
std::vector<Ecef> MakeEcef(std::size_t n) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> lat_deg(-89.0, 89.0);
    std::uniform_real_distribution<double> lon_deg(-180.0, 180.0);
    std::uniform_real_distribution<double> height(-1000.0, 10000.0);

    std::vector<Ecef> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Geodetic g{
            lat_deg(rng) * kDeg2Rad, lon_deg(rng) * kDeg2Rad, height(rng)};
        pts[i] = to_ecef(g);
    }
    return pts;
}

std::vector<Geodetic> ReferenceGeodetic(const std::vector<Ecef>& pts) {
    std::vector<Geodetic> out;
    out.reserve(pts.size());
    for (const Ecef& e : pts) {
        out.push_back(ReferenceGeodetic(e));
    }
    return out;
}

// Wrap a host vector of Ecef points as a non-owning [N, 3] double tensor.
// `pts` must outlive the returned tensor.
zt::Tensor EcefTensor(std::vector<Ecef>& pts) {
    return zt::from_blob(pts.data(),
                         {static_cast<int64_t>(pts.size()), 3},
                         zt::dtype(zt::kDouble));
}

// Compare a CPU [N, 3] double tensor against the expected geodetic values.
void ExpectNearGeodetic(const zt::Tensor& out,
                        const std::vector<Geodetic>& expected) {
    ASSERT_TRUE(out.is_cpu());
    ASSERT_EQ(out.dim(), 2);
    ASSERT_EQ(out.size(1), 3);
    ASSERT_EQ(out.size(0), static_cast<int64_t>(expected.size()));

    const double* p = out.data_ptr<double>();
    for (int64_t i = 0; i < out.size(0); ++i) {
        const Geodetic& g = expected[static_cast<std::size_t>(i)];
        EXPECT_NEAR(p[3 * i + 0], g.lat, kTolAng) << "point " << i << " lat";
        EXPECT_NEAR(p[3 * i + 1], g.lon, kTolAng) << "point " << i << " lon";
        EXPECT_NEAR(p[3 * i + 2], g.h, kTolH) << "point " << i << " h";
    }
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() {
    int n_devices = 0;
    return cudaGetDeviceCount(&n_devices) == cudaSuccess && n_devices > 0;
}
#endif  // BUILD_CUDA_MODULE

// ============================== CPU path ================================

TEST(EcefToWgs84Cpu, KnownPoints) {
    // (a, 0, 0) -> origin; (0, 0, +/-b) -> poles; (0, a, 0) -> lon=90 on the
    // equator; pole heights exercise the |cos(lat)| < 1e-3 guard.
    const double a = wgs84::kSemiMajorAxis;
    const double b = wgs84::kSemiMinorAxis;
    std::vector<Ecef> pts{
        {a, 0.0, 0.0},
        {0.0, 0.0, b},
        {0.0, 0.0, -b},
        {0.0, a, 0.0},
        {0.0, 0.0, b + 1000.0},
        {0.0, 0.0, -(b + 1000.0)},
    };

    zt::Tensor in = EcefTensor(pts);
    zt::Tensor out;
    ecef_to_wgs84(in, out);

    ASSERT_TRUE(out.is_cpu());
    ASSERT_EQ(out.size(0), 6);
    const double* p = out.data_ptr<double>();
    EXPECT_NEAR(p[0], 0.0, kTolAng);
    EXPECT_NEAR(p[1], 0.0, kTolAng);
    EXPECT_NEAR(p[2], 0.0, kTolH);
    EXPECT_NEAR(p[3], kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[4], 0.0, kTolAng);
    EXPECT_NEAR(p[5], 0.0, kTolH);
    EXPECT_NEAR(p[6], -kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[7], 0.0, kTolAng);
    EXPECT_NEAR(p[8], 0.0, kTolH);
    EXPECT_NEAR(p[9], 0.0, kTolAng);
    EXPECT_NEAR(p[10], kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[11], 0.0, kTolH);
    EXPECT_NEAR(p[12], kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[13], 0.0, kTolAng);
    EXPECT_NEAR(p[14], 1000.0, kTolH);
    EXPECT_NEAR(p[15], -kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[16], 0.0, kTolAng);
    EXPECT_NEAR(p[17], -1000.0, kTolH);
}

TEST(EcefToWgs84Cpu, BatchMatchesReference) {
    constexpr int64_t kNumPoints = 4096;
    std::vector<Ecef> pts = MakeEcef(static_cast<std::size_t>(kNumPoints));
    const std::vector<Geodetic> expected = ReferenceGeodetic(pts);

    zt::Tensor in = EcefTensor(pts);
    zt::Tensor out;
    ecef_to_wgs84(in, out);

    ExpectNearGeodetic(out, expected);
}

TEST(EcefToWgs84Cpu, ReusesProvidedOutput) {
    constexpr int64_t kNumPoints = 16;
    std::vector<Ecef> pts = MakeEcef(static_cast<std::size_t>(kNumPoints));
    const std::vector<Geodetic> expected = ReferenceGeodetic(pts);

    zt::Tensor in = EcefTensor(pts);
    zt::Tensor out = zt::full({kNumPoints, 3}, -1.0, zt::dtype(zt::kDouble));
    ecef_to_wgs84(in, out);

    ASSERT_TRUE(out.is_cpu());
    ASSERT_EQ(out.size(0), kNumPoints);
    ExpectNearGeodetic(out, expected);
}

TEST(EcefToWgs84Cpu, RoundTripsGeodeticPoints) {
    // Forward then inverse must recover the original geodetic coordinates.
    constexpr int64_t kNumPoints = 4096;
    std::vector<Geodetic> pts =
        MakePoints(static_cast<std::size_t>(kNumPoints));
    std::vector<Ecef> ecef;
    ecef.reserve(pts.size());
    for (const Geodetic& g : pts) {
        ecef.push_back(to_ecef(g));
    }

    zt::Tensor in = EcefTensor(ecef);
    zt::Tensor out;
    ecef_to_wgs84(in, out);

    ASSERT_TRUE(out.is_cpu());
    const double* p = out.data_ptr<double>();
    for (int64_t i = 0; i < kNumPoints; ++i) {
        const Geodetic& g = pts[static_cast<std::size_t>(i)];
        EXPECT_NEAR(p[3 * i + 0], g.lat, kTolAng) << "point " << i << " lat";
        EXPECT_NEAR(p[3 * i + 1], g.lon, kTolAng) << "point " << i << " lon";
        EXPECT_NEAR(p[3 * i + 2], g.h, kTolH) << "point " << i << " h";
    }
}

// =========================== error handling =============================

TEST(EcefToWgs84Errors, WrongInputRankThrows) {
    std::vector<Ecef> pts{{0.0, 0.0, 0.0}};
    zt::Tensor in = zt::from_blob(pts.data(), {3}, zt::dtype(zt::kDouble));
    zt::Tensor out;
    EXPECT_THROW(ecef_to_wgs84(in, out), std::runtime_error);
}

TEST(EcefToWgs84Errors, WrongInputWidthThrows) {
    std::vector<Ecef> pts{{0.0, 0.0, 0.0}};
    zt::Tensor in = zt::from_blob(pts.data(), {1, 2}, zt::dtype(zt::kDouble));
    zt::Tensor out;
    EXPECT_THROW(ecef_to_wgs84(in, out), std::runtime_error);
}

TEST(EcefToWgs84Errors, WrongInputDtypeThrows) {
    zt::Tensor in = zt::zeros({1, 3}, zt::dtype(zt::kFloat));
    zt::Tensor out;
    EXPECT_THROW(ecef_to_wgs84(in, out), std::runtime_error);
}

TEST(EcefToWgs84Errors, WrongOutputDtypeThrows) {
    std::vector<Ecef> pts{{0.0, 0.0, 0.0}};
    zt::Tensor in = EcefTensor(pts);
    zt::Tensor out = zt::zeros({1, 3}, zt::dtype(zt::kFloat));
    EXPECT_THROW(ecef_to_wgs84(in, out), std::runtime_error);
}

// ============================== CUDA path ===============================

#ifdef BUILD_CUDA_MODULE
class EcefToWgs84CudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!HasCudaDevice()) {
            GTEST_SKIP() << "no CUDA-capable device found; CUDA checks skipped";
        }
    }
};

TEST_F(EcefToWgs84CudaTest, KnownPoints) {
    const double a = wgs84::kSemiMajorAxis;
    const double b = wgs84::kSemiMinorAxis;
    std::vector<Ecef> pts{
        {a, 0.0, 0.0},
        {0.0, 0.0, b},
        {0.0, 0.0, -b},
        {0.0, a, 0.0},
        {0.0, 0.0, b + 1000.0},
    };

    zt::Tensor in = EcefTensor(pts).cuda();
    zt::Tensor out;
    ecef_to_wgs84(in, out);

    ASSERT_TRUE(out.is_cuda());
    ASSERT_EQ(out.size(0), 5);
    const zt::Tensor out_cpu = out.cpu();
    const double* p = out_cpu.data_ptr<double>();
    EXPECT_NEAR(p[0], 0.0, kTolAng);
    EXPECT_NEAR(p[1], 0.0, kTolAng);
    EXPECT_NEAR(p[2], 0.0, kTolH);
    EXPECT_NEAR(p[3], kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[5], 0.0, kTolH);
    EXPECT_NEAR(p[6], -kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[8], 0.0, kTolH);
    EXPECT_NEAR(p[9], 0.0, kTolAng);
    EXPECT_NEAR(p[10], kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[11], 0.0, kTolH);
    EXPECT_NEAR(p[12], kPi / 2.0, kTolAng);
    EXPECT_NEAR(p[14], 1000.0, kTolH);
}

TEST_F(EcefToWgs84CudaTest, BatchMatchesReference) {
    constexpr int64_t kNumPoints = 100'000;  // multiple kernel blocks.
    std::vector<Ecef> pts = MakeEcef(static_cast<std::size_t>(kNumPoints));
    const std::vector<Geodetic> expected = ReferenceGeodetic(pts);

    zt::Tensor in = EcefTensor(pts).cuda();
    zt::Tensor out;
    ecef_to_wgs84(in, out);

    ASSERT_TRUE(out.is_cuda());
    ExpectNearGeodetic(out.cpu(), expected);
}

TEST_F(EcefToWgs84CudaTest, ReusesProvidedOutput) {
    constexpr int64_t kNumPoints = 16;
    std::vector<Ecef> pts = MakeEcef(static_cast<std::size_t>(kNumPoints));
    const std::vector<Geodetic> expected = ReferenceGeodetic(pts);

    zt::Tensor in = EcefTensor(pts).cuda();
    zt::Tensor out =
        zt::full({kNumPoints, 3}, -1.0, zt::dtype(zt::kDouble)).cuda();
    ecef_to_wgs84(in, out);

    ASSERT_TRUE(out.is_cuda());
    ASSERT_EQ(out.size(0), kNumPoints);
    ExpectNearGeodetic(out.cpu(), expected);
}

TEST_F(EcefToWgs84CudaTest, CudaMatchesCpu) {
    constexpr int64_t kNumPoints = 4096;
    std::vector<Ecef> pts = MakeEcef(static_cast<std::size_t>(kNumPoints));

    zt::Tensor cpu_in = EcefTensor(pts);
    zt::Tensor cpu_out;
    ecef_to_wgs84(cpu_in, cpu_out);

    zt::Tensor gpu_in = EcefTensor(pts).cuda();
    zt::Tensor gpu_out;
    ecef_to_wgs84(gpu_in, gpu_out);

    ASSERT_TRUE(gpu_out.is_cuda());
    const zt::Tensor gpu_out_cpu = gpu_out.cpu();
    const double* a = cpu_out.data_ptr<double>();
    const double* b = gpu_out_cpu.data_ptr<double>();
    double max_diff_ang = 0.0;
    double max_diff_h = 0.0;
    for (int64_t i = 0; i < kNumPoints; ++i) {
        max_diff_ang =
            std::max(max_diff_ang, std::abs(a[3 * i + 0] - b[3 * i + 0]));
        max_diff_ang =
            std::max(max_diff_ang, std::abs(a[3 * i + 1] - b[3 * i + 1]));
        max_diff_h =
            std::max(max_diff_h, std::abs(a[3 * i + 2] - b[3 * i + 2]));
    }
    EXPECT_LT(max_diff_ang, kTolAng);
    EXPECT_LT(max_diff_h, kTolH);
}

TEST_F(EcefToWgs84CudaTest, RoundTripsGeodeticPoints) {
    constexpr int64_t kNumPoints = 4096;
    std::vector<Geodetic> pts =
        MakePoints(static_cast<std::size_t>(kNumPoints));
    std::vector<Ecef> ecef;
    ecef.reserve(pts.size());
    for (const Geodetic& g : pts) {
        ecef.push_back(to_ecef(g));
    }

    zt::Tensor in = EcefTensor(ecef).cuda();
    zt::Tensor out;
    ecef_to_wgs84(in, out);

    ASSERT_TRUE(out.is_cuda());
    const zt::Tensor out_cpu = out.cpu();
    const double* p = out_cpu.data_ptr<double>();
    for (int64_t i = 0; i < kNumPoints; ++i) {
        const Geodetic& g = pts[static_cast<std::size_t>(i)];
        EXPECT_NEAR(p[3 * i + 0], g.lat, kTolAng) << "point " << i << " lat";
        EXPECT_NEAR(p[3 * i + 1], g.lon, kTolAng) << "point " << i << " lon";
        EXPECT_NEAR(p[3 * i + 2], g.h, kTolH) << "point " << i << " h";
    }
}

TEST_F(EcefToWgs84CudaTest, MixedCpuInCudaOutThrows) {
    std::vector<Ecef> pts = MakeEcef(4);
    zt::Tensor in = EcefTensor(pts);
    zt::Tensor out = zt::zeros({4, 3}, zt::dtype(zt::kDouble)).cuda();
    EXPECT_THROW(ecef_to_wgs84(in, out), std::runtime_error);
}
#endif  // BUILD_CUDA_MODULE

}  // namespace

int main(int argc, char** argv) {
    zt::Logger::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
