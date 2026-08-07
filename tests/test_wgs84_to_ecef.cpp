// tests/test_wgs84_to_ecef.cpp
//
// Unit tests for zproj::crs::wgs84_to_ecef (the ztensor-backed host API).
//
// Built against the system GoogleTest (find_package(GTest CONFIG), never
// vendored). The CPU path always runs; the CUDA path runs only when a
// CUDA-capable device is present and otherwise skips, matching the rest of
// the zproj test suite.
//
// Expected values come from an independent host-side implementation of the
// WGS84 geodetic -> ECEF formulas plus closed-form analytic points, so the
// tests do not merely re-run the code under test.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#ifdef BUILD_CUDA_MODULE
#include "ztensor/zt/cuda/Guard.h"
#endif  // BUILD_CUDA_MODULE

#include "zproj/crs/wgs84.hpp"
#include "zproj/crs/wgs84_to_ecef.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace {

namespace wgs84 = zproj::crs::wgs84;
using zproj::crs::Ecef;
using zproj::crs::Geodetic;
using zproj::crs::wgs84_to_ecef;

// ~0.1 mm absolute tolerance for coordinates up to ~6.4e6 m.
constexpr double kTol = 1e-4;
constexpr double kPi = std::numbers::pi;
constexpr double kDeg2Rad = kPi / 180.0;

// Independent host-side reference for a single geodetic point.
Ecef ReferenceEcef(const Geodetic& g) {
    const double sin_lat = std::sin(g.y());
    const double cos_lat = std::cos(g.y());
    const double sin_lon = std::sin(g.x());
    const double cos_lon = std::cos(g.x());
    const double n =
        wgs84::kSemiMajorAxis /
        std::sqrt(1.0 - wgs84::kEccentricitySquared * sin_lat * sin_lat);
    return Ecef{(n + g.z()) * cos_lat * cos_lon,
                (n + g.z()) * cos_lat * sin_lon,
                (n * (1.0 - wgs84::kEccentricitySquared) + g.z()) * sin_lat};
}

// Deterministic pseudo-random points with a fixed seed.
std::vector<Geodetic> MakePoints(std::size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> lat_deg(-89.0, 89.0);
    std::uniform_real_distribution<double> lon_deg(-180.0, 180.0);
    std::uniform_real_distribution<double> height(0.0, 1000.0);

    std::vector<Geodetic> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = Geodetic(
            lon_deg(rng) * kDeg2Rad, lat_deg(rng) * kDeg2Rad, height(rng));
    }
    return pts;
}

std::vector<Ecef> ReferenceEcef(const std::vector<Geodetic>& pts) {
    std::vector<Ecef> out;
    out.reserve(pts.size());
    for (const Geodetic& g : pts) {
        out.push_back(ReferenceEcef(g));
    }
    return out;
}

// Wrap a host vector of Geodetic points as a non-owning [N, 3] double tensor.
// `pts` must outlive the returned tensor.
zt::Tensor GeodeticTensor(std::vector<Geodetic>& pts) {
    return zt::from_blob(pts.data(),
                         {static_cast<int64_t>(pts.size()), 3},
                         zt::dtype(zt::kDouble));
}

// Compare a CPU [N, 3] double tensor against the expected ECEF values.
void ExpectNearEcef(const zt::Tensor& out, const std::vector<Ecef>& expected) {
    ASSERT_TRUE(out.is_cpu());
    ASSERT_EQ(out.dim(), 2);
    ASSERT_EQ(out.size(1), 3);
    ASSERT_EQ(out.size(0), static_cast<int64_t>(expected.size()));

    const double* p = out.data_ptr<double>();
    for (int64_t i = 0; i < out.size(0); ++i) {
        const Ecef& e = expected[static_cast<std::size_t>(i)];
        EXPECT_NEAR(p[3 * i + 0], e.x(), kTol) << "point " << i << " x";
        EXPECT_NEAR(p[3 * i + 1], e.y(), kTol) << "point " << i << " y";
        EXPECT_NEAR(p[3 * i + 2], e.z(), kTol) << "point " << i << " z";
    }
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() { return zt::cuda::IsAvailable(); }
#endif  // BUILD_CUDA_MODULE

// ============================== CPU path ================================

TEST(Wgs84ToEcefCpu, KnownPoints) {
    // origin -> (a, 0, 0); north pole -> (0, 0, b); equator lon=90 -> (0, a,
    // 0).
    // (lon, lat, h): origin; lat=90; lon=90; lon=30, lat=45, h=500.
    std::vector<Geodetic> pts{
        Geodetic(0.0, 0.0, 0.0),
        Geodetic(0.0, 90.0 * kDeg2Rad, 0.0),
        Geodetic(90.0 * kDeg2Rad, 0.0, 0.0),
        Geodetic(30.0 * kDeg2Rad, 45.0 * kDeg2Rad, 500.0),
    };

    zt::Tensor in = GeodeticTensor(pts);
    zt::Tensor out;
    wgs84_to_ecef(in, out);

    ASSERT_TRUE(out.is_cpu());
    ASSERT_EQ(out.size(0), 4);
    const double* p = out.data_ptr<double>();
    EXPECT_NEAR(p[0], wgs84::kSemiMajorAxis, kTol);
    EXPECT_NEAR(p[1], 0.0, kTol);
    EXPECT_NEAR(p[2], 0.0, kTol);
    EXPECT_NEAR(p[3], 0.0, kTol);
    EXPECT_NEAR(p[4], 0.0, kTol);
    EXPECT_NEAR(p[5], wgs84::kSemiMinorAxis, kTol);
    EXPECT_NEAR(p[6], 0.0, kTol);
    EXPECT_NEAR(p[7], wgs84::kSemiMajorAxis, kTol);
    EXPECT_NEAR(p[8], 0.0, kTol);

    const Ecef ref = ReferenceEcef(pts[3]);
    EXPECT_NEAR(p[9], ref.x(), kTol);
    EXPECT_NEAR(p[10], ref.y(), kTol);
    EXPECT_NEAR(p[11], ref.z(), kTol);
}

TEST(Wgs84ToEcefCpu, BatchMatchesReference) {
    constexpr int64_t kNumPoints = 4096;
    std::vector<Geodetic> pts =
        MakePoints(static_cast<std::size_t>(kNumPoints));
    const std::vector<Ecef> expected = ReferenceEcef(pts);

    zt::Tensor in = GeodeticTensor(pts);
    zt::Tensor out;
    wgs84_to_ecef(in, out);

    ExpectNearEcef(out, expected);
}

TEST(Wgs84ToEcefCpu, ReusesProvidedOutput) {
    constexpr int64_t kNumPoints = 16;
    std::vector<Geodetic> pts =
        MakePoints(static_cast<std::size_t>(kNumPoints));
    const std::vector<Ecef> expected = ReferenceEcef(pts);

    zt::Tensor in = GeodeticTensor(pts);
    zt::Tensor out = zt::full({kNumPoints, 3}, -1.0, zt::dtype(zt::kDouble));
    wgs84_to_ecef(in, out);

    ASSERT_TRUE(out.is_cpu());
    ASSERT_EQ(out.size(0), kNumPoints);
    ExpectNearEcef(out, expected);
}

// =========================== error handling =============================

TEST(Wgs84ToEcefErrors, WrongInputRankThrows) {
    std::vector<Geodetic> pts{{0.0, 0.0, 0.0}};
    zt::Tensor in = zt::from_blob(pts.data(), {3}, zt::dtype(zt::kDouble));
    zt::Tensor out;
    EXPECT_THROW(wgs84_to_ecef(in, out), std::runtime_error);
}

TEST(Wgs84ToEcefErrors, WrongInputWidthThrows) {
    std::vector<Geodetic> pts{{0.0, 0.0, 0.0}};
    zt::Tensor in = zt::from_blob(pts.data(), {1, 2}, zt::dtype(zt::kDouble));
    zt::Tensor out;
    EXPECT_THROW(wgs84_to_ecef(in, out), std::runtime_error);
}

TEST(Wgs84ToEcefErrors, WrongInputDtypeThrows) {
    zt::Tensor in = zt::zeros({1, 3}, zt::dtype(zt::kFloat));
    zt::Tensor out;
    EXPECT_THROW(wgs84_to_ecef(in, out), std::runtime_error);
}

TEST(Wgs84ToEcefErrors, WrongOutputDtypeThrows) {
    std::vector<Geodetic> pts{{0.0, 0.0, 0.0}};
    zt::Tensor in = GeodeticTensor(pts);
    zt::Tensor out = zt::zeros({1, 3}, zt::dtype(zt::kFloat));
    EXPECT_THROW(wgs84_to_ecef(in, out), std::runtime_error);
}

// ============================== CUDA path ===============================

#ifdef BUILD_CUDA_MODULE
class Wgs84ToEcefCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!HasCudaDevice()) {
            GTEST_SKIP() << "no CUDA-capable device found; CUDA checks skipped";
        }
    }
};

TEST_F(Wgs84ToEcefCudaTest, KnownPoints) {
    // (lon, lat, h): origin; lat=90; lon=90; lon=30, lat=45, h=500.
    std::vector<Geodetic> pts{
        Geodetic(0.0, 0.0, 0.0),
        Geodetic(0.0, 90.0 * kDeg2Rad, 0.0),
        Geodetic(90.0 * kDeg2Rad, 0.0, 0.0),
        Geodetic(30.0 * kDeg2Rad, 45.0 * kDeg2Rad, 500.0),
    };

    zt::Tensor in = GeodeticTensor(pts).cuda();
    zt::Tensor out;
    wgs84_to_ecef(in, out);

    ASSERT_TRUE(out.is_cuda());
    ASSERT_EQ(out.size(0), 4);
    const zt::Tensor out_cpu = out.cpu();
    const double* p = out_cpu.data_ptr<double>();
    EXPECT_NEAR(p[0], wgs84::kSemiMajorAxis, kTol);
    EXPECT_NEAR(p[1], 0.0, kTol);
    EXPECT_NEAR(p[2], 0.0, kTol);
    EXPECT_NEAR(p[3], 0.0, kTol);
    EXPECT_NEAR(p[4], 0.0, kTol);
    EXPECT_NEAR(p[5], wgs84::kSemiMinorAxis, kTol);
    EXPECT_NEAR(p[6], 0.0, kTol);
    EXPECT_NEAR(p[7], wgs84::kSemiMajorAxis, kTol);
    EXPECT_NEAR(p[8], 0.0, kTol);

    const Ecef ref = ReferenceEcef(pts[3]);
    EXPECT_NEAR(p[9], ref.x(), kTol);
    EXPECT_NEAR(p[10], ref.y(), kTol);
    EXPECT_NEAR(p[11], ref.z(), kTol);
}

TEST_F(Wgs84ToEcefCudaTest, BatchMatchesReference) {
    constexpr int64_t kNumPoints = 100'000;  // multiple kernel blocks.
    std::vector<Geodetic> pts =
        MakePoints(static_cast<std::size_t>(kNumPoints));
    const std::vector<Ecef> expected = ReferenceEcef(pts);

    zt::Tensor in = GeodeticTensor(pts).cuda();
    zt::Tensor out;
    wgs84_to_ecef(in, out);

    ASSERT_TRUE(out.is_cuda());
    ExpectNearEcef(out.cpu(), expected);
}

TEST_F(Wgs84ToEcefCudaTest, ReusesProvidedOutput) {
    constexpr int64_t kNumPoints = 16;
    std::vector<Geodetic> pts =
        MakePoints(static_cast<std::size_t>(kNumPoints));
    const std::vector<Ecef> expected = ReferenceEcef(pts);

    zt::Tensor in = GeodeticTensor(pts).cuda();
    zt::Tensor out =
        zt::full({kNumPoints, 3}, -1.0, zt::dtype(zt::kDouble)).cuda();
    wgs84_to_ecef(in, out);

    ASSERT_TRUE(out.is_cuda());
    ASSERT_EQ(out.size(0), kNumPoints);
    ExpectNearEcef(out.cpu(), expected);
}

TEST_F(Wgs84ToEcefCudaTest, CudaMatchesCpu) {
    constexpr int64_t kNumPoints = 4096;
    std::vector<Geodetic> pts =
        MakePoints(static_cast<std::size_t>(kNumPoints));

    zt::Tensor cpu_in = GeodeticTensor(pts);
    zt::Tensor cpu_out;
    wgs84_to_ecef(cpu_in, cpu_out);

    zt::Tensor gpu_in = GeodeticTensor(pts).cuda();
    zt::Tensor gpu_out;
    wgs84_to_ecef(gpu_in, gpu_out);

    ASSERT_TRUE(gpu_out.is_cuda());
    const zt::Tensor gpu_out_cpu = gpu_out.cpu();
    const double* a = cpu_out.data_ptr<double>();
    const double* b = gpu_out_cpu.data_ptr<double>();
    double max_diff = 0.0;
    for (int64_t i = 0; i < 3 * kNumPoints; ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    EXPECT_LT(max_diff, kTol);
}

TEST_F(Wgs84ToEcefCudaTest, MixedCpuInCudaOutThrows) {
    std::vector<Geodetic> pts = MakePoints(4);
    zt::Tensor in = GeodeticTensor(pts);
    zt::Tensor out = zt::zeros({4, 3}, zt::dtype(zt::kDouble)).cuda();
    EXPECT_THROW(wgs84_to_ecef(in, out), std::runtime_error);
}
#endif  // BUILD_CUDA_MODULE

}  // namespace

int main(int argc, char** argv) {
    zt::Logger::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
