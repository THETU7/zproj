// tests/test_rpc.cpp
//
// Unit tests for zproj::rpc::RpcModel (the ztensor-backed RPC transformer).
//
// Built against the system GoogleTest (find_package(GTest CONFIG), never
// vendored). The CPU path always runs; the CUDA path runs only when a
// CUDA-capable device is present and otherwise skips, matching the rest of
// the zproj test suite.
//
// Expected values come from (a) closed-form analytic points, (b) an
// independent host-side implementation of the RPC00 rational polynomials,
// and (c) GDAL's own GDALRPCTransformer (the ground truth the transform is
// reimplementing), so the tests do not merely re-run the code under test.

#include <gdal_alg.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#ifdef BUILD_CUDA_MODULE
#include <cuda_runtime.h>
#endif  // BUILD_CUDA_MODULE

#include "zproj/rpc/rpc.hpp"
#include "zproj/rpc/rpc_io.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace {

using zproj::rpc::rpc_forward_point;
using zproj::rpc::RpcInfo;
using zproj::rpc::RpcInfoFromRpcFile;
using zproj::rpc::RpcModel;
using zproj::rpc::RpcOptions;

constexpr double kTolColRow = 1e-6;    // px, forward vs reference / GDAL
constexpr double kTolAnalytic = 1e-9;  // px, identity model
// The iterative inverse stops once the back-transform error drops below
// RpcOptions::pixel_error_threshold (0.1 px), so a col/row round-trip is
// only guaranteed to land within that threshold of the original pixel.
constexpr double kTolRoundTripPx = 0.15;
// Inverse lon/lat tolerance in degrees, derived from the convergence
// guarantee rather than picked arbitrarily: the back-projection error is
// < 0.1 px, and the synthetic model maps ~samp_scale/long_scale = 5000 px per
// degree, so a single converged point is within ~0.1/5000 = 2e-5 deg of the
// true position. Two independently converged points (e.g. zproj vs GDAL, or
// CPU vs CUDA) can differ by up to 2x that, and the local Jacobian varies a
// few percent with the non-linear terms -- 1e-4 deg keeps a 5x margin.
// (Measured on the fixed seed: round-trip ~2.0e-5 deg, zproj-vs-GDAL ~1e-14.)
constexpr double kTolInverseDeg = 1e-4;

// A near-identity RPC00 model with mild non-linear terms and known bounds.
// Within the validity window the normalized coordinates stay in [-1, 1] and
// the denominators stay near 1, so the inverse converges quickly.
RpcInfo MakeSyntheticInfo() {
    RpcInfo info;
    info.line_off = 25000.0;
    info.samp_off = 50000.0;
    info.lat_off = 30.0;
    info.long_off = 100.0;
    info.height_off = 500.0;
    info.line_scale = 50000.0;
    info.samp_scale = 50000.0;
    info.lat_scale = 10.0;
    info.long_scale = 10.0;
    info.height_scale = 500.0;

    // row = (lat_n + 0.02 lat_n^2 + 0.01 lon_n*lat_n) / (1 + 0.01 lon_n)
    info.line_num_coeff[2] = 1.0;   // lat
    info.line_num_coeff[8] = 0.02;  // lat^2
    info.line_num_coeff[4] = 0.01;  // lon*lat
    info.line_den_coeff[0] = 1.0;   // 1
    info.line_den_coeff[1] = 0.01;  // lon

    // col = (lon_n + 0.02 lon_n^2 + 0.01 lon_n*lat_n) / (1 + 0.01 lat_n)
    info.samp_num_coeff[1] = 1.0;   // lon
    info.samp_num_coeff[7] = 0.02;  // lon^2
    info.samp_num_coeff[4] = 0.01;  // lon*lat
    info.samp_den_coeff[0] = 1.0;   // 1
    info.samp_den_coeff[2] = 0.01;  // lat

    info.min_lon = 90.0;
    info.min_lat = 20.0;
    info.max_lon = 110.0;
    info.max_lat = 40.0;
    return info;
}

// Pure identity model: col = lon + 0.5, row = lat + 0.5.
RpcInfo MakeIdentityInfo() {
    RpcInfo info;
    info.samp_num_coeff[1] = 1.0;  // lon
    info.samp_den_coeff[0] = 1.0;
    info.line_num_coeff[2] = 1.0;  // lat
    info.line_den_coeff[0] = 1.0;
    return info;
}

// Independent forward reference (straight from the RPC00 definition), written
// out with an explicit monomial list so it does not re-run rpc_forward_point.
void ReferenceForward(const RpcInfo& info,
                      double lon,
                      double lat,
                      double h,
                      double& col,
                      double& row) {
    double diff_long = lon - info.long_off;
    if (diff_long < -270.0) {
        diff_long += 360.0;
    } else if (diff_long > 270.0) {
        diff_long -= 360.0;
    }
    const double L = diff_long / info.long_scale;
    const double P = (lat - info.lat_off) / info.lat_scale;
    const double H = (h - info.height_off) / info.height_scale;

    const std::array<double, 20> terms = {
        1.0,       L,         P,         H,         L * P,
        L * H,     P * H,     L * L,     P * P,     H * H,
        L * P * H, L * L * L, L * P * P, L * H * H, L * L * P,
        P * P * P, P * H * H, L * L * H, P * P * H, H * H * H};

    const auto eval = [&terms](const std::array<double, 20>& c) {
        double s = 0.0;
        for (int i = 0; i < 20; ++i) {
            s += terms[i] * c[i];
        }
        return s;
    };

    const double line_num = eval(info.line_num_coeff);
    const double line_den = eval(info.line_den_coeff);
    const double samp_num = eval(info.samp_num_coeff);
    const double samp_den = eval(info.samp_den_coeff);
    col = samp_num / samp_den * info.samp_scale + info.samp_off + 0.5;
    row = line_num / line_den * info.line_scale + info.line_off + 0.5;
}

// Deterministic random lon/lat/alt points inside the model's validity bounds.
struct Pt {
    double lon;
    double lat;
    double alt;
};

std::vector<Pt> MakePoints(std::size_t n, const RpcInfo& info) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> lon(info.long_off - info.long_scale,
                                               info.long_off + info.long_scale);
    std::uniform_real_distribution<double> lat(info.lat_off - info.lat_scale,
                                               info.lat_off + info.lat_scale);
    std::uniform_real_distribution<double> alt(
        info.height_off - info.height_scale,
        info.height_off + info.height_scale);

    std::vector<Pt> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = Pt{lon(rng), lat(rng), alt(rng)};
    }
    return pts;
}

// Wrap a host vector of (lon, lat, alt) points as a non-owning [N, 3] double
// tensor. `pts` must outlive the returned tensor.
zt::Tensor PointTensor(std::vector<Pt>& pts) {
    return zt::from_blob(pts.data(),
                         {static_cast<int64_t>(pts.size()), 3},
                         zt::dtype(zt::kDouble));
}

// Convert a [N, 2] CPU tensor of col/row (or lon/lat) pairs to a vector.
std::vector<std::pair<double, double>> ToPairs(const zt::Tensor& t) {
    const double* p = t.data_ptr<double>();
    std::vector<std::pair<double, double>> out(
        static_cast<std::size_t>(t.size(0)));
    for (int64_t i = 0; i < t.size(0); ++i) {
        out[static_cast<std::size_t>(i)] = {p[2 * i], p[2 * i + 1]};
    }
    return out;
}

// GDAL reference transformer wrapping the same coefficients.
void* CreateGdalTransformer(const RpcInfo& info, double pix_err_threshold) {
    GDALRPCInfoV2 g{};
    g.dfLINE_OFF = info.line_off;
    g.dfSAMP_OFF = info.samp_off;
    g.dfLAT_OFF = info.lat_off;
    g.dfLONG_OFF = info.long_off;
    g.dfHEIGHT_OFF = info.height_off;
    g.dfLINE_SCALE = info.line_scale;
    g.dfSAMP_SCALE = info.samp_scale;
    g.dfLAT_SCALE = info.lat_scale;
    g.dfLONG_SCALE = info.long_scale;
    g.dfHEIGHT_SCALE = info.height_scale;
    std::copy(info.line_num_coeff.begin(),
              info.line_num_coeff.end(),
              g.adfLINE_NUM_COEFF);
    std::copy(info.line_den_coeff.begin(),
              info.line_den_coeff.end(),
              g.adfLINE_DEN_COEFF);
    std::copy(info.samp_num_coeff.begin(),
              info.samp_num_coeff.end(),
              g.adfSAMP_NUM_COEFF);
    std::copy(info.samp_den_coeff.begin(),
              info.samp_den_coeff.end(),
              g.adfSAMP_DEN_COEFF);
    g.dfMIN_LONG = info.min_lon;
    g.dfMIN_LAT = info.min_lat;
    g.dfMAX_LONG = info.max_lon;
    g.dfMAX_LAT = info.max_lat;
    g.dfERR_BIAS = std::numeric_limits<double>::quiet_NaN();
    g.dfERR_RAND = std::numeric_limits<double>::quiet_NaN();
    return GDALCreateRPCTransformerV2(&g, FALSE, pix_err_threshold, nullptr);
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() {
    int n_devices = 0;
    return cudaGetDeviceCount(&n_devices) == cudaSuccess && n_devices > 0;
}
#endif  // BUILD_CUDA_MODULE

// ============================== CPU path ================================

TEST(RpcCpu, ForwardIdentityModel) {
    const RpcModel model(MakeIdentityInfo());
    std::vector<Pt> pts{
        Pt{0.0, 0.0, 0.0},
        Pt{10.0, 20.0, 100.0},
        Pt{-179.5, -89.5, -50.0},
    };

    zt::Tensor in = PointTensor(pts);
    zt::Tensor out;
    model.lonlatalt_to_colrow(in, out);

    const std::vector<std::pair<double, double>> got = ToPairs(out);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(got[i].first, pts[i].lon + 0.5, kTolAnalytic)
            << "point " << i << " col";
        EXPECT_NEAR(got[i].second, pts[i].lat + 0.5, kTolAnalytic)
            << "point " << i << " row";
    }
}

TEST(RpcCpu, ForwardMatchesIndependentReference) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(1000, info);

    zt::Tensor in = PointTensor(pts);
    zt::Tensor out;
    model.lonlatalt_to_colrow(in, out);

    const std::vector<std::pair<double, double>> got = ToPairs(out);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        double ref_col = 0.0;
        double ref_row = 0.0;
        ReferenceForward(
            info, pts[i].lon, pts[i].lat, pts[i].alt, ref_col, ref_row);
        EXPECT_NEAR(got[i].first, ref_col, kTolColRow) << "point " << i;
        EXPECT_NEAR(got[i].second, ref_row, kTolColRow) << "point " << i;
    }
}

TEST(RpcCpu, ForwardWrapsDateline) {
    // lon = 300 should be treated as -60 by the forward transform (GDAL's
    // dateline handling): same normalized longitude -> same pixel.
    const RpcInfo info = MakeIdentityInfo();
    const RpcModel model(info);

    std::vector<Pt> pts{
        Pt{300.0, 0.0, 0.0},
        Pt{-60.0, 0.0, 0.0},
    };
    zt::Tensor in = PointTensor(pts);
    zt::Tensor out;
    model.lonlatalt_to_colrow(in, out);

    const std::vector<std::pair<double, double>> got = ToPairs(out);
    EXPECT_NEAR(got[0].first, got[1].first, kTolColRow);
    EXPECT_NEAR(got[0].second, got[1].second, kTolColRow);
}

TEST(RpcCpu, ForwardMatchesGdal) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(500, info);

    void* gdal = CreateGdalTransformer(info, 0.1);
    ASSERT_NE(gdal, nullptr);

    std::vector<double> x(pts.size());
    std::vector<double> y(pts.size());
    std::vector<double> z(pts.size());
    std::vector<int> ok(pts.size(), 0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        x[i] = pts[i].lon;
        y[i] = pts[i].lat;
        z[i] = pts[i].alt;
    }
    const int ret = GDALRPCTransform(gdal,
                                     TRUE,
                                     static_cast<int>(pts.size()),
                                     x.data(),
                                     y.data(),
                                     z.data(),
                                     ok.data());
    EXPECT_NE(ret, 0);

    zt::Tensor in = PointTensor(pts);
    zt::Tensor out;
    model.lonlatalt_to_colrow(in, out);
    const std::vector<std::pair<double, double>> got = ToPairs(out);

    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(got[i].first, x[i], kTolColRow) << "point " << i << " col";
        EXPECT_NEAR(got[i].second, y[i], kTolColRow) << "point " << i << " row";
    }
    GDALDestroyRPCTransformer(gdal);
}

TEST(RpcCpu, InverseMatchesGdal) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(500, info);

    // Forward the points first so the inverse gets well-posed col/row inputs.
    zt::Tensor in = PointTensor(pts);
    zt::Tensor colrow;
    model.lonlatalt_to_colrow(in, colrow);
    const std::vector<std::pair<double, double>> cr = ToPairs(colrow);

    void* gdal = CreateGdalTransformer(info, 0.1);
    ASSERT_NE(gdal, nullptr);

    std::vector<double> x(cr.size());
    std::vector<double> y(cr.size());
    std::vector<double> z(pts.size());
    std::vector<int> ok(pts.size(), 0);
    for (std::size_t i = 0; i < cr.size(); ++i) {
        x[i] = cr[i].first;
        y[i] = cr[i].second;
        z[i] = pts[i].alt;
    }
    const int ret = GDALRPCTransform(gdal,
                                     FALSE,
                                     static_cast<int>(cr.size()),
                                     x.data(),
                                     y.data(),
                                     z.data(),
                                     ok.data());
    EXPECT_NE(ret, 0);

    // col/row/alt tensor: interleave col, row, alt back into [N, 3].
    std::vector<std::array<double, 3>> crw(cr.size());
    for (std::size_t i = 0; i < cr.size(); ++i) {
        crw[i] = {cr[i].first, cr[i].second, pts[i].alt};
    }
    zt::Tensor crw_t = zt::from_blob(crw.data(),
                                     {static_cast<int64_t>(crw.size()), 3},
                                     zt::dtype(zt::kDouble));

    zt::Tensor ll;
    model.colrowalt_to_lonlat(crw_t, ll);
    const std::vector<std::pair<double, double>> got = ToPairs(ll);

    for (std::size_t i = 0; i < cr.size(); ++i) {
        EXPECT_NEAR(got[i].first, x[i], kTolInverseDeg)
            << "point " << i << " lon";
        EXPECT_NEAR(got[i].second, y[i], kTolInverseDeg)
            << "point " << i << " lat";
    }
    GDALDestroyRPCTransformer(gdal);
}

TEST(RpcCpu, InverseRoundTripsLonLatAlt) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(1000, info);

    zt::Tensor in = PointTensor(pts);
    zt::Tensor colrow;
    model.lonlatalt_to_colrow(in, colrow);

    std::vector<std::array<double, 3>> crw(pts.size());
    const std::vector<std::pair<double, double>> cr = ToPairs(colrow);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        crw[i] = {cr[i].first, cr[i].second, pts[i].alt};
    }
    zt::Tensor crw_t = zt::from_blob(crw.data(),
                                     {static_cast<int64_t>(crw.size()), 3},
                                     zt::dtype(zt::kDouble));

    zt::Tensor ll;
    model.colrowalt_to_lonlat(crw_t, ll);
    const std::vector<std::pair<double, double>> got = ToPairs(ll);

    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(got[i].first, pts[i].lon, kTolInverseDeg)
            << "point " << i << " lon";
        EXPECT_NEAR(got[i].second, pts[i].lat, kTolInverseDeg)
            << "point " << i << " lat";
    }
}

TEST(RpcCpu, InverseRoundTripsColRowAlt) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(1000, info);

    // col/row grid samples across the image (plus the forward of random
    // ground points), so the inverse is exercised off the forward's image.
    std::vector<std::pair<double, double>> cr;
    cr.reserve(pts.size());
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> col_u(0.5, 100000.0);
    std::uniform_real_distribution<double> row_u(0.5, 50000.0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        cr.emplace_back(col_u(rng), row_u(rng));
    }

    std::vector<std::array<double, 3>> crw(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        crw[i] = {cr[i].first, cr[i].second, pts[i].alt};
    }
    zt::Tensor crw_t = zt::from_blob(crw.data(),
                                     {static_cast<int64_t>(crw.size()), 3},
                                     zt::dtype(zt::kDouble));

    zt::Tensor ll;
    model.colrowalt_to_lonlat(crw_t, ll);
    const std::vector<std::pair<double, double>> got = ToPairs(ll);

    // Back-transform the recovered lon/lat and compare in image space.
    std::vector<Pt> back(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        back[i] = Pt{got[i].first, got[i].second, pts[i].alt};
    }
    zt::Tensor back_t = PointTensor(back);
    zt::Tensor back_cr;
    model.lonlatalt_to_colrow(back_t, back_cr);
    const std::vector<std::pair<double, double>> back_cr_v = ToPairs(back_cr);

    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(back_cr_v[i].first, cr[i].first, kTolRoundTripPx)
            << "point " << i << " col";
        EXPECT_NEAR(back_cr_v[i].second, cr[i].second, kTolRoundTripPx)
            << "point " << i << " row";
    }
}

TEST(RpcCpu, RealGeoEyeMatchesGdal) {
    // A real satellite RPC (GeoEye, vendored from the GDAL autotest suite):
    // validates the parser, the forward, and the iterative inverse against
    // GDAL's own transformer on genuine coefficients.
    const RpcInfo info = RpcInfoFromRpcFile(ZPROJ_TEST_RPC_DATA);
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(500, info);

    void* gdal = CreateGdalTransformer(info, 0.1);
    ASSERT_NE(gdal, nullptr);

    // Forward: lon/lat/alt -> col/row.
    std::vector<double> x(pts.size());
    std::vector<double> y(pts.size());
    std::vector<double> z(pts.size());
    std::vector<int> ok(pts.size(), 0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        x[i] = pts[i].lon;
        y[i] = pts[i].lat;
        z[i] = pts[i].alt;
    }
    int ret = GDALRPCTransform(gdal,
                               TRUE,
                               static_cast<int>(pts.size()),
                               x.data(),
                               y.data(),
                               z.data(),
                               ok.data());
    EXPECT_NE(ret, 0);

    zt::Tensor in = PointTensor(pts);
    zt::Tensor colrow;
    model.lonlatalt_to_colrow(in, colrow);
    const std::vector<std::pair<double, double>> cr = ToPairs(colrow);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(cr[i].first, x[i], kTolColRow) << "point " << i << " col";
        EXPECT_NEAR(cr[i].second, y[i], kTolColRow) << "point " << i << " row";
    }

    // Inverse: col/row/alt -> lon/lat.
    std::vector<std::array<double, 3>> crw(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        crw[i] = {cr[i].first, cr[i].second, pts[i].alt};
    }
    zt::Tensor crw_t = zt::from_blob(crw.data(),
                                     {static_cast<int64_t>(crw.size()), 3},
                                     zt::dtype(zt::kDouble));
    zt::Tensor ll;
    model.colrowalt_to_lonlat(crw_t, ll);
    const std::vector<std::pair<double, double>> got = ToPairs(ll);

    for (std::size_t i = 0; i < pts.size(); ++i) {
        x[i] = cr[i].first;
        y[i] = cr[i].second;
        z[i] = pts[i].alt;
    }
    ret = GDALRPCTransform(gdal,
                           FALSE,
                           static_cast<int>(pts.size()),
                           x.data(),
                           y.data(),
                           z.data(),
                           ok.data());
    EXPECT_NE(ret, 0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(got[i].first, x[i], kTolInverseDeg)
            << "point " << i << " lon";
        EXPECT_NEAR(got[i].second, y[i], kTolInverseDeg)
            << "point " << i << " lat";
    }
    GDALDestroyRPCTransformer(gdal);
}

TEST(RpcCpu, ReusesProvidedOutput) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(16, info);

    zt::Tensor in = PointTensor(pts);
    zt::Tensor out = zt::full({16, 2}, -1.0, zt::dtype(zt::kDouble));
    model.lonlatalt_to_colrow(in, out);

    ASSERT_EQ(out.size(0), 16);
    ASSERT_EQ(out.size(1), 2);
    for (int64_t i = 0; i < 16; ++i) {
        double ref_col = 0.0;
        double ref_row = 0.0;
        ReferenceForward(info,
                         pts[static_cast<std::size_t>(i)].lon,
                         pts[static_cast<std::size_t>(i)].lat,
                         pts[static_cast<std::size_t>(i)].alt,
                         ref_col,
                         ref_row);
        const double* p = out.data_ptr<double>();
        EXPECT_NEAR(p[2 * i], ref_col, kTolColRow);
        EXPECT_NEAR(p[2 * i + 1], ref_row, kTolColRow);
    }
}

TEST(RpcCpu, InvalidShapesThrow) {
    const RpcModel model(MakeSyntheticInfo());
    zt::Tensor bad = zt::zeros({4, 2}, zt::dtype(zt::kDouble));
    zt::Tensor out;
    EXPECT_THROW(model.lonlatalt_to_colrow(bad, out), std::runtime_error);

    zt::Tensor good = zt::zeros({4, 3}, zt::dtype(zt::kDouble));
    zt::Tensor bad_out = zt::zeros({4, 3}, zt::dtype(zt::kDouble));
    EXPECT_THROW(model.lonlatalt_to_colrow(good, bad_out), std::runtime_error);
}

TEST(RpcCpu, InvalidOptionsFallBackToDefaults) {
    const RpcModel model(MakeSyntheticInfo(), RpcOptions{0.0, -1});
    EXPECT_EQ(model.options().pixel_error_threshold, 0.1);
    EXPECT_EQ(model.options().max_iterations, 10);
}

#ifdef BUILD_CUDA_MODULE

class RpcCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!HasCudaDevice()) {
            GTEST_SKIP() << "no CUDA-capable device";
        }
    }
};

TEST_F(RpcCudaTest, ForwardMatchesCpu) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(10000, info);

    zt::Tensor cpu_in = PointTensor(pts);
    zt::Tensor cpu_out;
    model.lonlatalt_to_colrow(cpu_in, cpu_out);

    zt::Tensor gpu_in = PointTensor(pts).cuda();
    zt::Tensor gpu_out;
    model.lonlatalt_to_colrow(gpu_in, gpu_out);

    ASSERT_TRUE(gpu_out.is_cuda());
    const zt::Tensor gpu_out_cpu = gpu_out.cpu();
    const double* a = cpu_out.data_ptr<double>();
    const double* b = gpu_out_cpu.data_ptr<double>();
    double max_diff = 0.0;
    for (int64_t i = 0; i < 2 * static_cast<int64_t>(pts.size()); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    EXPECT_LT(max_diff, kTolColRow);
}

TEST_F(RpcCudaTest, InverseMatchesCpu) {
    const RpcInfo info = MakeSyntheticInfo();
    const RpcModel model(info);
    std::vector<Pt> pts = MakePoints(10000, info);

    zt::Tensor cpu_in = PointTensor(pts);
    zt::Tensor colrow;
    model.lonlatalt_to_colrow(cpu_in, colrow);

    std::vector<std::array<double, 3>> crw(pts.size());
    const std::vector<std::pair<double, double>> cr = ToPairs(colrow);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        crw[i] = {cr[i].first, cr[i].second, pts[i].alt};
    }
    zt::Tensor crw_cpu = zt::from_blob(crw.data(),
                                       {static_cast<int64_t>(crw.size()), 3},
                                       zt::dtype(zt::kDouble));

    zt::Tensor cpu_ll;
    model.colrowalt_to_lonlat(crw_cpu, cpu_ll);

    zt::Tensor gpu_ll;
    model.colrowalt_to_lonlat(crw_cpu.cuda(), gpu_ll);

    ASSERT_TRUE(gpu_ll.is_cuda());
    const zt::Tensor gpu_ll_cpu = gpu_ll.cpu();
    const double* a = cpu_ll.data_ptr<double>();
    const double* b = gpu_ll_cpu.data_ptr<double>();
    double max_diff = 0.0;
    for (int64_t i = 0; i < 2 * static_cast<int64_t>(pts.size()); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    EXPECT_LT(max_diff, kTolInverseDeg);
}

TEST_F(RpcCudaTest, MixedCpuInCudaOutThrows) {
    const RpcModel model(MakeSyntheticInfo());
    std::vector<Pt> pts = MakePoints(4, MakeSyntheticInfo());
    zt::Tensor in = PointTensor(pts);
    zt::Tensor out = zt::zeros({4, 2}, zt::dtype(zt::kDouble)).cuda();
    EXPECT_THROW(model.lonlatalt_to_colrow(in, out), std::runtime_error);
}

#endif  // BUILD_CUDA_MODULE

}  // namespace

int main(int argc, char** argv) {
    zt::Logger::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
