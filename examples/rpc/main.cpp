// RPC sensor-model speed test: zproj CPU/CUDA (ztensor-backed) vs the
// GDAL GDALRPCTransformer reference, both forward (lon/lat/alt -> col/row)
// and iterative inverse (col/row/alt -> lon/lat).
//
// The model coefficients come from a real satellite RPC file (md_eros.rpc,
// vendored from the GDAL autotest suite, MIT-licensed) so the numbers are
// representative of actual rational-polynomial sensor models -- including
// denominators that vary by up to ~30% across the scene.
//
// Usage:
//   ./build/default/bin/rpc [points] [reps]
//   e.g. ./build/default/bin/rpc 1000000 5
//
// No DEM is used anywhere: every point carries its own height, which is what
// both the forward and the inverse consume directly.

#include <gdal_alg.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

using zproj::rpc::RpcInfo;
using zproj::rpc::RpcInfoFromRpcFile;
using zproj::rpc::RpcModel;

// ---------------------------------------------------------------------------
// GDAL reference transformer.
// ---------------------------------------------------------------------------
void* CreateGdalTransformer(const RpcInfo& info) {
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
    return GDALCreateRPCTransformerV2(&g, FALSE, 0.1, nullptr);
}

struct Pt {
    double lon;
    double lat;
    double alt;
};

// Deterministic pseudo-random points inside the RPC's normalization window
// (offsets +/- scales), i.e. where the model is meant to be evaluated.
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

// Average wall time in ms of `reps` invocations, after one warmup call.
template<typename Fn>
double BenchMs(int reps, Fn&& fn) {
    fn();  // warmup
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

void PrintRow(const char* name, std::size_t n, double ms) {
    const double mpts = static_cast<double>(n) / ms / 1e3;
    std::cout << "  " << std::left << std::setw(28) << name << std::right
              << std::setw(12) << std::fixed << std::setprecision(3) << ms
              << std::setw(12) << std::setprecision(3) << mpts << '\n';
}

struct MaxError {
    double first = 0.0;
    double second = 0.0;
};

template<typename T>
MaxError MaxDiff(const std::vector<T>& a, const std::vector<T>& b) {
    MaxError e;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e.first = std::max(e.first, std::abs(a[i].first - b[i].first));
        e.second = std::max(e.second, std::abs(a[i].second - b[i].second));
    }
    return e;
}

// A [N, 3] double tensor over (lon, lat, alt) triples.
zt::Tensor PointTensor(std::vector<Pt>& pts) {
    return zt::from_blob(pts.data(),
                         {static_cast<int64_t>(pts.size()), 3},
                         zt::dtype(zt::kDouble));
}

std::vector<std::pair<double, double>> ToPairs(const zt::Tensor& t) {
    const double* p = t.data_ptr<double>();
    std::vector<std::pair<double, double>> out(
        static_cast<std::size_t>(t.size(0)));
    for (int64_t i = 0; i < t.size(0); ++i) {
        out[static_cast<std::size_t>(i)] = {p[2 * i], p[2 * i + 1]};
    }
    return out;
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() {
    int n_devices = 0;
    return cudaGetDeviceCount(&n_devices) == cudaSuccess && n_devices > 0;
}
#endif  // BUILD_CUDA_MODULE

}  // namespace

int main(int argc, char** argv) {
    std::size_t n = 100'000;
    int reps = 5;
    if (argc > 1) {
        try {
            n = std::stoull(argv[1]);
        } catch (const std::exception&) {
            std::cerr << "invalid points count: " << argv[1] << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 2) {
        try {
            reps = std::stoi(argv[2]);
        } catch (const std::exception&) {
            std::cerr << "invalid reps count: " << argv[2] << '\n';
            return EXIT_FAILURE;
        }
    }
    if (n == 0 || reps <= 0) {
        std::cerr << "usage: rpc [points>0] [reps>0]\n";
        return EXIT_FAILURE;
    }

    const RpcInfo info = RpcInfoFromRpcFile(ZPROJ_RPC_DATA);
    const RpcModel model(info);

    std::cout << "RPC sensor-model speed test (no DEM; heights supplied)\n";
    std::cout << "  RPC file           : " << ZPROJ_RPC_DATA << '\n';
    std::cout << "  lon_off/scale      : " << info.long_off << " / "
              << info.long_scale << " deg\n";
    std::cout << "  lat_off/scale      : " << info.lat_off << " / "
              << info.lat_scale << " deg\n";
    std::cout << "  height_off/scale   : " << info.height_off << " / "
              << info.height_scale << " m\n";
    std::cout << "  points per batch   : " << n << '\n';
    std::cout << "  CPU/CUDA reps      : " << reps << '\n';
    std::cout << "  GDAL reference     : 1 pass\n";
    std::cout << '\n';

    const std::vector<Pt> pts = MakePoints(n, info);
    const zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));

    void* gdal = CreateGdalTransformer(info);
    if (gdal == nullptr) {
        std::cerr << "Failed to build GDAL RPC transformer\n";
        return EXIT_FAILURE;
    }

    // ---- forward: lon/lat/alt -> col/row ----
    zt::Tensor cpu_out;
    const double cpu_ms =
        BenchMs(reps, [&] { model.lonlatalt_to_colrow(in, cpu_out); });

    const auto f0 = std::chrono::steady_clock::now();
    std::vector<double> gx(pts.size());
    std::vector<double> gy(pts.size());
    std::vector<double> gz(pts.size());
    std::vector<int> gok(pts.size(), 0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        gx[i] = pts[i].lon;
        gy[i] = pts[i].lat;
        gz[i] = pts[i].alt;
    }
    if (GDALRPCTransform(gdal,
                         TRUE,
                         static_cast<int>(pts.size()),
                         gx.data(),
                         gy.data(),
                         gz.data(),
                         gok.data()) == 0) {
        std::cerr << "GDAL forward RPC transform failed\n";
        return EXIT_FAILURE;
    }
    const auto f1 = std::chrono::steady_clock::now();
    const double gdal_fwd_ms =
        std::chrono::duration<double, std::milli>(f1 - f0).count();

    std::cout << "  lon/lat/alt -> col/row (forward)\n";
    std::cout << "  " << std::left << std::setw(28) << "backend" << std::right
              << std::setw(12) << "ms/batch" << std::setw(12) << "Mpts/s"
              << '\n';
    PrintRow("zproj CPU", n, cpu_ms);
    PrintRow("GDAL (CPU ref)", n, gdal_fwd_ms);

    const std::vector<std::pair<double, double>> cpu_cr = ToPairs(cpu_out);
    std::vector<std::pair<double, double>> gdal_cr(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        gdal_cr[i] = {gx[i], gy[i]};
    }
    const MaxError fwd_err = MaxDiff(cpu_cr, gdal_cr);
    std::cout << "\nmax |zproj CPU - GDAL| (forward) = col " << std::scientific
              << std::setprecision(3) << fwd_err.first << " px, row "
              << fwd_err.second << " px\n"
              << std::defaultfloat;

    // ---- inverse: col/row/alt -> lon/lat ----
    std::vector<std::array<double, 3>> crw(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        crw[i] = {cpu_cr[i].first, cpu_cr[i].second, pts[i].alt};
    }
    const zt::Tensor crw_t =
        zt::from_blob(crw.data(),
                      {static_cast<int64_t>(crw.size()), 3},
                      zt::dtype(zt::kDouble));

    zt::Tensor cpu_ll;
    const double inv_cpu_ms =
        BenchMs(reps, [&] { model.colrowalt_to_lonlat(crw_t, cpu_ll); });

    for (std::size_t i = 0; i < pts.size(); ++i) {
        gx[i] = cpu_cr[i].first;
        gy[i] = cpu_cr[i].second;
        gz[i] = pts[i].alt;
    }
    const auto i0 = std::chrono::steady_clock::now();
    if (GDALRPCTransform(gdal,
                         FALSE,
                         static_cast<int>(pts.size()),
                         gx.data(),
                         gy.data(),
                         gz.data(),
                         gok.data()) == 0) {
        std::cerr << "GDAL inverse RPC transform failed\n";
        return EXIT_FAILURE;
    }
    const auto i1 = std::chrono::steady_clock::now();
    const double gdal_inv_ms =
        std::chrono::duration<double, std::milli>(i1 - i0).count();

    std::cout << "\n  col/row/alt -> lon/lat (inverse)\n";
    std::cout << "  " << std::left << std::setw(28) << "backend" << std::right
              << std::setw(12) << "ms/batch" << std::setw(12) << "Mpts/s"
              << '\n';
    PrintRow("zproj CPU", n, inv_cpu_ms);
    PrintRow("GDAL (CPU ref)", n, gdal_inv_ms);

    const std::vector<std::pair<double, double>> cpu_ll_v = ToPairs(cpu_ll);
    std::vector<std::pair<double, double>> gdal_ll_v(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
        gdal_ll_v[i] = {gx[i], gy[i]};
    }
    const MaxError inv_err = MaxDiff(cpu_ll_v, gdal_ll_v);
    std::cout << "\nmax |zproj CPU - GDAL| (inverse) = lon " << std::scientific
              << std::setprecision(3) << inv_err.first << " deg, lat "
              << inv_err.second << " deg\n"
              << std::defaultfloat;

#ifdef BUILD_CUDA_MODULE
    if (HasCudaDevice()) {
        const zt::Tensor gpu_in = in.cuda();
        zt::Tensor gpu_out;

        const double cuda_call_ms = BenchMs(reps, [&] {
            model.lonlatalt_to_colrow(gpu_in, gpu_out);
            cudaDeviceSynchronize();
        });
        for (int i = 0; i < reps; ++i) {
            model.lonlatalt_to_colrow(gpu_in, gpu_out);
        }
        cudaDeviceSynchronize();
        const auto s0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            model.lonlatalt_to_colrow(gpu_in, gpu_out);
        }
        cudaDeviceSynchronize();
        const auto s1 = std::chrono::steady_clock::now();
        const double cuda_sustained_ms =
            std::chrono::duration<double, std::milli>(s1 - s0).count() / reps;

        std::cout << "\n  CUDA forward:\n";
        PrintRow("zproj CUDA (per-call sync)", n, cuda_call_ms);
        PrintRow("zproj CUDA (sustained)", n, cuda_sustained_ms);

        const std::vector<std::pair<double, double>> gpu_cr =
            ToPairs(gpu_out.cpu());
        const MaxError gpu_fwd_err = MaxDiff(gpu_cr, gdal_cr);
        std::cout << "max |zproj CUDA - GDAL| (forward) = col "
                  << std::scientific << std::setprecision(3)
                  << gpu_fwd_err.first << " px, row " << gpu_fwd_err.second
                  << " px\n"
                  << std::defaultfloat;

        const zt::Tensor gpu_crw = crw_t.cuda();
        zt::Tensor gpu_ll;
        const double inv_cuda_call_ms = BenchMs(reps, [&] {
            model.colrowalt_to_lonlat(gpu_crw, gpu_ll);
            cudaDeviceSynchronize();
        });
        for (int i = 0; i < reps; ++i) {
            model.colrowalt_to_lonlat(gpu_crw, gpu_ll);
        }
        cudaDeviceSynchronize();
        const auto si0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            model.colrowalt_to_lonlat(gpu_crw, gpu_ll);
        }
        cudaDeviceSynchronize();
        const auto si1 = std::chrono::steady_clock::now();
        const double inv_cuda_sustained_ms =
            std::chrono::duration<double, std::milli>(si1 - si0).count() / reps;

        std::cout << "\n  CUDA inverse:\n";
        PrintRow("zproj CUDA (per-call sync)", n, inv_cuda_call_ms);
        PrintRow("zproj CUDA (sustained)", n, inv_cuda_sustained_ms);

        const std::vector<std::pair<double, double>> gpu_ll_v =
            ToPairs(gpu_ll.cpu());
        const MaxError gpu_inv_err = MaxDiff(gpu_ll_v, gdal_ll_v);
        std::cout << "max |zproj CUDA - GDAL| (inverse) = lon "
                  << std::scientific << std::setprecision(3)
                  << gpu_inv_err.first << " deg, lat " << gpu_inv_err.second
                  << " deg\n"
                  << std::defaultfloat;
    } else {
        std::cout << "\n  (no CUDA device; skipping GPU section)\n";
    }
#else
    std::cout << "\n  (built without the CUDA backend)\n";
#endif  // BUILD_CUDA_MODULE

    GDALDestroyRPCTransformer(gdal);
    return EXIT_SUCCESS;
}
