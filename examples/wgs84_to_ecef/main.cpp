// WGS84 geodetic -> ECEF speed test.
//
// Times the ztensor-backed zproj::crs::wgs84_to_ecef on a batch of points for
// both backends and reports throughput:
//   1. zproj CPU (ztensor, serial host loop)
//   2. zproj CUDA (ztensor, device kernel) -- when a GPU is present
//   3. GDAL/PROJ reference (EPSG:4326 -> EPSG:4978), on CPU
//
// The timed call is the plain public API with a pre-allocated output, so the
// measured cost is the transform itself: no per-call allocation/clone and no
// H2D/D2H transfer inside the timed region. The CUDA iteration is
// synchronized with the device so the reported time includes kernel
// execution. After timing, outputs are cross-checked against the GDAL
// reference (max |delta| in metres).
//
// Usage:
//   ./build/default/bin/wgs84_to_ecef [points] [reps]
//   e.g. ./build/default/bin/wgs84_to_ecef 1000000 5

#include <ogr_spatialref.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#ifdef BUILD_CUDA_MODULE
#include <cuda_runtime.h>
#endif  // BUILD_CUDA_MODULE

#include "zproj/crs/wgs84.hpp"
#include "zproj/crs/wgs84_to_ecef.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"

namespace {

using zproj::crs::Ecef;
using zproj::crs::Geodetic;
using zproj::crs::wgs84_to_ecef;

constexpr double kDeg2Rad = std::numbers::pi / 180.0;

// Deterministic pseudo-random points with a fixed seed.
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

// CPU reference via GDAL/PROJ: EPSG:4326 (lon/lat, degrees) -> EPSG:4978
// (ECEF).
std::vector<Ecef> Wgs84ToEcefGdal(const std::vector<Geodetic>& geo) {
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

double MaxError(const std::vector<Ecef>& a, const std::vector<Ecef>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Eigen::Vector3d va(a[i].x, a[i].y, a[i].z);
        const Eigen::Vector3d vb(b[i].x, b[i].y, b[i].z);
        worst = std::max(worst, (va - vb).norm());
    }
    return worst;
}

// A CPU [N, 3] double tensor laid out as consecutive Ecef triples.
std::vector<Ecef> ToEcefVector(const zt::Tensor& t) {
    const auto* p = t.data_ptr<Ecef>();
    const std::size_t n = static_cast<std::size_t>(t.size(0));
    return std::vector<Ecef>(p, p + n);
}

void PrintRow(const char* name, std::size_t n, double ms) {
    const double mpts = static_cast<double>(n) / ms / 1e3;
    std::cout << "  " << std::left << std::setw(22) << name << std::right
              << std::setw(12) << std::fixed << std::setprecision(3) << ms
              << std::setw(12) << std::setprecision(3) << mpts << '\n';
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() {
    int n_devices = 0;
    return cudaGetDeviceCount(&n_devices) == cudaSuccess && n_devices > 0;
}

std::string CudaDeviceName() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return "unknown";
    }
    return prop.name;
}
#endif  // BUILD_CUDA_MODULE

}  // namespace

int main(int argc, char** argv) {
    std::size_t n = 1'000'000;
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
        std::cerr << "usage: wgs84_to_ecef [points>0] [reps>0]\n";
        return EXIT_FAILURE;
    }

    std::cout << "WGS84 geodetic -> ECEF speed test\n";
    std::cout << "  points per batch : " << n << '\n';
    std::cout << "  CPU/CUDA reps    : " << reps << '\n';
    std::cout << "  GDAL reference   : 1 pass\n\n";

    const std::vector<Geodetic> pts = MakePoints(n);
    const zt::Tensor in = zt::from_blob(const_cast<Geodetic*>(pts.data()),
                                        {static_cast<int64_t>(n), 3},
                                        zt::dtype(zt::kDouble));

    // ---- zproj CPU ----
    zt::Tensor cpu_out = in.clone();
    const double cpu_ms = BenchMs(reps, [&] { wgs84_to_ecef(in, cpu_out); });

    // ---- GDAL/PROJ reference (also used for the cross-checks below) ----
    const auto g0 = std::chrono::steady_clock::now();
    const std::vector<Ecef> gdal_ref = Wgs84ToEcefGdal(pts);
    const auto g1 = std::chrono::steady_clock::now();
    const double gdal_ms =
        std::chrono::duration<double, std::milli>(g1 - g0).count();

    // ---- report ----
    std::cout << "  " << std::left << std::setw(22) << "backend" << std::right
              << std::setw(12) << "ms/batch" << std::setw(12) << "Mpts/s"
              << '\n';
    PrintRow("zproj CPU (ztensor)", n, cpu_ms);
    PrintRow("GDAL/PROJ (CPU ref)", n, gdal_ms);

    const std::vector<Ecef> cpu_ecef = ToEcefVector(cpu_out);
    std::cout << "\nmax |zproj CPU - GDAL| = " << std::fixed
              << std::setprecision(6) << MaxError(cpu_ecef, gdal_ref) << " m\n";

#ifdef BUILD_CUDA_MODULE
    if (HasCudaDevice()) {
        const zt::Tensor gpu_in = in.cuda();
        zt::Tensor gpu_out = gpu_in.clone();
        const double cuda_ms = BenchMs(reps, [&] {
            wgs84_to_ecef(gpu_in, gpu_out);
            cudaDeviceSynchronize();
        });

        std::cout << "\n  CUDA device: " << CudaDeviceName() << '\n';
        PrintRow("zproj CUDA (ztensor)", n, cuda_ms);
        std::cout << "  speedup vs zproj CPU : " << std::setprecision(2)
                  << (cpu_ms / cuda_ms) << "x\n";

        const std::vector<Ecef> gpu_ecef = ToEcefVector(gpu_out.cpu());
        std::cout << "\nmax |zproj CUDA - GDAL| = " << std::setprecision(6)
                  << MaxError(gpu_ecef, gdal_ref) << " m\n";
        std::cout << "max |zproj CUDA - CPU | = "
                  << MaxError(gpu_ecef, cpu_ecef) << " m\n";
    } else {
        std::cout << "\nno CUDA-capable device found; CUDA timing skipped\n";
    }
#else
    std::cout << "\nztensor built without CUDA; CUDA timing skipped\n";
#endif  // BUILD_CUDA_MODULE

    return EXIT_SUCCESS;
}
