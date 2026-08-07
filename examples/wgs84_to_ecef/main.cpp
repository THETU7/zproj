// WGS84 geodetic <-> ECEF speed test.
//
// Times the ztensor-backed zproj::crs::wgs84_to_ecef and
// zproj::crs::ecef_to_wgs84 on a batch of points and reports throughput for:
//   1. zproj CPU, single-threaded
//   2. zproj CPU, OpenMP-parallel (the default library path when built with
//      OpenMP)
//   3. zproj CUDA, per-call cost (launch + kernel + device sync each call)
//   4. zproj CUDA, sustained throughput (back-to-back launches, one sync)
//   5. GDAL/PROJ reference (EPSG:4326 <-> EPSG:4978), on CPU
//
// The timed call is the plain public API with a pre-allocated output, so the
// measured cost is the transform itself: no per-call allocation/clone and no
// H2D/D2H transfer inside the timed region. A theoretical memory-bandwidth
// floor is printed as a reference so you can tell whether the kernel is
// memory- or compute-bound. After timing, outputs are cross-checked against
// the GDAL reference (max |delta| in metres for ECEF; lat/lon in degrees and
// height in metres for geodetic).
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

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#ifdef BUILD_CUDA_MODULE
#include "ztensor/zt/cuda/Guard.h"
#endif  // BUILD_CUDA_MODULE

#include "zproj/crs/ecef_to_wgs84.hpp"
#include "zproj/crs/wgs84.hpp"
#include "zproj/crs/wgs84_to_ecef.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"

namespace {

using zproj::crs::Ecef;
using zproj::crs::ecef_to_wgs84;
using zproj::crs::Geodetic;
using zproj::crs::wgs84_to_ecef;

constexpr double kDeg2Rad = std::numbers::pi / 180.0;
constexpr double kRad2Deg = 180.0 / std::numbers::pi;

// Deterministic pseudo-random points with a fixed seed.
std::vector<Geodetic> MakePoints(std::size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> lat_deg(-89.0, 89.0);
    std::uniform_real_distribution<double> lon_deg(-180.0, 180.0);
    std::uniform_real_distribution<double> height(0.0, 1000.0);

    std::vector<Geodetic> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Geodetic is (lon, lat, h): x = lon, y = lat, z = h.
        pts[i] = Geodetic(
            lon_deg(rng) * kDeg2Rad, lat_deg(rng) * kDeg2Rad, height(rng));
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

    std::vector<Ecef> out(geo.size());
    for (std::size_t i = 0; i < geo.size(); ++i) {
        double x = geo[i].x() * kRad2Deg;  // lon
        double y = geo[i].y() * kRad2Deg;  // lat
        double z = geo[i].z();             // h
        if (ct->Transform(1, &x, &y, &z) == 0) {
            std::cerr << "GDAL transform failed at index " << i << "\n";
            std::exit(EXIT_FAILURE);
        }
        out[i] = Ecef(x, y, z);
    }
    return out;
}

// CPU reference via GDAL/PROJ: EPSG:4978 (ECEF) -> EPSG:4326 (lon/lat,
// degrees); returned as geodetic radians to match zproj::crs::Geodetic.
std::vector<Geodetic> EcefToWgs84Gdal(const std::vector<Ecef>& ecef) {
    OGRSpatialReference src;
    src.importFromEPSG(4978);
    src.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);  // x, y, z

    OGRSpatialReference dst;
    dst.importFromEPSG(4326);
    dst.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);  // x=lon, y=lat

    std::unique_ptr<OGRCoordinateTransformation> ct(
        OGRCreateCoordinateTransformation(&src, &dst));
    if (ct == nullptr) {
        std::cerr << "Failed to build GDAL transformation 4978 -> 4326\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<Geodetic> out(ecef.size());
    for (std::size_t i = 0; i < ecef.size(); ++i) {
        double x = ecef[i].x();
        double y = ecef[i].y();
        double z = ecef[i].z();
        if (ct->Transform(1, &x, &y, &z) == 0) {
            std::cerr << "GDAL transform failed at index " << i << "\n";
            std::exit(EXIT_FAILURE);
        }
        // Geodetic is (lon, lat, h): x = lon, y = lat, z = h.
        out[i] = Geodetic(x * kDeg2Rad, y * kDeg2Rad, z);
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
        const Eigen::Vector3d va(a[i].x(), a[i].y(), a[i].z());
        const Eigen::Vector3d vb(b[i].x(), b[i].y(), b[i].z());
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

// Max |delta| between two geodetic point sets.
struct GeodeticError {
    double lat = 0.0;  // rad
    double lon = 0.0;  // rad
    double h = 0.0;    // m
};

GeodeticError MaxGeodeticError(const std::vector<Geodetic>& a,
                               const std::vector<Geodetic>& b) {
    GeodeticError worst;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst.lat = std::max(worst.lat, std::abs(a[i].y() - b[i].y()));
        worst.lon = std::max(worst.lon, std::abs(a[i].x() - b[i].x()));
        worst.h = std::max(worst.h, std::abs(a[i].z() - b[i].z()));
    }
    return worst;
}

void PrintGeodeticError(const char* label, const GeodeticError& e) {
    std::cout << "  max |" << label << "| = lat " << std::fixed
              << std::setprecision(9) << e.lat * kRad2Deg << " deg, lon "
              << e.lon * kRad2Deg << " deg, h " << std::setprecision(6) << e.h
              << " m\n";
}

// A CPU [N, 3] double tensor laid out as consecutive Geodetic triples.
std::vector<Geodetic> ToGeodeticVector(const zt::Tensor& t) {
    const auto* p = t.data_ptr<Geodetic>();
    const std::size_t n = static_cast<std::size_t>(t.size(0));
    return std::vector<Geodetic>(p, p + n);
}

void PrintRow(const char* name, std::size_t n, double ms) {
    const double mpts = static_cast<double>(n) / ms / 1e3;
    std::cout << "  " << std::left << std::setw(26) << name << std::right
              << std::setw(12) << std::fixed << std::setprecision(3) << ms
              << std::setw(12) << std::setprecision(3) << mpts << '\n';
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() { return zt::cuda::IsAvailable(); }

std::string CudaDeviceName() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return "unknown";
    }
    return prop.name;
}

// Theoretical per-batch time (ms) if the kernel were purely memory-bound:
// one pass moves 2 x 24 B per point (read Geodetic, write Ecef) at the
// device's peak bandwidth. Approximate; real GDDR clocks vary.
double MemFloorMs(std::size_t n) {
    // CUDA 13 removed memoryClockRate/memoryBusWidth from cudaDeviceProp, so
    // query the equivalent device attributes instead. The vendor shim remaps
    // cuda* -> hip* for the HIP backend, but cudaDeviceGetAttribute is not
    // part of the shim, so spell the HIP branch explicitly.
    int clock_khz = 0;
    int bus_bits = 0;
#ifdef BUILD_HIP_MODULE
    if (hipDeviceGetAttribute(&clock_khz, hipDevAttrMemoryClockRate, 0) !=
            hipSuccess ||
        hipDeviceGetAttribute(&bus_bits, hipDevAttrGlobalMemoryBusWidth, 0) !=
            hipSuccess) {
        return 0.0;
    }
#else
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0) !=
            cudaSuccess) {
        return 0.0;
    }
#endif
    const double bw_bytes_s = 2.0 * static_cast<double>(clock_khz) * 1e3 *
                              static_cast<double>(bus_bits) / 8.0;
    const double bytes = 2.0 * 24.0 * static_cast<double>(n);
    return bytes / bw_bytes_s * 1e3;
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

    std::cout << "WGS84 geodetic <-> ECEF speed test\n";
    std::cout << "  points per batch : " << n << '\n';
    std::cout << "  CPU/CUDA reps    : " << reps << '\n';
    std::cout << "  GDAL reference   : 1 pass\n";
#ifdef _OPENMP
    std::cout << "  CPU threads      : " << omp_get_max_threads()
              << " (OpenMP)\n";
#else
    std::cout << "  CPU threads      : 1 (no OpenMP)\n";
#endif  // _OPENMP
    std::cout << '\n';

    const std::vector<Geodetic> pts = MakePoints(n);
    const zt::Tensor in = zt::from_blob(const_cast<Geodetic*>(pts.data()),
                                        {static_cast<int64_t>(n), 3},
                                        zt::dtype(zt::kDouble));

    // ---- zproj CPU, single-threaded: isolates the per-core cost ----
    zt::Tensor cpu_out_1t = in.clone();
    double cpu_1t_ms = 0.0;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif  // _OPENMP
    cpu_1t_ms = BenchMs(reps, [&] { wgs84_to_ecef(in, cpu_out_1t); });
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif  // _OPENMP

    // ---- zproj CPU, default (OpenMP-parallel when built with OpenMP) ----
    zt::Tensor cpu_out = in.clone();
    const double cpu_ms = BenchMs(reps, [&] { wgs84_to_ecef(in, cpu_out); });

    // ---- GDAL/PROJ reference (also used for the cross-checks below) ----
    const auto g0 = std::chrono::steady_clock::now();
    const std::vector<Ecef> gdal_ref = Wgs84ToEcefGdal(pts);
    const auto g1 = std::chrono::steady_clock::now();
    const double gdal_ms =
        std::chrono::duration<double, std::milli>(g1 - g0).count();

    // ---- report ----
    std::cout << "  " << std::left << std::setw(26) << "backend" << std::right
              << std::setw(12) << "ms/batch" << std::setw(12) << "Mpts/s"
              << '\n';
    PrintRow("zproj CPU (1 thread)", n, cpu_1t_ms);
    PrintRow("zproj CPU (OpenMP)", n, cpu_ms);
    PrintRow("GDAL/PROJ (CPU ref)", n, gdal_ms);

    const std::vector<Ecef> cpu_ecef = ToEcefVector(cpu_out);
    std::cout << "\nmax |zproj CPU - GDAL| = " << std::fixed
              << std::setprecision(6) << MaxError(cpu_ecef, gdal_ref) << " m\n";

    // ---- inverse: ECEF -> WGS84 geodetic ----
    zt::Tensor geo_out_1t = in.clone();
    double inv_cpu_1t_ms = 0.0;
#ifdef _OPENMP
    const int saved_inv_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif  // _OPENMP
    inv_cpu_1t_ms = BenchMs(reps, [&] { ecef_to_wgs84(cpu_out, geo_out_1t); });
#ifdef _OPENMP
    omp_set_num_threads(saved_inv_threads);
#endif  // _OPENMP

    zt::Tensor geo_out = in.clone();
    const double inv_cpu_ms =
        BenchMs(reps, [&] { ecef_to_wgs84(cpu_out, geo_out); });

    // ---- GDAL/PROJ inverse reference (also used for the cross-checks) ----
    const auto gi0 = std::chrono::steady_clock::now();
    const std::vector<Geodetic> gdal_inv = EcefToWgs84Gdal(cpu_ecef);
    const auto gi1 = std::chrono::steady_clock::now();
    const double gdal_inv_ms =
        std::chrono::duration<double, std::milli>(gi1 - gi0).count();

    std::cout << "\nWGS84 ECEF -> geodetic (inverse) speed test\n";
    std::cout << "  " << std::left << std::setw(26) << "backend" << std::right
              << std::setw(12) << "ms/batch" << std::setw(12) << "Mpts/s"
              << '\n';
    PrintRow("zproj CPU (1 thread)", n, inv_cpu_1t_ms);
    PrintRow("zproj CPU (OpenMP)", n, inv_cpu_ms);
    PrintRow("GDAL/PROJ (CPU ref)", n, gdal_inv_ms);

    const std::vector<Geodetic> cpu_geo = ToGeodeticVector(geo_out);
    std::cout << "\nmax |zproj CPU - GDAL| (inverse)\n";
    PrintGeodeticError("dlat/dlon/dh", MaxGeodeticError(cpu_geo, gdal_inv));

#ifdef BUILD_CUDA_MODULE
    if (HasCudaDevice()) {
        const zt::Tensor gpu_in = in.cuda();
        zt::Tensor gpu_out = gpu_in.clone();

        // Per-call cost: launch + kernel + a full device sync every iteration,
        // i.e. what a synchronous consumer pays per API call.
        const double cuda_call_ms = BenchMs(reps, [&] {
            wgs84_to_ecef(gpu_in, gpu_out);
            cudaDeviceSynchronize();
        });

        // Sustained throughput: back-to-back launches with one sync at the
        // end, amortizing launch/sync overhead to expose raw kernel time.
        for (int i = 0; i < reps; ++i) {
            wgs84_to_ecef(gpu_in, gpu_out);
        }
        cudaDeviceSynchronize();
        const auto s0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            wgs84_to_ecef(gpu_in, gpu_out);
        }
        cudaDeviceSynchronize();
        const auto s1 = std::chrono::steady_clock::now();
        const double cuda_sustained_ms =
            std::chrono::duration<double, std::milli>(s1 - s0).count() / reps;

        std::cout << "\n  CUDA device: " << CudaDeviceName() << '\n';
        PrintRow("zproj CUDA (per-call sync)", n, cuda_call_ms);
        PrintRow("zproj CUDA (sustained)", n, cuda_sustained_ms);
        std::cout << "  speedup vs CPU OpenMP (per-call) : "
                  << std::setprecision(2) << (cpu_ms / cuda_call_ms) << "x\n";
        std::cout << "  speedup vs CPU OpenMP (sustained): "
                  << (cpu_ms / cuda_sustained_ms) << "x\n";
        std::cout << "  mem-bandwidth floor (approx)     : "
                  << std::setprecision(3) << MemFloorMs(n) << " ms/batch\n";

        const std::vector<Ecef> gpu_ecef = ToEcefVector(gpu_out.cpu());
        std::cout << "\nmax |zproj CUDA - GDAL| = " << std::setprecision(6)
                  << MaxError(gpu_ecef, gdal_ref) << " m\n";
        std::cout << "max |zproj CUDA - CPU | = "
                  << MaxError(gpu_ecef, cpu_ecef) << " m\n";

        // ---- inverse (ECEF -> geodetic) on GPU ----
        const zt::Tensor gpu_ecef_in = cpu_out.cuda();
        zt::Tensor gpu_geo = gpu_ecef_in.clone();

        const double inv_cuda_call_ms = BenchMs(reps, [&] {
            ecef_to_wgs84(gpu_ecef_in, gpu_geo);
            cudaDeviceSynchronize();
        });

        for (int i = 0; i < reps; ++i) {
            ecef_to_wgs84(gpu_ecef_in, gpu_geo);
        }
        cudaDeviceSynchronize();
        const auto si0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            ecef_to_wgs84(gpu_ecef_in, gpu_geo);
        }
        cudaDeviceSynchronize();
        const auto si1 = std::chrono::steady_clock::now();
        const double inv_cuda_sustained_ms =
            std::chrono::duration<double, std::milli>(si1 - si0).count() / reps;

        PrintRow("zproj CUDA (per-call sync)", n, inv_cuda_call_ms);
        PrintRow("zproj CUDA (sustained)", n, inv_cuda_sustained_ms);
        std::cout << "  speedup vs CPU OpenMP (per-call) : "
                  << std::setprecision(2) << (inv_cpu_ms / inv_cuda_call_ms)
                  << "x\n";
        std::cout << "  speedup vs CPU OpenMP (sustained): "
                  << (inv_cpu_ms / inv_cuda_sustained_ms) << "x\n";
        std::cout << "  mem-bandwidth floor (approx)     : "
                  << std::setprecision(3) << MemFloorMs(n) << " ms/batch\n";

        const std::vector<Geodetic> gpu_geo_vec =
            ToGeodeticVector(gpu_geo.cpu());
        std::cout << "\nmax |zproj CUDA - GDAL| (inverse)\n";
        PrintGeodeticError("dlat/dlon/dh",
                           MaxGeodeticError(gpu_geo_vec, gdal_inv));
        std::cout << "max |zproj CUDA - CPU | (inverse)\n";
        PrintGeodeticError("dlat/dlon/dh",
                           MaxGeodeticError(gpu_geo_vec, cpu_geo));
    } else {
        std::cout << "\nno CUDA-capable device found; CUDA timing skipped\n";
    }
#else
    std::cout << "\nztensor built without CUDA; CUDA timing skipped\n";
#endif  // BUILD_CUDA_MODULE

    return EXIT_SUCCESS;
}
