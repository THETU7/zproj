// RPC stereo triangulation speed test: back-project a matched pixel pair
// through the analytic inverse at two heights to build viewing rays, then
// intersect them (RpcStereo). Runs both precision paths -- the all-double
// reference and the float-ENU pipeline (float Newton inverse + scene-local
// ENU float rays; ~4x faster on consumer GeForce, GPU-oriented) -- and
// reports CPU (OpenMP) and CUDA throughput plus a closed-loop accuracy check
// (a known ground point is projected into two synthetic RPC images -- one
// nadir, one oblique -- and triangulated back) and a ray-height-span accuracy
// sweep.
//
// The synthetic models are deterministic and need no data file; their per-point
// cost (two analytic inverses + two WGS84->ECEF + a 2-view intersection) is
// representative of a real RPC, so the throughput numbers carry over.
//
// Usage:
//   ./build/default/bin/triangulation [points] [reps]
//   e.g. ./build/default/bin/triangulation 1000000 5

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef BUILD_CUDA_MODULE
#include "ztensor/zt/cuda/Guard.h"
#endif  // BUILD_CUDA_MODULE

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/triangulation.hpp"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace {

using zproj::crs::RpcInfo;
using zproj::crs::RpcModel;
using zproj::crs::RpcStereo;
using zproj::crs::StereoPrecision;

// Nadir-ish model: col = lon, row = lat (height-independent), so the ray
// through any pixel is the geodetic vertical -- an exact straight line in ECEF.
RpcInfo MakeNadirInfo() {
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

    info.line_num_coeff[2] = 1.0;  // row = lat
    info.line_den_coeff[0] = 1.0;
    info.samp_num_coeff[1] = 1.0;  // col = lon
    info.samp_den_coeff[0] = 1.0;

    info.min_lon = 90.0;
    info.min_lat = 20.0;
    info.max_lon = 110.0;
    info.max_lat = 40.0;
    return info;
}

// Oblique model: col = lon + k*h (height coupling). The back-projection locus
// leans with height, so its ray genuinely converges with the nadir vertical.
RpcInfo MakeObliqueInfo() {
    RpcInfo info = MakeNadirInfo();
    info.samp_num_coeff[3] = 1e-4;  // height term
    return info;
}

struct Pt {
    double lon;
    double lat;
    double alt;
};

// Deterministic pseudo-random ground points inside the models' validity window
// (offsets +/- half the scale -- where the analytic inverse is best
// conditioned).
std::vector<Pt> MakePoints(std::size_t n, const RpcInfo& info) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> lon(
        info.long_off - (0.5 * info.long_scale),
        info.long_off + (0.5 * info.long_scale));
    std::uniform_real_distribution<double> lat(
        info.lat_off - (0.5 * info.lat_scale),
        info.lat_off + (0.5 * info.lat_scale));
    std::uniform_real_distribution<double> alt(
        info.height_off - (0.5 * info.height_scale),
        info.height_off + (0.5 * info.height_scale));

    std::vector<Pt> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = Pt{lon(rng), lat(rng), alt(rng)};
    }
    return pts;
}

// Wrap a host vector of (lon, lat, alt) as a non-owning [N, 3] double tensor.
zt::Tensor PointTensor(std::vector<Pt>& pts) {
    return zt::from_blob(pts.data(),
                         {static_cast<int64_t>(pts.size()), 3},
                         zt::dtype(zt::kDouble));
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

double Mpts(std::size_t n, double ms) {
    return static_cast<double>(n) / ms / 1e3;
}

// Closed-loop error vs the known ground truth. Returns max |out - truth| for
// lon [deg], lat [deg], alt [m], and the max ray-to-ray rms [m]. Non-convergent
// points (HUGE_VAL) are skipped -- there should be none for valid inputs.
struct Acc {
    double lon_deg;
    double lat_deg;
    double alt_m;
    double rms_m;
};
Acc ClosedLoopError(const zt::Tensor& lonlath,
                    const zt::Tensor& rms,
                    const std::vector<Pt>& truth) {
    const double* ll = lonlath.data_ptr<double>();
    const double* rm = rms.data_ptr<double>();
    Acc a{0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (!std::isfinite(ll[3 * i])) {
            continue;
        }
        a.lon_deg = std::max(a.lon_deg, std::fabs(ll[3 * i] - truth[i].lon));
        a.lat_deg =
            std::max(a.lat_deg, std::fabs(ll[(3 * i) + 1] - truth[i].lat));
        a.alt_m = std::max(a.alt_m, std::fabs(ll[(3 * i) + 2] - truth[i].alt));
        a.rms_m = std::max(a.rms_m, rm[i]);
    }
    return a;
}

#ifdef BUILD_CUDA_MODULE
bool HasCudaDevice() { return zt::cuda::IsAvailable(); }
#endif  // BUILD_CUDA_MODULE

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = (argc > 1) ? std::stoull(argv[1]) : 1000000;
    const int reps = (argc > 2) ? std::stoi(argv[2]) : 5;

    zt::Logger::Init();

    const RpcInfo left_info = MakeNadirInfo();
    const RpcInfo right_info = MakeObliqueInfo();
    const RpcModel left(left_info);
    const RpcModel right(right_info);

    const std::vector<Pt> pts = MakePoints(n, left_info);
    zt::Tensor in = PointTensor(const_cast<std::vector<Pt>&>(pts));

    // Forward the known ground points into both images to get matched pixels.
    zt::Tensor left_cr;
    zt::Tensor right_cr;
    left.lonlatalt_to_colrow(in, left_cr);
    right.lonlatalt_to_colrow(in, right_cr);

    std::cout
        << "RPC stereo triangulation speed test (synthetic nadir + oblique)\n"
        << "  points per batch   : " << n << '\n'
        << "  CPU/CUDA reps      : " << reps << '\n';

    // ---- Main config: ray span +/-50 m about height_off (ASP's default span).
    constexpr double kSpanM = 50.0;
    const double h_low = left_info.height_off - kSpanM;
    const double h_high = left_info.height_off + kSpanM;
    std::cout << "  ray height span    : +/-" << kSpanM
              << " m (about height_off)\n";

    const RpcStereo stereo(left_info, right_info, h_low, h_high);
    // Float path: float Newton inverse + ENU-local float rays/intersection
    // (GPU-oriented; ~4x on consumer GeForce at mm-level accuracy).
    const RpcStereo stereo_float(
        left_info, right_info, h_low, h_high, StereoPrecision::FloatEnu);

    // ---- CPU throughput.
    zt::Tensor cpu_ll;
    zt::Tensor cpu_rms;
    const double cpu_ms = BenchMs(
        reps, [&] { stereo.triangulate(left_cr, right_cr, cpu_ll, cpu_rms); });
    zt::Tensor cpu_ll_f;
    zt::Tensor cpu_rms_f;
    const double cpu_ms_f = BenchMs(reps, [&] {
        stereo_float.triangulate(left_cr, right_cr, cpu_ll_f, cpu_rms_f);
    });

    std::cout << "\n  CPU backend                     ms/batch      Mpts/s\n";
    std::cout << "  " << std::left << std::setw(28) << "zproj CPU (OpenMP, dbl)"
              << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << cpu_ms << std::setw(12)
              << std::setprecision(3) << Mpts(n, cpu_ms) << '\n';
    std::cout << "  " << std::left << std::setw(28)
              << "zproj CPU (OpenMP, flt-ENU)" << std::right << std::setw(12)
              << std::fixed << std::setprecision(3) << cpu_ms_f << std::setw(12)
              << std::setprecision(3) << Mpts(n, cpu_ms_f) << '\n';

    // ---- Closed-loop accuracy vs the known ground truth. lon/lat recover to
    // machine precision (the nadir ray is an exact ECEF vertical); the height
    // error is dominated by the oblique ray's straight-chord approximation of
    // the slightly-curved RPC back-projection locus, amplified by the shallow
    // (~11 deg) ray convergence. The ray heights bracket height_off but not the
    // full +/-0.5*height_scale alt range, so the worst-case points (whose true
    // height lies outside the +/-50 m ray bracket) extrapolate the chord; the
    // rms reflects the typical (in-bracket) sub-millimetre accuracy. A DEM
    // supplying per-point bracketing heights is the planned extension.
    const Acc acc = ClosedLoopError(cpu_ll, cpu_rms, pts);
    const Acc acc_f = ClosedLoopError(cpu_ll_f, cpu_rms_f, pts);
    std::cout << "\n  closed-loop accuracy (vs known ground truth, span +/-"
              << kSpanM << " m):\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "    double    : max |lon| " << acc.lon_deg << ", |lat| "
              << acc.lat_deg << ", |alt| " << acc.alt_m << " m, rms "
              << acc.rms_m << " m\n";
    std::cout << "    float-ENU : max |lon| " << acc_f.lon_deg << ", |lat| "
              << acc_f.lat_deg << ", |alt| " << acc_f.alt_m << " m, rms "
              << acc_f.rms_m << " m\n"
              << std::defaultfloat;

#ifdef BUILD_CUDA_MODULE
    if (HasCudaDevice()) {
        const zt::Tensor gpu_left = left_cr.cuda();
        const zt::Tensor gpu_right = right_cr.cuda();
        zt::Tensor gpu_ll;
        zt::Tensor gpu_rms;

        // Sustained timing: launch back-to-back, sync once at the end.
        for (int i = 0; i < reps; ++i) {
            stereo.triangulate(gpu_left, gpu_right, gpu_ll, gpu_rms);
        }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            stereo.triangulate(gpu_left, gpu_right, gpu_ll, gpu_rms);
        }
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const double cuda_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

        zt::Tensor gpu_ll_f;
        zt::Tensor gpu_rms_f;
        for (int i = 0; i < reps; ++i) {
            stereo_float.triangulate(gpu_left, gpu_right, gpu_ll_f, gpu_rms_f);
        }
        cudaDeviceSynchronize();
        const auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            stereo_float.triangulate(gpu_left, gpu_right, gpu_ll_f, gpu_rms_f);
        }
        cudaDeviceSynchronize();
        const auto t3 = std::chrono::steady_clock::now();
        const double cuda_ms_f =
            std::chrono::duration<double, std::milli>(t3 - t2).count() / reps;

        std::cout
            << "\n  CUDA backend                    ms/batch      Mpts/s\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "zproj CUDA (sustained, dbl)" << std::right
                  << std::setw(12) << std::fixed << std::setprecision(3)
                  << cuda_ms << std::setw(12) << std::setprecision(3)
                  << Mpts(n, cuda_ms) << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "zproj CUDA (sustained, flt-ENU)" << std::right
                  << std::setw(12) << std::fixed << std::setprecision(3)
                  << cuda_ms_f << std::setw(12) << std::setprecision(3)
                  << Mpts(n, cuda_ms_f) << '\n';
        std::cout << std::defaultfloat;

        // CUDA vs CPU agreement, double and float paths.
        const auto ReportGpuCpu = [&](const zt::Tensor& gpu,
                                      const zt::Tensor& cpu,
                                      const char* tag) {
            const zt::Tensor gpu_cpu = gpu.cpu();
            const double* a = cpu.data_ptr<double>();
            const double* b = gpu_cpu.data_ptr<double>();
            double dl = 0.0;
            double da = 0.0;
            double dh = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                if (!std::isfinite(b[3 * i])) {
                    continue;
                }
                dl = std::max(dl, std::fabs(b[3 * i] - a[3 * i]));
                da = std::max(da, std::fabs(b[(3 * i) + 1] - a[(3 * i) + 1]));
                dh = std::max(dh, std::fabs(b[(3 * i) + 2] - a[(3 * i) + 2]));
            }
            std::cout << "  max |gpu - cpu| (" << tag
                      << ") = " << std::scientific << std::setprecision(3)
                      << "lon " << dl << ", lat " << da << ", alt " << dh
                      << " (deg, deg, m)\n"
                      << std::defaultfloat;
        };
        ReportGpuCpu(gpu_ll, cpu_ll, "dbl");
        ReportGpuCpu(gpu_ll_f, cpu_ll_f, "flt-ENU");
    } else {
        std::cout << "\n  (no CUDA-capable device; skipping CUDA path)\n";
    }
#endif  // BUILD_CUDA_MODULE

    return 0;
}
