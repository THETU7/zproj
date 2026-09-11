// Multi-view (N >= 2) RPC triangulation benchmark: back-project each view's
// matched pixel through the analytic inverse at two heights to build viewing
// rays, then intersect all of them in the least-squares sense
// (RpcMultiStereo -- the Slabaugh normal equations, like ASP's
// RPCStereoModel / VisionWorkbench's StereoModel::triangulate_point).
//
// The synthetic view set is deterministic: view 0 is nadir (col = lon,
// row = lat), views 1.. lean ~11 deg off nadir in rotating azimuths, so every
// view pair has a genuine convergence angle and the normal equations are
// well conditioned. Per-point cost scales ~linearly with the view count (one
// analytic inverse pair per view), which the benchmark reports.
//
// Runs both precision paths -- the all-double reference and the float-ENU
// pipeline (float Newton inverse + scene-local ENU float rays; GPU-oriented)
// -- on CPU (OpenMP) and CUDA, with a closed-loop accuracy check against
// known ground truth and a CUDA-vs-CPU agreement check.
//
// Usage:
//   ./build/default/bin/nview_triangulation [points] [views] [reps]
//   e.g. ./build/default/bin/nview_triangulation 1000000 4 5

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
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
using zproj::crs::RpcMultiStereo;
using zproj::crs::StereoPrecision;

// Nadir-ish model: col = lon, row = lat (height-independent), so the ray
// through any pixel is the geodetic vertical -- an exact straight line in
// ECEF.
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

// Oblique model with parametric height coupling: col = lon + samp_h * h,
// row = lat + line_h * h -- the ray leans in the given image directions.
RpcInfo MakeLeaningInfo(double samp_h, double line_h) {
    RpcInfo info = MakeNadirInfo();
    info.samp_num_coeff[3] = samp_h;
    info.line_num_coeff[3] = line_h;
    return info;
}

// View set: view 0 nadir, views 1.. lean in rotating azimuths (+col, +row,
// -col, -row, diagonals, ...) at the same ~11 deg convergence as the two-view
// demo's oblique model.
std::vector<RpcInfo> MakeMultiViewInfos(int num_views) {
    constexpr double k = 1e-4;
    constexpr double kAzimuths[8][2] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
    std::vector<RpcInfo> infos;
    infos.reserve(static_cast<std::size_t>(num_views));
    infos.push_back(MakeNadirInfo());
    for (int i = 1; i < num_views; ++i) {
        const auto& az = kAzimuths[(i - 1) % 8];
        infos.push_back(MakeLeaningInfo(k * az[0], k * az[1]));
    }
    return infos;
}

struct Pt {
    double lon;
    double lat;
    double alt;
};

// Deterministic pseudo-random ground points inside the models' validity
// window.
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

// Forward the ground points into every view and stack the per-view [N, 2]
// tensors into the [V, N, 2] multi-view input. `pts` must outlive the
// returned tensor's use.
zt::Tensor MultiViewColRow(const std::vector<RpcInfo>& infos,
                           std::vector<Pt>& pts) {
    zt::Tensor in = zt::from_blob(pts.data(),
                                  {static_cast<int64_t>(pts.size()), 3},
                                  zt::dtype(zt::kDouble));
    const int64_t n = static_cast<int64_t>(pts.size());
    zt::Tensor colrow = zt::empty({static_cast<int64_t>(infos.size()), n, 2},
                                  zt::dtype(zt::kDouble));
    double* dst = colrow.data_ptr<double>();
    for (const RpcInfo& info : infos) {
        zt::Tensor cr;
        RpcModel(info).lonlatalt_to_colrow(in, cr);
        std::memcpy(dst,
                    cr.data_ptr<double>(),
                    static_cast<std::size_t>(n) * 2 * sizeof(double));
        dst += 2 * n;
    }
    return colrow;
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

// Closed-loop error vs the known ground truth: max |out - truth| for lon
// [deg], lat [deg], alt [m], and the max ray-bundle rms [m]. Non-finite
// outputs are skipped (there should be none for valid inputs).
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
    const int views = (argc > 2) ? std::stoi(argv[2]) : 4;
    const int reps = (argc > 3) ? std::stoi(argv[3]) : 5;

    zt::Logger::Init();

    const std::vector<RpcInfo> infos = MakeMultiViewInfos(views);
    std::vector<Pt> pts = MakePoints(n, infos[0]);
    const zt::Tensor colrow = MultiViewColRow(infos, pts);

    std::cout << "Multi-view RPC triangulation speed test (synthetic nadir + "
              << (views - 1) << " oblique views)\n"
              << "  points per batch   : " << n << '\n'
              << "  views per point    : " << views << '\n'
              << "  CPU/CUDA reps      : " << reps << '\n';

    // Ray span +/-50 m about height_off (ASP's default span).
    constexpr double kSpanM = 50.0;
    const double h_low = infos[0].height_off - kSpanM;
    const double h_high = infos[0].height_off + kSpanM;
    std::cout << "  ray height span    : +/-" << kSpanM
              << " m (about height_off)\n";

    const RpcMultiStereo multi(infos, h_low, h_high);
    const RpcMultiStereo multi_float(
        infos, h_low, h_high, StereoPrecision::FloatEnu);

    // ---- CPU throughput.
    zt::Tensor cpu_ll;
    zt::Tensor cpu_rms;
    const double cpu_ms =
        BenchMs(reps, [&] { multi.triangulate(colrow, cpu_ll, cpu_rms); });
    zt::Tensor cpu_ll_f;
    zt::Tensor cpu_rms_f;
    const double cpu_ms_f = BenchMs(
        reps, [&] { multi_float.triangulate(colrow, cpu_ll_f, cpu_rms_f); });

    std::cout << "\n  CPU backend                     ms/batch      Mpts/s\n";
    std::cout << "  " << std::left << std::setw(28) << "zproj CPU (OpenMP, dbl)"
              << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << cpu_ms << std::setw(12)
              << std::setprecision(3) << Mpts(n, cpu_ms) << '\n';
    std::cout << "  " << std::left << std::setw(28)
              << "zproj CPU (OpenMP, flt-ENU)" << std::right << std::setw(12)
              << std::fixed << std::setprecision(3) << cpu_ms_f << std::setw(12)
              << std::setprecision(3) << Mpts(n, cpu_ms_f) << '\n';

    // ---- Closed-loop accuracy vs the known ground truth (same profile as
    // the two-view demo: lon/lat to machine precision, height dominated by
    // the straight-chord approximation of the curved RPC back-projection
    // locus).
    const Acc acc = ClosedLoopError(cpu_ll, cpu_rms, pts);
    const Acc acc_f = ClosedLoopError(cpu_ll_f, cpu_rms_f, pts);
    std::cout << "\n  closed-loop accuracy (vs known ground truth):\n";
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
        const zt::Tensor gpu_colrow = colrow.cuda();
        zt::Tensor gpu_ll;
        zt::Tensor gpu_rms;

        // Sustained timing: launch back-to-back, sync once at the end.
        for (int i = 0; i < reps; ++i) {
            multi.triangulate(gpu_colrow, gpu_ll, gpu_rms);
        }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            multi.triangulate(gpu_colrow, gpu_ll, gpu_rms);
        }
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const double cuda_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

        zt::Tensor gpu_ll_f;
        zt::Tensor gpu_rms_f;
        for (int i = 0; i < reps; ++i) {
            multi_float.triangulate(gpu_colrow, gpu_ll_f, gpu_rms_f);
        }
        cudaDeviceSynchronize();
        const auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            multi_float.triangulate(gpu_colrow, gpu_ll_f, gpu_rms_f);
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
