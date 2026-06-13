// CUDA implementation of the WGS84 geodetic -> ECEF transform.
//
// The per-point math (to_ecef) is independent across points, so this is the
// canonical "embarrassingly parallel" GPU workload and the starting point for
// learning CUDA acceleration in this project.
#include "zproj/crs/wgs84.hpp"
#include "zproj/crs/wgs84_to_ecef.hpp"
#include "zproj/cuda/error.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace zproj::crs {

namespace {
constexpr int kBlock = 256;  // threads per block
}  // namespace

__global__ void wgs84_to_ecef_kernel(const Geodetic* in, Ecef* out, std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = to_ecef(in[idx]);
    }
}

void wgs84_to_ecef_device(const Geodetic* d_in, Ecef* d_out, std::size_t n)
{
    if (n == 0)
    {
        return;
    }
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    wgs84_to_ecef_kernel<<<grid, kBlock>>>(d_in, d_out, n);
    ZPROJ_CUDA_CHECK(cudaGetLastError());
}

std::vector<Ecef> wgs84_to_ecef(const std::vector<Geodetic>& geodetic)
{
    std::vector<Ecef> out(geodetic.size());
    if (geodetic.empty())
    {
        return out;
    }

    Geodetic* d_in = nullptr;
    Ecef* d_out = nullptr;
    ZPROJ_CUDA_CHECK(cudaMalloc(&d_in, geodetic.size() * sizeof(Geodetic)));
    ZPROJ_CUDA_CHECK(cudaMalloc(&d_out, geodetic.size() * sizeof(Ecef)));
    ZPROJ_CUDA_CHECK(cudaMemcpy(d_in, geodetic.data(), geodetic.size() * sizeof(Geodetic), cudaMemcpyHostToDevice));

    wgs84_to_ecef_device(d_in, d_out, geodetic.size());
    ZPROJ_CUDA_CHECK(cudaDeviceSynchronize());

    ZPROJ_CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(Ecef), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

}  // namespace zproj::crs
