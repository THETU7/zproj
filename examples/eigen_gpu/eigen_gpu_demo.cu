// examples/eigen_gpu/eigen_gpu_demo.cu
//
// DISABLED. This demo validated Eigen's GPU Core + Tensor modules, which
// required the vendored master Eigen (third_party/eigen, now removed). The
// system Eigen has no GPU/Tensor backend, so the demo is commented out.
//
// Reimplementation plan: replace it with a ztensor-based GPU demo
// (zt::Tensor on CUDA devices); see examples/ztensor_basic/main.cpp and
// tests/test_ztensor.cpp for the ztensor usage pattern.
//
// Original source, kept for reference:
/*// Demonstration that Eigen's GPU path works in this toolchain
// (Eigen master + CUDA 13.2 + g++-15 host + C++20), validating the reason we
// vendor Eigen as a submodule rather than using the system copy.
//
// Two independent checks:
//   (A) Eigen Core GPU module: fixed-size vectors in a device kernel.
//   (B) Eigen Tensor GPU module: an elementwise expression evaluated through
//       Eigen::GpuDevice + GpuStreamDevice (Eigen's own kernel launcher).
//
// This is a CUDA TU (compiled by nvcc); EIGEN_USE_GPU switches on Eigen's
// GPU backends, EIGEN_NO_DEBUG silences host-only asserts on the device path.
#define EIGEN_USE_GPU
#define EIGEN_NO_DEBUG

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

void check_cuda(cudaError_t e, const char* file, int line)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
cudaGetErrorString(e)); std::exit(EXIT_FAILURE);
    }
}
#define CK(e) check_cuda((e), __FILE__, __LINE__)

// (A) Eigen fixed-size vector math directly inside a device kernel.
__global__ void eigen_vec_kernel(const double* in, double* out)
{
    const Eigen::Map<const Eigen::Vector3d> v(in);
    Eigen::Map<Eigen::Vector3d> r(out);
    r = 2.0 * v + Eigen::Vector3d(1.0, 1.0, 1.0);  // expect 2*[1,2,3]+1 =
[3,5,7]
}

int run_eigen_vec()
{
    const double host_in[3] = {1.0, 2.0, 3.0};
    double host_out[3] = {0.0, 0.0, 0.0};
    double* d_in = nullptr;
    double* d_out = nullptr;
    CK(cudaMalloc(&d_in, sizeof(double) * 3));
    CK(cudaMalloc(&d_out, sizeof(double) * 3));
    CK(cudaMemcpy(d_in, host_in, sizeof(double) * 3, cudaMemcpyHostToDevice));

    eigen_vec_kernel<<<1, 1>>>(d_in, d_out);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host_out, d_out, sizeof(double) * 3, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);

    const bool ok = host_out[0] == 3.0 && host_out[1] == 5.0 && host_out[2]
== 7.0; std::printf("[A] Eigen Vector3d on device: [%.1f, %.1f, %.1f] -> %s\n",
                host_out[0], host_out[1], host_out[2], ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// (B) Eigen Tensor elementwise op evaluated through Eigen's GPU device.
int run_eigen_tensor()
{
    constexpr int n = 1 << 20;  // 1M elements
    float* h_in = static_cast<float*>(std::malloc(sizeof(float) * n));
    float* h_out = static_cast<float*>(std::malloc(sizeof(float) * n));
    for (int i = 0; i < n; ++i)
    {
        h_in[i] = 1.0f;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    CK(cudaMalloc(&d_in, sizeof(float) * n));
    CK(cudaMalloc(&d_out, sizeof(float) * n));
    CK(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

    Eigen::GpuStreamDevice stream;            // default stream
    const Eigen::GpuDevice device(&stream);
    Eigen::TensorMap<Eigen::Tensor<float, 1>> in(d_in, n);
    Eigen::TensorMap<Eigen::Tensor<float, 1>> out(d_out, n);
    out.device(device) = in * 2.0f + 1.0f;    // Eigen launches its own kernel
    CK(cudaDeviceSynchronize());

    CK(cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);

    const bool ok = h_out[0] == 3.0f && h_out[n - 1] == 3.0f;
    std::printf("[B] Eigen Tensor + GpuDevice elementwise (n=%d): %.1f .. %.1f
-> %s\n", n, h_out[0], h_out[n - 1], ok ? "OK" : "FAIL");

    std::free(h_in);
    std::free(h_out);
    return ok ? 0 : 1;
}

}  // namespace

int main()
{
    int rc = 0;
    rc |= run_eigen_vec();
    rc |= run_eigen_tensor();
    if (rc == 0)
    {
        std::printf("Eigen GPU/Tensor paths verified.\n");
    }
    return rc;
}
*/
