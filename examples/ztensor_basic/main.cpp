// examples/ztensor_basic/main.cpp
//
// End-to-end smoke example for the vendored ztensor (third_party/ztensor).
// Exercises the CPU backend always and the CUDA backend when a device is
// present, so the vendored integration is compiled and linked into every
// zproj build.

#include <cstdint>
#include <iostream>

#include "ztensor/zt/Device.h"
#include "ztensor/zt/Scalar.h"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

#ifdef BUILD_CUDA_MODULE
#include <cuda_runtime.h>
#endif

int main() {
    zt::Logger::Init();

    // ---- CPU: ztensor's canonical acceptance check ----
    const auto total = zt::ones({4, 4}, zt::dtype(zt::kFloat))
                           .add(zt::ones({4, 4}, zt::dtype(zt::kFloat)))
                           .sum();
    const float s = total.item<float>();
    std::cout << "ones(4,4) + ones(4,4), summed = " << s << "\n";
    ZT_CHECK(s == 32.0f, "unexpected sum: {}", s);

#ifdef BUILD_CUDA_MODULE
    // ---- CUDA round-trip: build on CPU, add on GPU, copy back ----
    int n_devices = 0;
    const auto err = cudaGetDeviceCount(&n_devices);
    if (err != cudaSuccess || n_devices == 0) {
        std::cout << "no CUDA-capable device found; skipping CUDA checks\n";
        return 0;
    }

    const auto a =
        zt::arange(0.0, 6.0, 1.0, zt::dtype(zt::kFloat)).reshape({2, 3});
    const auto b = zt::full({2, 3}, zt::Scalar(10.0f), zt::dtype(zt::kFloat));

    const auto g = a.cuda();
    ZT_CHECK(g.is_cuda(), "expected a CUDA tensor");
    std::cout << "a on device: " << g.device().string() << "\n";

    const auto r = (g + b.cuda()).cpu();
    ZT_CHECK(r.is_cpu(), "expected the result back on CPU");
    const auto* p = r.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        ZT_CHECK(p[i] == static_cast<float>(i) + 10.0f,
                 "mismatch at [{}]: {} != {}",
                 i,
                 p[i],
                 i + 10);
    }
    std::cout << "CUDA a + 10 = [" << p[0] << ", ..., " << p[5] << "]\n";
    std::cout << "ztensor integration OK (CPU + CUDA)\n";
#else
    std::cout << "ztensor built without CUDA; CPU checks OK\n";
#endif

    return 0;
}
