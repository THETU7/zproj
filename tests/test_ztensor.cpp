// tests/test_ztensor.cpp
//
// Smoke tests for the vendored ztensor integration (third_party/ztensor).
// CPU checks always run; CUDA checks run only when a device is present.
// Kept dependency-free (no gtest), matching the rest of the zproj test suite.

#include <cstdint>
#include <iostream>

#include "ztensor/zt/Device.h"
#include "ztensor/zt/Scalar.h"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

#ifdef BUILD_CUDA_MODULE
#include "ztensor/zt/cuda/Guard.h"
#endif

int main() {
    zt::Logger::Init();

    // ---- CPU acceptance: ones(4,4) + ones(4,4), summed == 32 ----
    {
        const auto total = zt::ones({4, 4}, zt::dtype(zt::kFloat))
                               .add(zt::ones({4, 4}, zt::dtype(zt::kFloat)))
                               .sum();
        ZT_CHECK(total.item<float>() == 32.0f, "CPU sum mismatch");
    }

    // ---- CPU reductions / arithmetic ----
    {
        const auto r =
            zt::arange(0.0, 6.0, 1.0, zt::dtype(zt::kFloat)).reshape({2, 3});
        ZT_CHECK(r.sum().item<float>() == 15.0f, "CPU arange sum mismatch");
    }

#ifdef BUILD_CUDA_MODULE
    // ---- CUDA: transfer, add, reduction, copy back ----
    if (zt::cuda::IsAvailable()) {
        const auto a = zt::arange(0.0, 6.0, 1.0, zt::dtype(zt::kFloat))
                           .reshape({2, 3})
                           .cuda();
        ZT_CHECK(a.is_cuda(), "expected a CUDA tensor");

        const auto r =
            (a +
             zt::full({2, 3}, zt::Scalar(1.0f), zt::dtype(zt::kFloat)).cuda())
                .sum()
                .cpu();
        // 0+1 + 1+1 + ... + 5+1 = 21
        ZT_CHECK(
            r.item<float>() == 21.0f, "CUDA sum mismatch: {}", r.item<float>());
    } else {
        std::cout << "no CUDA-capable device found; CUDA checks skipped\n";
    }
#endif

    return 0;
}
