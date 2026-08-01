// ztensor/kernel/CheckIndexBoundsCUDA.cu
//
// CUDA bounds scan for scatter/gather indices. The whole index lives on the
// device, so scanning it on the host would cost an O(numel) D2H copy plus a
// serial scan plus a stream sync. Instead a grid-stride kernel (the standard
// ParallelFor elementwise launch) checks every element and records a
// {flag, bad-value} pair in a 2-int64 device slot; only those 16 bytes cross
// back to the host (D2H, synchronized), and the host throws on flag != 0.
// Stream ordering guarantees the scan completes before the scatter/gather
// kernel is launched on the same stream.

#include <cstdint>

#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/utility/Log.h"

#include "core/cuda/CUDAUtils.h"  // CUDAScopedDevice
#include "core/ParallelFor.h"
#include "kernel/CheckIndexBounds.h"

namespace zt::kernel {

void CheckIndexBoundsCUDA(const Tensor& index,
                          int64_t dim_size,
                          const char* op) {
    const int64_t n = index.numel();
    if (n == 0) {
        return;
    }
    const Device dev = index.device();
    const auto* idx = index.data_ptr<int64_t>();

    // flag[0] != 0 -> a violating index exists; flag[1] holds one such value
    // (any is fine for the error message; only the exception type matters).
    Tensor::ShapeVector shape{2};
    Tensor flag(shape, ScalarType::Long, dev);
    flag.zero_();
    int64_t* flag_ptr = flag.data_ptr<int64_t>();
    {
        CUDAScopedDevice scoped(dev);
        core::ParallelFor(dev, n, [=] __device__(int64_t i) {
            const int64_t v = idx[i];
            if (v < -dim_size || v >= dim_size) {
                if (atomicExch(
                        reinterpret_cast<unsigned long long*>(&flag_ptr[0]),
                        1ull) == 0ull) {
                    flag_ptr[1] = v;
                }
            }
        });
    }

    const Tensor h = flag.cpu();  // D2H; synchronizes (see copy contract)
    const int64_t* r = h.data_ptr<int64_t>();
    if (r[0] != 0) {
        ZT_LOG_ERROR("{}: index {} is out of range for dimension of size {}",
                     op,
                     r[1],
                     dim_size);
    }
}

}  // namespace zt::kernel
