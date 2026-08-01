// ztensor/kernel/CheckIndexBoundsCPU.cpp
//
// CPU bounds scan for scatter/gather indices. The index is already staged
// contiguous int64 on the host, so this is a parallel read-only scan: each
// element is compared against [-dim_size, dim_size) and the (rare) violating
// values record an error flag. OpenMP is optional; without it ParallelFor
// degrades to a serial loop.

#include <atomic>
#include <cstdint>

#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/utility/Log.h"

#include "core/ParallelFor.h"
#include "kernel/CheckIndexBounds.h"

namespace zt::kernel {

void CheckIndexBoundsCPU(const Tensor& index,
                         int64_t dim_size,
                         const char* op) {
    const int64_t n = index.numel();
    const auto* p = index.data_ptr<int64_t>();

    std::atomic<bool> bad{false};
    std::atomic<int64_t> bad_value{0};
    core::ParallelFor(index.device(), n, [&](int64_t i) {
        const int64_t v = p[i];
        // dim_size == 0: [-0, 0) is empty, so every value violates the check
        // (PyTorch parity — the kernels must never touch a zero-element dim).
        if (v < -dim_size || v >= dim_size) {
            bad.store(true, std::memory_order_relaxed);
            bad_value.store(v, std::memory_order_relaxed);
        }
    });

    if (bad.load(std::memory_order_relaxed)) {
        ZT_LOG_ERROR("{}: index {} is out of range for dimension of size {}",
                     op,
                     bad_value.load(std::memory_order_relaxed),
                     dim_size);
    }
}

}  // namespace zt::kernel
