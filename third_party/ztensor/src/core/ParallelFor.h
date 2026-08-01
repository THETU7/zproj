// ztensor/core/ParallelFor.h
//
// The single parallel primitive backing every element-wise and reduction op.
// Modeled on Open3D's open3d/core/ParallelFor.h.
//
//   template <typename func_t>
//   void ParallelFor(const Device& device, int64_t n, const func_t& func);
//
// CPU path: an OpenMP `parallel for` over [0, n). The CUDA grid-stride kernel
// (ElementWiseKernel_ / ParallelForCUDA_) is added in phase 4 behind
// #ifdef __CUDACC__; the same template API is used by both backends, so kernels
// are written once.
//
// `func` takes an int64_t workload index and returns void. Capture only what
// each workload needs (by value where a kernel may run on both host and
// device).

#pragma once

#include <cstdint>

#include "ztensor/zt/Device.h"
#include "ztensor/zt/utility/Log.h"

#ifdef __CUDACC__
#include "ztensor/zt/Macros.h"
#include "ztensor/zt/cuda/Exception.h"
#include "ztensor/zt/cuda/Guard.h"
#include "ztensor/zt/cuda/Stream.h"
#endif

namespace zt::core {

// Estimate the number of worker threads for CPU parallel regions. Wraps
// omp_get_max_threads() when OpenMP is linked, else 1 (serial).
int EstimateMaxThreads();

// True iff the calling thread is inside an active OpenMP parallel region. The
// reduction engine uses this to avoid nested parallelism.
bool InParallel();

#ifdef __CUDACC__
// ── CUDA path (visible only to .cu TUs) ─────────────────────────────────────
// Grid-stride kernel: each thread handles `THREAD` work items strided by the
// block size. `block * thread` items per CTA. Mirrors Open3D's
// ElementWiseKernel_ / ParallelForCUDA_. Defined BEFORE the ParallelFor
// template because ParallelFor() forwards to it under __CUDACC__.
inline constexpr int64_t ZT_PARFOR_BLOCK = 128;  // threads per block
inline constexpr int64_t ZT_PARFOR_THREAD = 4;   // work items per thread

template<int64_t BLOCK_SIZE, int64_t THREAD_SIZE, typename func_t>
__global__ void ElementWiseKernel_(int64_t n, func_t f) {
    const int64_t items_per_block = BLOCK_SIZE * THREAD_SIZE;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < THREAD_SIZE; ++i) {
        if (idx < n) {
            f(idx);
            idx += BLOCK_SIZE;
        }
    }
}

template<typename func_t>
void ParallelForCUDA_(const Device& device, int64_t n, const func_t& func) {
    if (n <= 0) {
        return;
    }
    if (!device.is_cuda()) {
        ZT_LOG_ERROR("ParallelForCUDA: device {} is not CUDA", device.string());
    }
    ::zt::CUDAScopedDevice scoped(device);
    const int64_t items_per_block = ZT_PARFOR_BLOCK * ZT_PARFOR_THREAD;
    const int64_t grid_size = (n + items_per_block - 1) / items_per_block;
    ElementWiseKernel_<ZT_PARFOR_BLOCK, ZT_PARFOR_THREAD>
        <<<static_cast<unsigned int>(grid_size),
           static_cast<unsigned int>(ZT_PARFOR_BLOCK),
           0,
           ::zt::cuda::GetStream()>>>(n, func);
    ZT_CUDA_GET_LAST_ERROR("ParallelForCUDA kernel launch failed");
}
#endif  // __CUDACC__

// Run func(i) for i in [0, n) on `device`. No-op for n <= 0.
template<typename func_t>
void ParallelFor(const Device& device, int64_t n, const func_t& func) {
#ifdef __CUDACC__
    ParallelForCUDA_(device, n, func);
#else
    if (!device.is_cpu()) {
        ZT_LOG_ERROR("ParallelFor(CPU): device {} is not CPU", device.string());
    }
    if (n <= 0) {
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(EstimateMaxThreads())
#endif
    for (int64_t i = 0; i < n; ++i) {
        func(i);
    }
#endif
}

}  // namespace zt::core
