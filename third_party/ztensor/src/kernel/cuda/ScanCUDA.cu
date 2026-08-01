// ztensor/kernel/ScanCUDA.cu
//
// CUDA Scan kernel.
//
// Correctness-first implementation: copies src to CPU, runs the CPU Scan
// kernel, then copies the result back to the GPU dst.
// Future optimization: use CUB DeviceScan for contiguous dim=-1 scans.

#include "ztensor/zt/cuda/Guard.h"

#include "kernel/Scan.h"

namespace zt {
namespace kernel {

void ScanCUDA(const Tensor& src, Tensor& dst, int64_t dim, ScanOpCode op) {
    CUDAScopedDevice scoped(dst.device());

    Tensor src_cpu = src.cpu();
    Tensor dst_cpu = src_cpu.clone();  // same shape/dtype as src
    ScanCPU(src_cpu, dst_cpu, dim, op);
    const_cast<Tensor&>(dst).copy_(dst_cpu);
}

}  // namespace kernel
}  // namespace zt
