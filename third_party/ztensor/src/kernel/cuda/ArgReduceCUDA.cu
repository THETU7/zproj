// ztensor/kernel/ArgReduceCUDA.cu
//
// CUDA ArgReduce kernel.
//
// Correctness-first implementation: copies src to CPU, runs the CPU
// ArgReduce kernel, then copies the int64 result back to the GPU dst.
// A future optimization can add a 32-bit packed CAS atomic path for
// float/int32 dtypes (with host-precomputed flat indices).

#include "ztensor/zt/cuda/Guard.h"

#include "kernel/ArgReduce.h"

namespace zt {
namespace kernel {

void ArgReduceCUDA(const Tensor& src,
                   const Tensor& dst,
                   IntArrayRef reduction_dims,
                   ArgReduceOp op) {
    CUDAScopedDevice scoped(dst.device());

    // Copy src to CPU, run CPU kernel, copy result back.
    Tensor src_cpu = src.cpu();
    Tensor dst_cpu(dst.sizes(), dst.scalar_type(), Device("cpu"));
    ArgReduceCPU(src_cpu, dst_cpu, reduction_dims, op);
    const_cast<Tensor&>(dst).copy_(dst_cpu);
}

}  // namespace kernel
}  // namespace zt
