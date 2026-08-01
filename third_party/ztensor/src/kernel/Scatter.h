// ztensor/kernel/Scatter.h
//
// Dim-based gather / scatter (DESIGN §8.6.D). These complement the TensorKey-
// based IndexGet/IndexSet: here a single `dim` plus one int64 `index` tensor
// select elements along that axis.
//
//   gather(dim, index):  dst[i]            = src[..., index[i], ...]
//   scatter(dim, index): dst[..., index[i], ...] = src[i]   (overwrite or +=)
//
// `index` must be int64 and contiguous (PyTorch parity). For gather, `dst` and
// `index` share a shape and `src` matches except along `dim` (where it is at
// least max(index)+1); for scatter, `index` and `src` share a shape and `dst`
// matches except along `dim`. All operands must already share a device and
// (for the value operands) a dtype.

#pragma once

#include <cstdint>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

/// `dst = gather(src, index, dim)`. `dst`/`index` share a shape; `src` matches
/// except along `dim`. Dispatches on the device.
void Gather(const Tensor& src,
            const Tensor& index,
            const Tensor& dst,
            int64_t dim);

/// `dst = scatter(src, index, dim)` (dst is the write target; `index`/`src`
/// share a shape). `accumulate=false` overwrites (last-writer-wins on duplicate
/// indices); `accumulate=true` atomically adds (Float/Double/Int32/Int64 only).
void Scatter(const Tensor& src,
             const Tensor& index,
             const Tensor& dst,
             int64_t dim,
             bool accumulate);

/// CPU implementations.
void GatherCPU(const Tensor& src,
               const Tensor& index,
               const Tensor& dst,
               int64_t dim);
void ScatterCPU(const Tensor& src,
                const Tensor& index,
                const Tensor& dst,
                int64_t dim,
                bool accumulate);

/// CUDA implementations (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void GatherCUDA(const Tensor& src,
                const Tensor& index,
                const Tensor& dst,
                int64_t dim);
void ScatterCUDA(const Tensor& src,
                 const Tensor& index,
                 const Tensor& dst,
                 int64_t dim,
                 bool accumulate);
#endif

}  // namespace zt::kernel
