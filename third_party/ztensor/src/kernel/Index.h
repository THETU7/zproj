// ztensor/kernel/Index.h
//
// Advanced-indexing gather/scatter kernels. The preprocessor
// (core::AdvancedIndexPreprocessor) computes the restrided src/dst and the
// indexed shape/strides; these kernels do the per-workload copy:
//   * IndexGet  — gather src into dst (mode = GET).
//   * IndexSet  — scatter src into dst (mode = SET, non-atomic).
//
// `indexed_shape` / `indexed_strides` are the per-dim extents and element
// strides of `src`'s indexed axes (output of the preprocessor). Both src and
// dst must already be on the same device; cross-device transfer is the
// caller's responsibility (Tensor::index routes through Copy for that).

#pragma once

#include <cstdint>
#include <vector>

#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

/// `dst = gather(src, index_tensors)` along the indexed axes. `src` and `dst`
/// share a dtype; `index_tensors` are the preprocessor's restrided index
/// tensors (0-d sentinels included). Dispatches on the device. The dst Tensor
/// is `const` because writes go through its data_ptr (the Tensor view object
/// itself is unchanged) — matching the convention of kernel::Copy.
void IndexGet(const Tensor& src,
              const Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const std::vector<int64_t>& indexed_shape,
              const std::vector<int64_t>& indexed_strides);

/// `dst = scatter(src, index_tensors)` (dst is the indexed-write target; src
/// is the RHS value tensor, broadcast over the indexed result shape).
void IndexSet(const Tensor& src,
              const Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const std::vector<int64_t>& indexed_shape,
              const std::vector<int64_t>& indexed_strides);

/// `dst[indexed] += src` — same scatter geometry as IndexSet but *accumulating*
/// (atomic on CUDA, serial on CPU). Used by `Tensor::index_add_`. Duplicate
/// indices sum rather than last-writer-wins. Supported dtypes:
/// Float/Double/Half/BFloat16/Int32/Int64 (native atomicAdd on CUDA; others
/// are rejected — see DESIGN §8.6.D).
void IndexAdd(const Tensor& src,
              const Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const std::vector<int64_t>& indexed_shape,
              const std::vector<int64_t>& indexed_strides);

/// CPU implementations.
void IndexGetCPU(const Tensor& src,
                 const Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const std::vector<int64_t>& indexed_shape,
                 const std::vector<int64_t>& indexed_strides);
void IndexSetCPU(const Tensor& src,
                 const Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const std::vector<int64_t>& indexed_shape,
                 const std::vector<int64_t>& indexed_strides);
void IndexAddCPU(const Tensor& src,
                 const Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const std::vector<int64_t>& indexed_shape,
                 const std::vector<int64_t>& indexed_strides);

/// CUDA implementations (only declared under BUILD_CUDA_MODULE).
#ifdef BUILD_CUDA_MODULE
void IndexGetCUDA(const Tensor& src,
                  const Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const std::vector<int64_t>& indexed_shape,
                  const std::vector<int64_t>& indexed_strides);
void IndexSetCUDA(const Tensor& src,
                  const Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const std::vector<int64_t>& indexed_shape,
                  const std::vector<int64_t>& indexed_strides);
void IndexAddCUDA(const Tensor& src,
                  const Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const std::vector<int64_t>& indexed_shape,
                  const std::vector<int64_t>& indexed_strides);
#endif

}  // namespace zt::kernel
