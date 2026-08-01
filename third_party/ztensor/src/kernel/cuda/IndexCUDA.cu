// ztensor/kernel/IndexCUDA.cu
//
// CUDA IndexGet/IndexSet. Structurally identical to IndexCPU: build an
// AdvancedIndexer, run a grid-stride ParallelFor with a typed element copy.
// The only CUDA-specific pieces are the __device__ lambda annotation and the
// CUDAScopedDevice guard. The AdvancedIndexer's mode-conditional pointer
// arithmetic (gather for GET, scatter for SET) is ZT_HOST_DEVICE, so the same
// indexer drives both backends.
//
// Non-atomic: duplicate indices in IndexSet give last-writer-wins (matches
// PyTorch index_put_). See DESIGN.md phase-5 notes.

#include "ztensor/zt/cuda/Guard.h"
#include "ztensor/zt/Macros.h"
#include "ztensor/zt/utility/Log.h"

#include "core/AdvancedIndexing.h"
#include "core/Dispatch.h"
#include "core/ParallelFor.h"
#include "kernel/Index.h"

namespace zt {
namespace kernel {
namespace {

template<typename scalar_t>
void index_get_typed(const core::AdvancedIndexer& ai) {
    core::ParallelFor(
        ai.GetDevice(), ai.NumWorkloads(), [=] __device__(int64_t i) {
            const scalar_t v =
                *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(i));
            *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(i)) = v;
        });
}

template<typename scalar_t>
void index_set_typed(const core::AdvancedIndexer& ai) {
    core::ParallelFor(
        ai.GetDevice(), ai.NumWorkloads(), [=] __device__(int64_t i) {
            const scalar_t v =
                *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(i));
            *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(i)) = v;
        });
}

// atomic_add for the native-atomic dtypes (same set as ReductionCUDA's
// atomic_supported): int32/int64/float/double.
__device__ inline void index_atomic_add(std::int32_t* addr, std::int32_t v) {
    atomicAdd(addr, v);
}
__device__ inline void index_atomic_add(std::int64_t* addr, std::int64_t v) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(v));
}
__device__ inline void index_atomic_add(float* addr, float v) {
    atomicAdd(addr, v);
}
__device__ inline void index_atomic_add(double* addr, double v) {
    atomicAdd(addr, v);
}

// IndexAdd: scatter-accumulate via atomicAdd. Same Mode::SET geometry as
// IndexSet (dst advances by the indexed offset, src is sequential); the body
// adds instead of overwriting, so duplicate indices sum.
template<typename scalar_t>
void index_add_typed(const core::AdvancedIndexer& ai) {
    core::ParallelFor(
        ai.GetDevice(), ai.NumWorkloads(), [=] __device__(int64_t i) {
            const scalar_t v =
                *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(i));
            index_atomic_add(reinterpret_cast<scalar_t*>(ai.GetOutputPtr(i)),
                             v);
        });
}

}  // namespace

void IndexGetCUDA(const Tensor& src,
                  const Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const std::vector<int64_t>& indexed_shape,
                  const std::vector<int64_t>& indexed_strides) {
    CUDAScopedDevice scoped(dst.device());
    core::AdvancedIndexer ai(src,
                             dst,
                             index_tensors,
                             indexed_shape,
                             indexed_strides,
                             core::AdvancedIndexer::Mode::GET);
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(
        src.scalar_type(), [&] { index_get_typed<scalar_t>(ai); });
}

void IndexSetCUDA(const Tensor& src,
                  const Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const std::vector<int64_t>& indexed_shape,
                  const std::vector<int64_t>& indexed_strides) {
    CUDAScopedDevice scoped(dst.device());
    core::AdvancedIndexer ai(src,
                             dst,
                             index_tensors,
                             indexed_shape,
                             indexed_strides,
                             core::AdvancedIndexer::Mode::SET);
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(
        src.scalar_type(), [&] { index_set_typed<scalar_t>(ai); });
}

void IndexAddCUDA(const Tensor& src,
                  const Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const std::vector<int64_t>& indexed_shape,
                  const std::vector<int64_t>& indexed_strides) {
    CUDAScopedDevice scoped(dst.device());
    core::AdvancedIndexer ai(src,
                             dst,
                             index_tensors,
                             indexed_shape,
                             indexed_strides,
                             core::AdvancedIndexer::Mode::SET);
    // Same dtype scope as IndexAddCPU (native atomicAdd set).
    const auto dt = src.scalar_type();
    if (dt == ScalarType::Float) {
        index_add_typed<float>(ai);
    } else if (dt == ScalarType::Double) {
        index_add_typed<double>(ai);
    } else if (dt == ScalarType::Int) {
        index_add_typed<std::int32_t>(ai);
    } else if (dt == ScalarType::Long) {
        index_add_typed<std::int64_t>(ai);
    } else {
        ZT_LOG_ERROR(
            "IndexAdd: dtype {} not supported (use Float/Double/Int32/Int64)",
            toString(dt));
    }
}

}  // namespace kernel
}  // namespace zt
