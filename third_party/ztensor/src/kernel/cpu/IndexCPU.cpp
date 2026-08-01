// ztensor/kernel/IndexCPU.cpp
//
// CPU IndexGet/IndexSet. Each is a ParallelFor over the AdvancedIndexer's
// workloads: GET copies src[i_advanced] -> dst[i], SET copies src[i] ->
// dst[i_advanced]. The dtype dispatch matches Copy (single-dispatch on src
// dtype, then a typed element copy). The AdvancedIndexer's mode-conditional
// pointer arithmetic handles the gather/scatter offset, so the element
// kernel body is identical to a plain copy.

#include <cstdint>

#include "ztensor/zt/utility/Log.h"

#include "core/AdvancedIndexing.h"
#include "core/Dispatch.h"
#include "core/ParallelFor.h"
#include "kernel/Index.h"

namespace zt::kernel {
namespace {

template<typename scalar_t>
void index_get_typed(const core::AdvancedIndexer& ai) {
    core::ParallelFor(ai.GetDevice(), ai.NumWorkloads(), [&](int64_t i) {
        const scalar_t v =
            *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(i));
        *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(i)) = v;
    });
}

template<typename scalar_t>
void index_set_typed(const core::AdvancedIndexer& ai) {
    core::ParallelFor(ai.GetDevice(), ai.NumWorkloads(), [&](int64_t i) {
        const scalar_t v =
            *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(i));
        *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(i)) = v;
    });
}

// IndexAdd: same scatter geometry as IndexSet (Mode::SET: dst advances, src is
// sequential) but *accumulating* — `dst[indexed] += src[i]`. Run serially:
// duplicate indices must sum, and without atomics a parallel loop would race.
// (v1; a parallel CPU path with `#pragma omp atomic` is a future optimization.)
template<typename scalar_t>
void index_add_typed(const core::AdvancedIndexer& ai) {
    const int64_t n = ai.NumWorkloads();
    for (int64_t i = 0; i < n; ++i) {
        const scalar_t v =
            *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(i));
        *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(i)) += v;
    }
}

}  // namespace

void IndexGetCPU(const Tensor& src,
                 const Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const std::vector<int64_t>& indexed_shape,
                 const std::vector<int64_t>& indexed_strides) {
    core::AdvancedIndexer ai(src,
                             dst,
                             index_tensors,
                             indexed_shape,
                             indexed_strides,
                             core::AdvancedIndexer::Mode::GET);
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(
        src.scalar_type(), [&] { index_get_typed<scalar_t>(ai); });
}

void IndexSetCPU(const Tensor& src,
                 const Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const std::vector<int64_t>& indexed_shape,
                 const std::vector<int64_t>& indexed_strides) {
    core::AdvancedIndexer ai(src,
                             dst,
                             index_tensors,
                             indexed_shape,
                             indexed_strides,
                             core::AdvancedIndexer::Mode::SET);
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(
        src.scalar_type(), [&] { index_set_typed<scalar_t>(ai); });
}

void IndexAddCPU(const Tensor& src,
                 const Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const std::vector<int64_t>& indexed_shape,
                 const std::vector<int64_t>& indexed_strides) {
    core::AdvancedIndexer ai(src,
                             dst,
                             index_tensors,
                             indexed_shape,
                             indexed_strides,
                             core::AdvancedIndexer::Mode::SET);
    // Accumulation is restricted to the native-atomic dtypes (Float/Double/
    // Int32/Int64), matching kernel::Reduction's `atomic_supported` set.
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

}  // namespace zt::kernel
