// ztensor/core/AdvancedIndexing.cpp
//
// AdvancedIndexPreprocessor implementation. Port of Open3D's
// open3d/core/AdvancedIndexing.cpp (RunPreprocess + the restride helpers),
// trimmed: no SYCL, no 32-bit index splitting. The bool->int64 expansion is
// done by the caller (via kernel::NonZero) before constructing this; here we
// only deal with int64 index tensors.

#include "core/AdvancedIndexing.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ztensor/zt/utility/Log.h"

#include "core/ShapeUtil.h"

namespace zt::core {
namespace {

// A 0-d (scalar) int64 index tensor is the sentinel for "this dim is a plain
// full slice" (not advanced-indexed). See the design note in the header.
inline bool is_slice_sentinel(const Tensor& t) {
    return t.dim() == 0;  // 0-d tensor
}

}  // namespace

// ---- static helpers (mirror Open3D AdvancedIndexing.cpp:17-108) ----

bool AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
    const std::vector<Tensor>& indices) {
    // True iff a slice (0-d sentinel) sits between two advanced (non-0-d)
    // indices. NumPy then moves the indexed dims to the front of the output.
    bool seen_advanced = false;
    for (const auto& idx : indices) {
        if (!is_slice_sentinel(idx)) {
            seen_advanced = true;
        } else if (seen_advanced) {
            return true;
        }
    }
    return false;
}

std::pair<Tensor, std::vector<Tensor>>
AdvancedIndexPreprocessor::ShuffleIndexedDimsToFront(
    const Tensor& t, const std::vector<Tensor>& indices) {
    const int64_t ndim = t.dim();
    ZT_CHECK(static_cast<int64_t>(indices.size()) == ndim,
             "ShuffleIndexedDimsToFront: indices count {} != ndim {}",
             indices.size(),
             ndim);
    std::vector<int64_t> order;
    order.reserve(static_cast<std::size_t>(ndim));
    std::vector<int64_t> indexed_dims;
    std::vector<int64_t> slice_dims;
    for (int64_t d = 0; d < ndim; ++d) {
        if (is_slice_sentinel(indices[static_cast<std::size_t>(d)])) {
            slice_dims.push_back(d);
        } else {
            indexed_dims.push_back(d);
        }
    }
    order.insert(order.end(), indexed_dims.begin(), indexed_dims.end());
    order.insert(order.end(), slice_dims.begin(), slice_dims.end());
    Tensor shuffled = t.permute(IntArrayRef(order.data(), order.size()));
    std::vector<Tensor> new_indices;
    new_indices.reserve(static_cast<std::size_t>(ndim));
    for (int64_t d = 0; d < ndim; ++d) {
        new_indices.push_back(indices[static_cast<std::size_t>(
            order[static_cast<std::size_t>(d)])]);
    }
    return {shuffled, new_indices};
}

std::pair<std::vector<Tensor>, std::vector<int64_t>>
AdvancedIndexPreprocessor::ExpandToCommonShapeExceptZeroDim(
    const std::vector<Tensor>& indices) {
    // Compute the broadcast shape over all non-0-d index tensors.
    std::vector<int64_t> replacement;
    bool first = true;
    for (const auto& idx : indices) {
        if (is_slice_sentinel(idx)) continue;
        std::vector<int64_t> shape(idx.sizes().begin(), idx.sizes().end());
        if (first) {
            replacement = shape;
            first = false;
        } else {
            replacement = BroadcastedShape(
                IntArrayRef(replacement.data(), replacement.size()),
                IntArrayRef(shape.data(), shape.size()));
        }
    }
    // Expand each non-0-d index to the replacement shape (0-d left as-is).
    std::vector<Tensor> out;
    out.reserve(indices.size());
    for (const auto& idx : indices) {
        if (is_slice_sentinel(idx)) {
            out.push_back(idx);
        } else {
            out.push_back(idx.expand(
                IntArrayRef(replacement.data(), replacement.size())));
        }
    }
    return {out, replacement};
}

Tensor AdvancedIndexPreprocessor::RestrideTensor(
    const Tensor& t,
    int64_t dims_before,
    int64_t dims_indexed,
    const std::vector<int64_t>& replacement_shape) {
    // The indexed dims are [dims_before, dims_before + dims_indexed). Erase
    // them from shape/strides, then insert replacement_shape at position
    // dims_before (with zero strides so the Indexer broadcasts over them).
    Tensor::ShapeVector shape, strides;
    const int64_t ndim = t.dim();
    for (int64_t d = 0; d < ndim; ++d) {
        const auto ud = static_cast<std::size_t>(d);
        if (d < dims_before) {
            shape.push_back(t.sizes()[ud]);
            strides.push_back(t.strides()[ud]);
        } else if (d == dims_before) {
            for (int64_t r : replacement_shape) {
                shape.push_back(r);
                strides.push_back(0);  // stride 0 -> broadcast
            }
        } else if (d >= dims_before + dims_indexed) {
            shape.push_back(t.sizes()[ud]);
            strides.push_back(t.strides()[ud]);
        }
        // indexed dims (in [dims_before, dims_before+dims_indexed)) are
        // dropped.
    }
    return Tensor(shape,
                  strides,
                  const_cast<void*>(t.data_ptr()),
                  t.scalar_type(),
                  t.GetBlob());
}

Tensor AdvancedIndexPreprocessor::RestrideIndexTensor(const Tensor& idx,
                                                      int64_t dims_before,
                                                      int64_t dims_after) {
    // Pad with leading size-1 (dims_before) and trailing size-1 (dims_after)
    // dims so the index broadcasts against the restrided src. idx is already
    // expanded to the replacement shape; here we only add the 1-size padding.
    Tensor::ShapeVector shape;
    for (int64_t i = 0; i < dims_before; ++i) shape.push_back(1);
    for (int64_t s : idx.sizes()) shape.push_back(s);
    for (int64_t i = 0; i < dims_after; ++i) shape.push_back(1);
    return idx.reshape(IntArrayRef(shape.data(), shape.size()));
}

// ---- RunPreprocess (mirror AdvancedIndexing.cpp:110-228) ----

void AdvancedIndexPreprocessor::RunPreprocess() {
    const int64_t ndim = tensor_.dim();
    ZT_CHECK(static_cast<int64_t>(index_tensors_.size()) == ndim,
             "AdvancedIndexPreprocessor: indices count {} != ndim {}",
             index_tensors_.size(),
             ndim);

    // Dtype + device checks: every index tensor must be int64 on tensor_'s
    // device.
    for (const auto& idx : index_tensors_) {
        ZT_CHECK(
            idx.scalar_type() == ScalarType::Long,
            "AdvancedIndexPreprocessor: index tensors must be Int64, got {}",
            toString(idx.scalar_type()));
        ZT_CHECK(
            idx.device() == tensor_.device(),
            "AdvancedIndexPreprocessor: index device {} != tensor device {}",
            idx.device().string(),
            tensor_.device().string());
    }

    // NumPy's "split" case: shuffle indexed dims to the front.
    if (IsIndexSplittedBySlice(index_tensors_)) {
        auto shuffled = ShuffleIndexedDimsToFront(tensor_, index_tensors_);
        tensor_ = shuffled.first;
        index_tensors_ = std::move(shuffled.second);
    }

    // Broadcast all non-0-d index tensors to a common replacement shape.
    auto expanded = ExpandToCommonShapeExceptZeroDim(index_tensors_);
    index_tensors_ = std::move(expanded.first);
    const auto& replacement_shape = expanded.second;

    // Walk dims to classify: count advanced dims, compute output shape, and
    // record the real per-dim extents/strides of `t`'s indexed axes.
    int64_t dims_before = 0;
    int64_t dims_indexed = 0;
    int64_t dims_after = 0;
    bool started_indexing = false;
    bool emitted_replacement = false;
    output_shape_.clear();
    indexed_shape_.clear();
    indexed_strides_.clear();
    for (int64_t d = 0; d < ndim; ++d) {
        const auto ud = static_cast<std::size_t>(d);
        const Tensor& idx = index_tensors_[ud];
        if (is_slice_sentinel(idx)) {
            // Full-slice dim: contributes tensor_'s extent to the output.
            output_shape_.push_back(tensor_.sizes()[ud]);
            if (!started_indexing) {
                ++dims_before;
            } else {
                ++dims_after;
            }
        } else {
            // Advanced dim: on the first such dim, emit the replacement shape
            // into the output; record tensor_'s real extent + stride for
            // gather/scatter offset computation.
            if (!emitted_replacement) {
                for (int64_t r : replacement_shape) output_shape_.push_back(r);
                emitted_replacement = true;
            }
            indexed_shape_.push_back(tensor_.sizes()[ud]);
            indexed_strides_.push_back(tensor_.strides()[ud]);
            ++dims_indexed;
            started_indexing = true;
        }
    }
    ZT_CHECK(dims_indexed > 0,
             "AdvancedIndexPreprocessor: no advanced index tensors");

    // Restride the tensor (indexed dims -> stride 0) and each index tensor.
    tensor_ =
        RestrideTensor(tensor_, dims_before, dims_indexed, replacement_shape);
    for (auto& idx : index_tensors_) {
        if (!is_slice_sentinel(idx)) {
            idx = RestrideIndexTensor(idx, dims_before, dims_after);
        }
    }
}

AdvancedIndexPreprocessor::AdvancedIndexPreprocessor(
    const Tensor& t, const std::vector<Tensor>& index_tensors)
    : tensor_(t), index_tensors_(index_tensors) {
    RunPreprocess();
}

// ---- AdvancedIndexer ----

AdvancedIndexer::AdvancedIndexer(const Tensor& src,
                                 const Tensor& dst,
                                 const std::vector<Tensor>& index_tensors,
                                 const std::vector<int64_t>& indexed_shape,
                                 const std::vector<int64_t>& indexed_strides,
                                 Mode mode)
    : mode_(mode) {
    ZT_CHECK(indexed_shape.size() == indexed_strides.size(),
             "AdvancedIndexer: indexed_shape/strides size mismatch");
    ZT_CHECK(static_cast<int64_t>(indexed_shape.size()) <= ADV_INDEX_MAX_DIMS,
             "AdvancedIndexer: too many indexed dims {} > {}",
             indexed_shape.size(),
             ADV_INDEX_MAX_DIMS);
    num_indexed_ = static_cast<int64_t>(indexed_shape.size());
    for (int64_t i = 0; i < num_indexed_; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        indexed_shape_[ui] = indexed_shape[ui];
        indexed_strides_[ui] = indexed_strides[ui];
    }
    element_byte_size_ = static_cast<int64_t>(src.element_size());
    device_ = dst.device();

    // Build an Indexer over [src, idx0, idx1, ...] (0-d slice sentinels
    // excluded — they are not operands). Iterate in input-tensor order so the
    // kernel can fetch each index value by position.
    std::vector<Tensor> indexer_inputs;
    indexer_inputs.push_back(src);
    for (const auto& idx : index_tensors) {
        if (idx.dim() == 0) continue;  // skip 0-d slice sentinels
        indexer_inputs.push_back(idx);
    }
    indexer_ = Indexer(indexer_inputs, dst, DtypePolicy::NONE);
}

}  // namespace zt::core
