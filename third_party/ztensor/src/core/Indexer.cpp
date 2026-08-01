// ztensor/core/Indexer.cpp
//
// Out-of-line Indexer / TensorRef setup: broadcasting & reduction restriding,
// dimension coalescing/reordering, and the shrink / contiguity helpers. The
// hot per-workload pointer queries live inline in Indexer.h. Port of Open3D's
// open3d/core/Indexer.cpp, trimmed (see Indexer.h header note).

#include "core/Indexer.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "core/ShapeUtil.h"

namespace zt::core {

// ---------------------------------------------------------------------------
// TensorRef
// ---------------------------------------------------------------------------

void TensorRef::Permute(IntArrayRef dims) {
    if (static_cast<int64_t>(dims.size()) != ndims_) {
        ZT_LOG_ERROR("TensorRef::Permute: dims count {} != ndims {}",
                     dims.size(),
                     ndims_);
    }
    std::vector<bool> seen_dims(static_cast<std::size_t>(ndims_), false);
    for (const int64_t dim : dims) {
        const auto w = static_cast<std::size_t>(WrapDim(dim, ndims_));
        seen_dims[w] = true;
    }
    if (!std::all_of(seen_dims.begin(), seen_dims.end(), [](bool seen) {
            return seen;
        })) {
        ZT_LOG_ERROR(
            "TensorRef::Permute: dims must be a permutation of [0, {})",
            ndims_);
    }
    std::vector<int64_t> new_shape(static_cast<std::size_t>(ndims_));
    std::vector<int64_t> new_byte_strides(static_cast<std::size_t>(ndims_));
    for (int64_t i = 0; i < ndims_; ++i) {
        const auto old_dim = static_cast<std::size_t>(WrapDim(dims[i], ndims_));
        new_shape[static_cast<std::size_t>(i)] = shape_[old_dim];
        new_byte_strides[static_cast<std::size_t>(i)] = byte_strides_[old_dim];
    }
    for (int64_t i = 0; i < ndims_; ++i) {
        shape_[static_cast<std::size_t>(i)] =
            new_shape[static_cast<std::size_t>(i)];
        byte_strides_[static_cast<std::size_t>(i)] =
            new_byte_strides[static_cast<std::size_t>(i)];
    }
}

bool TensorRef::IsContiguous() const {
    std::vector<int64_t> shape(static_cast<std::size_t>(ndims_));
    std::vector<int64_t> strides(static_cast<std::size_t>(ndims_));
    for (int64_t i = 0; i < ndims_; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        shape[ui] = shape_[ui];
        // byte_strides_ is in bytes; DefaultStrides works in elements.
        strides[ui] = byte_strides_[ui] / dtype_byte_size_;
    }
    return DefaultStrides(shape) == strides;
}

// ---------------------------------------------------------------------------
// Indexer
// ---------------------------------------------------------------------------

Indexer::Indexer(const std::vector<Tensor>& input_tensors,
                 const Tensor& output_tensor,
                 DtypePolicy dtype_policy,
                 IntArrayRef reduction_dims) {
    num_inputs_ = static_cast<int64_t>(input_tensors.size());
    num_outputs_ = 1;
    if (num_inputs_ < 1) {
        ZT_LOG_ERROR("Indexer: needs at least 1 input, got {}", num_inputs_);
    }
    if (num_inputs_ > MAX_INPUTS) {
        ZT_LOG_ERROR(
            "Indexer: at most {} inputs, got {}", MAX_INPUTS, num_inputs_);
    }
    if (num_outputs_ > MAX_OUTPUTS) {
        ZT_LOG_ERROR(
            "Indexer: at most {} outputs, got {}", MAX_OUTPUTS, num_outputs_);
    }

    // Dtype-policy checks.
    const auto ref_dtype = input_tensors[0].scalar_type();
    const auto check_inputs_same = [&]() {
        for (const auto& t : input_tensors) {
            if (t.scalar_type() != ref_dtype) {
                ZT_LOG_ERROR("Indexer: input dtype mismatch {} != {}",
                             toString(t.scalar_type()),
                             toString(ref_dtype));
            }
        }
    };
    if (dtype_policy == DtypePolicy::ALL_SAME) {
        check_inputs_same();
        if (output_tensor.scalar_type() != ref_dtype) {
            ZT_LOG_ERROR("Indexer: output dtype {} != {}",
                         toString(output_tensor.scalar_type()),
                         toString(ref_dtype));
        }
    } else if (dtype_policy == DtypePolicy::INPUT_SAME) {
        check_inputs_same();
    } else if (dtype_policy == DtypePolicy::INPUT_SAME_OUTPUT_BOOL) {
        check_inputs_same();
        if (output_tensor.scalar_type() != ScalarType::Bool) {
            ZT_LOG_ERROR("Indexer: comparison output must be Bool, got {}",
                         toString(output_tensor.scalar_type()));
        }
    } else if (dtype_policy == DtypePolicy::NONE) {
        // No dtype checks.
    } else {
        ZT_LOG_ERROR("Indexer: unimplemented dtype policy");
    }

    // Convert to TensorRef.
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_[static_cast<std::size_t>(i)] = TensorRef(input_tensors[i]);
    }
    outputs_[0] = TensorRef(output_tensor);

    if (!reduction_dims.empty()) {
        if (num_inputs_ != 1) {
            ZT_LOG_ERROR("Indexer: reduction op needs exactly 1 input, got {}",
                         num_inputs_);
        }
        // The indexer only handles keepdim-shaped outputs (size-1 on reduced
        // axes); the caller reshapes a non-keepdim dst to keepdim first.
        const std::vector<int64_t> expected =
            ReductionShape(input_tensors[0].sizes(), reduction_dims, true);
        if (IntArrayRef(expected) != output_tensor.sizes()) {
            ZT_LOG_ERROR(
                "Indexer: reduction output shape is not the keepdim shape "
                "of (input reduced over reduction_dims)");
        }
        // Zero the output strides on reduced axes; stride==0 marks a reduction
        // dim throughout the indexer.
        ReductionRestride(
            outputs_[0], inputs_[0].ndims_, inputs_[0].shape_.data());

        ndims_ = inputs_[0].ndims_;
        ReorderDimensions();
        for (int64_t i = 0; i < ndims_; ++i) {
            primary_shape_[static_cast<std::size_t>(i)] =
                inputs_[0].shape_[static_cast<std::size_t>(i)];
        }
        CoalesceDimensions();
    } else {
        // Broadcast inputs to the (single) output shape.
        for (int64_t i = 0; i < num_inputs_; ++i) {
            BroadcastRestride(inputs_[static_cast<std::size_t>(i)],
                              outputs_[0].ndims_,
                              outputs_[0].shape_.data());
        }
        ndims_ = outputs_[0].ndims_;
        for (int64_t i = 0; i < ndims_; ++i) {
            primary_shape_[static_cast<std::size_t>(i)] =
                outputs_[0].shape_[static_cast<std::size_t>(i)];
        }
    }

    UpdatePrimaryStrides();
    UpdateContiguousFlags();
}

void Indexer::BroadcastRestride(TensorRef& src,
                                int64_t dst_ndims,
                                const int64_t* dst_shape) {
    const int64_t src_ndims = src.ndims_;
    // Shift the existing dims to the right to make room for omitted (leading)
    // dims, then pad the front with (shape=1, stride=0).
    const int64_t ndims_omitted = dst_ndims - src_ndims;
    for (int64_t i = src_ndims - 1; i >= 0; --i) {
        src.shape_[static_cast<std::size_t>(ndims_omitted + i)] =
            src.shape_[static_cast<std::size_t>(i)];
        src.byte_strides_[static_cast<std::size_t>(ndims_omitted + i)] =
            src.byte_strides_[static_cast<std::size_t>(i)];
    }
    for (int64_t i = 0; i < ndims_omitted; ++i) {
        src.shape_[static_cast<std::size_t>(i)] = 1;
        src.byte_strides_[static_cast<std::size_t>(i)] = 0;
    }
    src.ndims_ = dst_ndims;

    // Zero the stride on broadcast dims (src size 1, dst size > 1).
    for (int64_t i = 0; i < dst_ndims; ++i) {
        if (src.shape_[static_cast<std::size_t>(i)] == 1 && dst_shape[i] != 1) {
            src.byte_strides_[static_cast<std::size_t>(i)] = 0;
        }
    }
}

void Indexer::ReductionRestride(TensorRef& dst,
                                int64_t src_ndims,
                                const int64_t* src_shape) {
    if (dst.ndims_ != src_ndims) {
        ZT_LOG_ERROR("Indexer::ReductionRestride: src ndims {} != dst ndims {}",
                     src_ndims,
                     dst.ndims_);
    }
    for (int64_t i = 0; i < dst.ndims_; ++i) {
        if (dst.shape_[static_cast<std::size_t>(i)] == 1 && src_shape[i] != 1) {
            dst.byte_strides_[static_cast<std::size_t>(i)] = 0;
        }
    }
}

void Indexer::CoalesceDimensions() {
    if (ndims_ <= 1) {
        return;
    }

    // Two dims can merge if either is size 1, or shape0 * stride[dim0] ==
    // stride[dim1] for every operand (so a single linear step covers both).
    const auto can_coalesce = [&](int64_t dim0, int64_t dim1) {
        const auto u0 = static_cast<std::size_t>(dim0);
        const auto u1 = static_cast<std::size_t>(dim1);
        const int64_t shape0 = primary_shape_[u0];
        const int64_t shape1 = primary_shape_[u1];
        if (shape0 == 1 || shape1 == 1) {
            return true;
        }
        for (int64_t i = 0; i < num_inputs_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            const int64_t stride = inputs_[ui].byte_strides_[u0];
            if (shape0 * stride != inputs_[ui].byte_strides_[u1]) {
                return false;
            }
        }
        for (int64_t i = 0; i < num_outputs_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            const int64_t stride = outputs_[ui].byte_strides_[u0];
            if (shape0 * stride != outputs_[ui].byte_strides_[u1]) {
                return false;
            }
        }
        return true;
    };

    // Replace every operand's stride at dim0 with its stride at dim1.
    const auto replace_stride = [&](int64_t dim0, int64_t dim1) {
        const auto u0 = static_cast<std::size_t>(dim0);
        const auto u1 = static_cast<std::size_t>(dim1);
        for (int64_t i = 0; i < num_inputs_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            inputs_[ui].byte_strides_[u0] = inputs_[ui].byte_strides_[u1];
        }
        for (int64_t i = 0; i < num_outputs_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            outputs_[ui].byte_strides_[u0] = outputs_[ui].byte_strides_[u1];
        }
    };

    int64_t prev_dim = 0;
    for (int64_t dim = 1; dim < ndims_; ++dim) {
        if (can_coalesce(prev_dim, dim)) {
            if (primary_shape_[static_cast<std::size_t>(prev_dim)] == 1) {
                replace_stride(prev_dim, dim);
            }
            primary_shape_[static_cast<std::size_t>(prev_dim)] *=
                primary_shape_[static_cast<std::size_t>(dim)];
        } else {
            ++prev_dim;
            if (prev_dim != dim) {
                replace_stride(prev_dim, dim);
                primary_shape_[static_cast<std::size_t>(prev_dim)] =
                    primary_shape_[static_cast<std::size_t>(dim)];
            }
        }
    }

    ndims_ = prev_dim + 1;
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_[static_cast<std::size_t>(i)].ndims_ = ndims_;
    }
    for (int64_t i = 0; i < num_outputs_; ++i) {
        outputs_[static_cast<std::size_t>(i)].ndims_ = ndims_;
    }

    UpdatePrimaryStrides();
    UpdateContiguousFlags();
}

void Indexer::ReorderDimensions() {
    if (ndims_ == 1) {
        return;
    }

    std::vector<int64_t> permute(static_cast<std::size_t>(ndims_));
    std::iota(permute.rbegin(), permute.rend(), 0);

    // Returns -1 / 0 / 1: no-swap / undecided / swap dim0 with dim1. Reduction
    // dims (output stride 0) are sorted to the front; otherwise dims are
    // ordered by ascending stride for better memory coalescing.
    const auto should_swap = [&](int64_t dim0, int64_t dim1) {
        const auto u0 = static_cast<std::size_t>(dim0);
        const auto u1 = static_cast<std::size_t>(dim1);
        for (int64_t i = 0; i < num_outputs_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            const int64_t stride0 = outputs_[ui].byte_strides_[u0];
            const int64_t stride1 = outputs_[ui].byte_strides_[u1];
            if (stride0 == 0 && stride1 != 0) {
                return -1;
            }
            if (stride1 == 0 && stride0 != 0) {
                return 1;
            }
            if (stride0 != 0 && stride1 != 0) {
                return stride0 <= stride1 ? -1 : 1;
            }
        }
        for (int64_t i = 0; i < num_inputs_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            const int64_t stride0 = inputs_[ui].byte_strides_[u0];
            const int64_t stride1 = inputs_[ui].byte_strides_[u1];
            if (stride0 == 0 || stride1 == 0) {
                continue;
            }
            return stride0 <= stride1 ? -1 : 1;
        }
        return 0;
    };

    // Insertion sort tolerating ambiguous (== 0) comparisons.
    for (int64_t i = 1; i < ndims_; ++i) {
        int64_t dim1 = i;
        for (int64_t dim0 = i - 1; dim0 >= 0; --dim0) {
            const int comparison =
                should_swap(permute[static_cast<std::size_t>(dim0)],
                            permute[static_cast<std::size_t>(dim1)]);
            if (comparison > 0) {
                std::swap(permute[static_cast<std::size_t>(dim0)],
                          permute[static_cast<std::size_t>(dim1)]);
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
        }
    }

    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_[static_cast<std::size_t>(i)].Permute(permute);
    }
    for (int64_t i = 0; i < num_outputs_; ++i) {
        outputs_[static_cast<std::size_t>(i)].Permute(permute);
    }
}

void Indexer::UpdatePrimaryStrides() {
    int64_t stride = 1;
    for (int64_t i = ndims_ - 1; i >= 0; --i) {
        primary_strides_[static_cast<std::size_t>(i)] = stride;
        // Guard 0-sized dims so they don't collapse the running product.
        stride = primary_shape_[static_cast<std::size_t>(i)] > 1
                     ? stride * primary_shape_[static_cast<std::size_t>(i)]
                     : stride;
    }
}

void Indexer::UpdateContiguousFlags() {
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_contiguous_[static_cast<std::size_t>(i)] =
            inputs_[static_cast<std::size_t>(i)].IsContiguous();
    }
    for (int64_t i = 0; i < num_outputs_; ++i) {
        outputs_contiguous_[static_cast<std::size_t>(i)] =
            outputs_[static_cast<std::size_t>(i)].IsContiguous();
    }
}

int64_t Indexer::NumOutputElements() const {
    // All outputs share a shape; query outputs_[0].
    int64_t n = 1;
    for (int64_t i = 0; i < ndims_; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        if (outputs_[0].byte_strides_[ui] != 0 || primary_shape_[ui] == 0) {
            n *= primary_shape_[ui];
        }
    }
    return n;
}

int64_t Indexer::NumReductionDims() const {
    int64_t count = 0;
    for (int64_t dim = 0; dim < ndims_; ++dim) {
        if (outputs_[0].byte_strides_[static_cast<std::size_t>(dim)] == 0) {
            ++count;
        }
    }
    return count;
}

void Indexer::ShrinkDim(int64_t dim, int64_t start, int64_t size) {
    if (dim < 0 || dim >= ndims_) {
        ZT_LOG_ERROR(
            "Indexer::ShrinkDim: 0 <= dim < {} required, got {}", ndims_, dim);
    }
    if (size <= 0) {
        ZT_LOG_ERROR("Indexer::ShrinkDim: size {} must be > 0", size);
    }
    const auto udim = static_cast<std::size_t>(dim);
    // Advance every operand's data pointer to the slice start along `dim`.
    for (int64_t i = 0; i < num_inputs_; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        inputs_[ui].data_ptr_ = static_cast<char*>(inputs_[ui].data_ptr_) +
                                (inputs_[ui].byte_strides_[udim] * start);
    }
    for (int64_t i = 0; i < num_outputs_; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        outputs_[ui].data_ptr_ = static_cast<char*>(outputs_[ui].data_ptr_) +
                                 (outputs_[ui].byte_strides_[udim] * start);
    }

    primary_shape_[udim] = size;
    UpdatePrimaryStrides();
    UpdateContiguousFlags();

    if (size == 1) {
        CoalesceDimensions();
    }
}

}  // namespace zt::core
