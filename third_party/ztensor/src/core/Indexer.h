// ztensor/core/Indexer.h
//
// Indexing engine for element-wise and reduction ops with broadcasting.
// Trimmed port of Open3D's open3d/core/Indexer.{h,cpp}: same TensorRef/Indexer
// design and the restride / coalesce / reorder algorithms, with the CUDA-only
// and 32-bit-index-splitting machinery dropped (deferred to phase 4 / unused by
// the CPU path).
//
// * TensorRef is a POD view {data_ptr, ndims, byte_size, shape, byte_strides}.
//   Byte strides are stored internally even though the public Tensor reports
//   strides in *elements*; the conversion lives in the TensorRef(Tensor) ctor.
// * Indexer binds N inputs + 1 output, broadcasts inputs to the output shape
//   (element-wise) or marks reduction axes via output stride==0 (reuction),
//   coalesces/reorders dimensions for cheaper per-workload indexing, and
//   answers "pointer for workload i" queries from the typed element kernels.
//
// Deliberately dropped vs Open3D (phase 3 scope): OffsetCalculator,
// CanUse32BitIndexing / SplitLargestDim / SplitTo32BitIndexing /
// IndexerIterator, GetPerOutputIndexer, ISPC bindings, final_output_ /
// accumulate_ (CUDA-only).

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "ztensor/zt/ArrayRef.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/utility/Log.h"

#include "core/cuda/CUDAUtils.h"

namespace zt::core {

// Maximum number of dimensions a TensorRef can describe (design §4.8).
inline constexpr int64_t MAX_DIMS = 8;

// Maximum number of input operands an op may have.
inline constexpr int64_t MAX_INPUTS = 5;

// ztensor ops are single-output (including reductions), so one slot suffices.
inline constexpr int64_t MAX_OUTPUTS = 1;

/// A minimalistic POD view that references a Tensor for indexing. The default
/// copy constructor copies the full shape/stride arrays, which is what kernel
/// lambdas capture by value.
struct TensorRef {
    TensorRef() = default;

    /// Build a TensorRef view of `t`. Copies the (shape, byte_strides) metadata
    /// only; it neither owns nor copies the underlying storage.
    TensorRef(const Tensor& t) {
        if (t.dim() > MAX_DIMS) {
            ZT_LOG_ERROR("TensorRef: tensor has {} dims > MAX_DIMS {}",
                         t.dim(),
                         MAX_DIMS);
        }
        data_ptr_ = const_cast<void*>(t.data_ptr());
        ndims_ = t.dim();
        dtype_byte_size_ = static_cast<int64_t>(t.element_size());
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[static_cast<std::size_t>(i)] = t.size(i);
            byte_strides_[static_cast<std::size_t>(i)] =
                t.stride(i) * dtype_byte_size_;
        }
    }

    /// Permute (dimension shuffle) this reference. Defined out-of-line.
    void Permute(IntArrayRef dims);

    /// True iff the referenced buffer is row-major contiguous. Out-of-line.
    bool IsContiguous() const;

    void* data_ptr_ = nullptr;
    int64_t ndims_ = 0;
    int64_t dtype_byte_size_ = 0;
    std::array<int64_t, MAX_DIMS> shape_ = {};
    std::array<int64_t, MAX_DIMS> byte_strides_ = {};
};

/// Dtype contract the Indexer enforces between inputs and the output.
enum class DtypePolicy : std::uint8_t {
    NONE,        ///< No checks; the kernel handles casting (e.g. Copy).
    ALL_SAME,    ///< All inputs and the output share one dtype.
    INPUT_SAME,  ///< All inputs share one dtype; output unconstrained.
    INPUT_SAME_OUTPUT_BOOL,  ///< Inputs share one dtype; the output is Bool.
};

/// Indexing engine for element-wise ops with broadcasting and for reductions.
///
/// `workload_idx` is a row-major linear element index over `primary_shape_`.
/// Contiguous operands take a linear fast path; general strided operands decode
/// per-dimension coordinates from `primary_strides_`.
///
/// Example:
///   Indexer indexer({a, b}, out, DtypePolicy::ALL_SAME);
///   for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
///       auto* pa = indexer.GetInputPtr<float>(0, i);
///       auto* pb = indexer.GetInputPtr<float>(1, i);
///       auto* po = indexer.GetOutputPtr<float>(i);
///       *po = *pa + *pb;
///   }
class Indexer {
public:
    Indexer() = default;
    Indexer(const Indexer&) = default;
    Indexer& operator=(const Indexer&) = default;

    /// Bind `input_tensors` (broadcast against each other) to a single
    /// `output_tensor`. With non-empty `reduction_dims`, `output_tensor` must
    /// already carry the keepdim reduction shape (size-1 on reduced axes); the
    /// indexer then marks those axes by zeroing the output's stride there.
    Indexer(const std::vector<Tensor>& input_tensors,
            const Tensor& output_tensor,
            DtypePolicy dtype_policy = DtypePolicy::ALL_SAME,
            IntArrayRef reduction_dims = {});

    // ── shape / iteration metadata ──────────────────────────────────────────
    int64_t NumDims() const noexcept { return ndims_; }
    int64_t NumInputs() const noexcept { return num_inputs_; }
    int64_t NumOutputs() const noexcept { return num_outputs_; }

    const int64_t* GetPrimaryShape() const noexcept {
        return primary_shape_.data();
    }
    int64_t* GetPrimaryShape() noexcept { return primary_shape_.data(); }
    const int64_t* GetPrimaryStrides() const noexcept {
        return primary_strides_.data();
    }

    /// Total workloads == product(primary_shape_). For element-wise ops this is
    /// the output element count; for reductions it is the input element count.
    ZT_HOST_DEVICE int64_t NumWorkloads() const {
        int64_t n = 1;
        for (int64_t i = 0; i < ndims_; ++i) {
            n *= primary_shape_[static_cast<std::size_t>(i)];
        }
        return n;
    }

    /// Number of output elements (excludes reduced axes).
    int64_t NumOutputElements() const;
    /// Number of reduced axes (output stride == 0).
    int64_t NumReductionDims() const;

    /// True iff `dim` is a reduced axis (output stride 0 and extent > 1).
    bool IsReductionDim(int64_t dim) const {
        const auto udim = static_cast<std::size_t>(dim);
        return outputs_[0].byte_strides_[udim] == 0 && primary_shape_[udim] > 1;
    }

    // ── operand access ──────────────────────────────────────────────────────
    TensorRef& GetInput(int64_t i) {
        if (i < 0 || i >= num_inputs_) {
            ZT_LOG_ERROR("Indexer::GetInput: 0 <= i < {} required, got {}",
                         num_inputs_,
                         i);
        }
        return inputs_[static_cast<std::size_t>(i)];
    }
    const TensorRef& GetInput(int64_t i) const {
        if (i < 0 || i >= num_inputs_) {
            ZT_LOG_ERROR("Indexer::GetInput: 0 <= i < {} required, got {}",
                         num_inputs_,
                         i);
        }
        return inputs_[static_cast<std::size_t>(i)];
    }
    TensorRef& GetOutput() { return outputs_[0]; }
    const TensorRef& GetOutput() const { return outputs_[0]; }

    // ── per-workload pointer queries (hot path, inline) ─────────────────────

    ZT_HOST_DEVICE char* GetInputPtr(int64_t input_idx,
                                     int64_t workload_idx) const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr(
            inputs_[static_cast<std::size_t>(input_idx)],
            inputs_contiguous_[static_cast<std::size_t>(input_idx)],
            workload_idx);
    }
    template<typename T>
    ZT_HOST_DEVICE T* GetInputPtr(int64_t input_idx,
                                  int64_t workload_idx) const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr<T>(
            inputs_[static_cast<std::size_t>(input_idx)],
            inputs_contiguous_[static_cast<std::size_t>(input_idx)],
            workload_idx);
    }
    ZT_HOST_DEVICE char* GetOutputPtr(int64_t workload_idx) const {
        return GetWorkloadDataPtr(
            outputs_[0], outputs_contiguous_[0], workload_idx);
    }
    template<typename T>
    ZT_HOST_DEVICE T* GetOutputPtr(int64_t workload_idx) const {
        return GetWorkloadDataPtr<T>(
            outputs_[0], outputs_contiguous_[0], workload_idx);
    }

    /// Shrink iteration to [start, start + size) along `dim`. Used by the
    /// parallel-dim reduction strategy to hand each thread a disjoint slice.
    void ShrinkDim(int64_t dim, int64_t start, int64_t size);

protected:
    /// Merge adjacent dims when coalescable (either is size 1, or
    /// shape[n] * stride[n] == stride[n+1] for every operand).
    void CoalesceDimensions();
    /// Permute reduction dims to the front, ordered by stride.
    void ReorderDimensions();
    /// Recompute primary_strides_ from primary_shape_ (row-major).
    void UpdatePrimaryStrides();
    /// Refresh inputs_contiguous_ / outputs_contiguous_.
    void UpdateContiguousFlags();

    /// Broadcast `src` up to `dst_ndims` / `dst_shape`: pad omitted leading
    /// dims with (shape=1, stride=0) and zero the stride on broadcast dims.
    static void BroadcastRestride(TensorRef& src,
                                  int64_t dst_ndims,
                                  const int64_t* dst_shape);
    /// Symmetric to BroadcastRestride for reductions: zero the output stride on
    /// axes where the output is size 1 but the source is not.
    static void ReductionRestride(TensorRef& dst,
                                  int64_t src_ndims,
                                  const int64_t* src_shape);

    /// Pointer to the `workload_idx`-th element of `tr`. Contiguous operands
    /// take the linear fast path; strided operands decode via primary_strides_.
    ZT_HOST_DEVICE char* GetWorkloadDataPtr(const TensorRef& tr,
                                            bool tr_contiguous,
                                            int64_t workload_idx) const {
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<char*>(tr.data_ptr_) +
                   (workload_idx * tr.dtype_byte_size_);
        }
        int64_t offset = 0;
        for (int64_t i = 0; i < ndims_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            offset +=
                (workload_idx / primary_strides_[ui]) * tr.byte_strides_[ui];
            workload_idx = workload_idx % primary_strides_[ui];
        }
        return static_cast<char*>(tr.data_ptr_) + offset;
    }
    template<typename T>
    ZT_HOST_DEVICE T* GetWorkloadDataPtr(const TensorRef& tr,
                                         bool tr_contiguous,
                                         int64_t workload_idx) const {
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<T*>(tr.data_ptr_) + workload_idx;
        }
        int64_t offset = 0;
        for (int64_t i = 0; i < ndims_; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            offset +=
                (workload_idx / primary_strides_[ui]) * tr.byte_strides_[ui];
            workload_idx = workload_idx % primary_strides_[ui];
        }
        return reinterpret_cast<T*>(static_cast<char*>(tr.data_ptr_) + offset);
    }

    int64_t num_inputs_ = 0;
    int64_t num_outputs_ = 0;

    std::array<TensorRef, MAX_INPUTS> inputs_ = {};
    std::array<TensorRef, MAX_OUTPUTS> outputs_ = {};
    std::array<bool, MAX_INPUTS> inputs_contiguous_ = {};
    std::array<bool, MAX_OUTPUTS> outputs_contiguous_ = {};

    /// Global iteration shape. For broadcasting it equals the output shape; for
    /// reductions it equals the input shape (reduced axes carry stride 0).
    std::array<int64_t, MAX_DIMS> primary_shape_ = {};
    /// Default row-major strides for primary_shape_ (internal bookkeeping).
    std::array<int64_t, MAX_DIMS> primary_strides_ = {};

    int64_t ndims_ = 0;
};

}  // namespace zt::core
