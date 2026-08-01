// ztensor/core/AdvancedIndexing.h
//
// Advanced-indexing preprocessor + per-workload gather/scatter indexer. A
// trimmed port of Open3D's open3d/core/AdvancedIndexing.{h,cpp} (SYCL and
// 32-bit-index-splitting machinery dropped).
//
// Given a tensor `t` and a list of int64 index tensors (one per dim, with a
// 0-d int64 scalar marking "this dim is a plain full slice"), the
// AdvancedIndexPreprocessor computes:
//   * the restrided `t` (indexed dims -> stride 0, shape = the broadcast of
//     the index tensors), and the restrided index tensors (padded to
//     broadcast against `t`),
//   * the output shape (the gather result shape),
//   * the indexed shape / indexed strides (the real per-dim extents and
//     element strides of `t`'s indexed axes, used by the kernel to compute
//     gather/scatter offsets).
//
// AdvancedIndexer then wraps the preprocessed tensors + a core::Indexer and
// answers per-workload input/output pointers, unifying GET (gather) and SET
// (scatter) through mode-conditional pointer arithmetic — exactly Open3D's
// design (one code path for both).
//
// Boolean masks must be expanded to int64 (via kernel::NonZero) BEFORE
// constructing the preprocessor; this layer only deals with int64 indices.

#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "ztensor/zt/Tensor.h"

#include "core/cuda/CUDAUtils.h"
#include "core/Indexer.h"

namespace zt::core {

// Maximum number of dimensions the advanced indexer can describe. Matches the
// MAX_DIMS budget of core::Indexer (the gather/scatter kernel delegates to it).
inline constexpr int64_t ADV_INDEX_MAX_DIMS = 8;

/// Compute the restrided tensor, the (padded/broadcast) index tensors, and the
/// indexed shape/strides for an advanced-indexing operation.
///
/// `index_tensors` has exactly `t.dim()` entries (caller pads trailing dims
/// with 0-d int64 sentinels). A 0-d (scalar) entry marks "this dim is a full
/// slice"; a 1+D int64 entry is an advanced index. All index tensors must be
/// int64 and on the same device as `t`.
class AdvancedIndexPreprocessor {
public:
    AdvancedIndexPreprocessor(const Tensor& t,
                              const std::vector<Tensor>& index_tensors);

    const Tensor& GetTensor() const { return tensor_; }
    const std::vector<Tensor>& GetIndexTensors() const {
        return index_tensors_;
    }
    const std::vector<int64_t>& GetOutputShape() const { return output_shape_; }
    const std::vector<int64_t>& GetIndexedShape() const {
        return indexed_shape_;
    }
    const std::vector<int64_t>& GetIndexedStrides() const {
        return indexed_strides_;
    }

private:
    void RunPreprocess();

    // True iff a 0-d (slice) index sits between two non-0-d (advanced) indices
    // along the dim sequence (NumPy's "split" case -> indexed dims go to
    // front).
    static bool IsIndexSplittedBySlice(const std::vector<Tensor>& indices);
    // Permute `t` so indexed dims come first, reordering `indices` to match.
    static std::pair<Tensor, std::vector<Tensor>> ShuffleIndexedDimsToFront(
        const Tensor& t, const std::vector<Tensor>& indices);
    // Broadcast the non-0-d index tensors to a common shape (0-d entries are
    // left untouched).
    static std::pair<std::vector<Tensor>, std::vector<int64_t>>
    ExpandToCommonShapeExceptZeroDim(const std::vector<Tensor>& indices);
    // Restride `t`: erase indexed dims from shape/strides, insert the
    // broadcast (replacement) shape into shape and that many zeros into
    // strides.
    static Tensor RestrideTensor(const Tensor& t,
                                 int64_t dims_before,
                                 int64_t dims_indexed,
                                 const std::vector<int64_t>& replacement_shape);
    // Pad an index tensor with leading size-1 (dims_before) and trailing
    // size-1 (dims_after) dims so it broadcasts against the restrided `t`.
    static Tensor RestrideIndexTensor(const Tensor& idx,
                                      int64_t dims_before,
                                      int64_t dims_after);

    Tensor tensor_;
    std::vector<Tensor> index_tensors_;
    std::vector<int64_t> output_shape_;
    std::vector<int64_t> indexed_shape_;
    std::vector<int64_t> indexed_strides_;
};

/// Per-workload gather/scatter engine. Mode-conditional pointer arithmetic
/// unifies GET (gather: the src pointer advances by the indexed offset) and
/// SET (scatter: the dst pointer advances by the indexed offset).
///
/// Construction builds a core::Indexer over [src, idx0, idx1, ...] (0-d slice
/// indices are NOT passed to the Indexer) and copies indexed_shape/strides
/// into fixed arrays for cheap kernel access.
class AdvancedIndexer {
public:
    enum class Mode { GET, SET };

    AdvancedIndexer(const Tensor& src,
                    const Tensor& dst,
                    const std::vector<Tensor>& index_tensors,
                    const std::vector<int64_t>& indexed_shape,
                    const std::vector<int64_t>& indexed_strides,
                    Mode mode);

    int64_t NumWorkloads() const { return indexer_.NumWorkloads(); }
    // Device of the operands (src/dst share a device by construction).
    const Device& GetDevice() const { return device_; }

    // Pointer to the src element for workload i (GET: advanced by indexed
    // offset = gather; SET: the plain indexer pointer). ZT_HOST_DEVICE so the
    // same indexer drives the CPU ParallelFor and the CUDA grid-stride kernel.
    // Defined inline here so the .cu TU sees the body for device compilation.
    ZT_HOST_DEVICE char* GetInputPtr(int64_t i) const {
        char* base = indexer_.GetInputPtr(0, i);
        if (mode_ == Mode::GET) {
            return base + GetIndexedOffset(i) * element_byte_size_;
        }
        return base;
    }
    ZT_HOST_DEVICE char* GetOutputPtr(int64_t i) const {
        char* base = indexer_.GetOutputPtr(i);
        if (mode_ == Mode::SET) {
            return base + GetIndexedOffset(i) * element_byte_size_;
        }
        return base;
    }

private:
    // Compute the byte offset into `src`/`dst` along the indexed dims, by
    // reading the index values for workload `i` and accumulating
    // index * indexed_strides_[k] (with negative-index wrapping). Device-safe:
    // wraps out-of-range indices instead of throwing (no exceptions on GPU).
    ZT_HOST_DEVICE int64_t GetIndexedOffset(int64_t workload_idx) const {
        int64_t offset = 0;
        for (int64_t k = 0; k < num_indexed_; ++k) {
            const auto uk = static_cast<std::size_t>(k);
            const int64_t index_val =
                *indexer_.GetInputPtr<int64_t>(k + 1, workload_idx);
            int64_t wrapped = index_val;
            if (wrapped < 0) wrapped += indexed_shape_[uk];
            // Clamp on device (no throw); host callers see valid indices by
            // construction so this only masks genuine garbage.
            if (wrapped < 0) wrapped = 0;
            if (wrapped >= indexed_shape_[uk]) wrapped = indexed_shape_[uk] - 1;
            offset += wrapped * indexed_strides_[uk];
        }
        return offset;
    }

    Indexer indexer_;
    Device device_;
    Mode mode_;
    int64_t num_indexed_ = 0;
    std::array<int64_t, static_cast<std::size_t>(ADV_INDEX_MAX_DIMS)>
        indexed_shape_{};
    std::array<int64_t, static_cast<std::size_t>(ADV_INDEX_MAX_DIMS)>
        indexed_strides_{};
    int64_t element_byte_size_ = 0;
};

}  // namespace zt::core
