// ztensor/kernel/NonZeroCPU.cpp
//
// CPU NonZero: two-pass scan over the bool mask.
//   pass 1: count true elements.
//   pass 2: for each true element, write its per-dim coordinates into the
//           matching output int64 tensor.
// Coordinates are decoded from the linear element index using the mask's
// strides (handles non-contiguous masks). Output order is row-major over the
// mask's logical shape (NumPy semantics).

#include <cstdint>
#include <vector>

#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/NonZero.h"

namespace zt::kernel {

std::vector<Tensor> NonZeroCPU(const Tensor& mask) {
    const int64_t ndim = mask.dim();
    const Device device = mask.device();

    // Pass 1: count true elements. Use a strided Indexer so non-contiguous
    // masks (e.g. a transposed view) are decoded correctly.
    int64_t num_true = 0;
    {
        core::Indexer indexer({mask}, mask, core::DtypePolicy::NONE);
        // The indexer broadcasts inputs against its output shape; here both are
        // the mask, so NumWorkloads == mask.numel().
        for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
            if (*indexer.GetInputPtr<bool>(0, i)) ++num_true;
        }
    }

    // Allocate one {num_true} int64 tensor per dim.
    std::vector<Tensor> out;
    out.reserve(static_cast<std::size_t>(ndim > 0 ? ndim : 1));
    Tensor::ShapeVector one_dim{num_true};
    for (int64_t d = 0; d < std::max<int64_t>(ndim, 1); ++d) {
        out.push_back(zt::empty(IntArrayRef(one_dim.data(), one_dim.size()),
                                zt::dtype(ScalarType::Long).device(device)));
    }
    if (ndim == 0) {
        // 0-d bool tensor: a single true -> one empty coordinate; nothing to
        // write (no dims). The single output tensor has shape {num_true}.
        return out;
    }

    // Pass 2: write coordinates. Walk elements in row-major order; decode the
    // linear index into per-dim coordinates via the mask's shape.
    const auto* shape = mask.sizes().data();
    // Precompute row-major strides over the LOGICAL shape (not the storage
    // strides) for coordinate decoding.
    std::vector<int64_t> coord_strides(static_cast<std::size_t>(ndim), 1);
    for (int64_t i = ndim - 1; i-- > 0;) {
        coord_strides[static_cast<std::size_t>(i)] =
            coord_strides[static_cast<std::size_t>(i + 1)] *
            shape[static_cast<std::size_t>(i + 1)];
    }

    std::vector<int64_t> write_pos(static_cast<std::size_t>(ndim), 0);
    core::Indexer indexer({mask}, mask, core::DtypePolicy::NONE);
    for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
        if (!*indexer.GetInputPtr<bool>(0, i)) continue;
        // Decode linear i into per-dim coordinates.
        int64_t rem = i;
        for (int64_t d = 0; d < ndim; ++d) {
            const auto ud = static_cast<std::size_t>(d);
            const int64_t coord = rem / coord_strides[ud];
            rem -= coord * coord_strides[ud];
            out[ud].data_ptr<int64_t>()[write_pos[ud]] = coord;
            ++write_pos[ud];
        }
    }
    return out;
}

}  // namespace zt::kernel
