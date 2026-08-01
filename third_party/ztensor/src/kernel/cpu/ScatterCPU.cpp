// ztensor/kernel/ScatterCPU.cpp
//
// Dim-based gather/scatter (DESIGN §8.6.D). A single int64 `index` tensor
// selects along one axis:
//
//   gather(dim, index):  dst[i]                = src[..., index[i], ...]
//   scatter(dim, index): dst[..., index[i], ...] = src[i]
//
// `index` is contiguous int64, so a linear position `lin` over its row-major
// layout decodes to logical coords once and is reused for both operands. The
// `dim` axis is special: gather reads src at coord index[lin] (bound by
// src.size(dim)); scatter writes dst at coord index[lin] (bound by
// dst.size(dim)). All other axes share the index/dst (gather) or index/src
// (scatter) coordinate, which is valid because the index is no larger than the
// operand along those axes (PyTorch parity).
//
// Accumulation (scatter_add) is restricted to the native-atomic dtypes
// (Float/Double/Int32/Int64) and runs serially — duplicate target indices must
// sum, and without atomics a parallel loop would race.

#include <array>
#include <cstdint>

#include "ztensor/zt/utility/Log.h"

#include "core/AdvancedIndexing.h"  // ADV_INDEX_MAX_DIMS
#include "core/Dispatch.h"
#include "core/ParallelFor.h"
#include "core/ShapeUtil.h"  // WrapDim, DefaultStrides
#include "kernel/Scatter.h"

namespace zt::kernel {
namespace {

// Gather: dst[i] = src[..., index[i], ...]. `index` and `dst` share a shape.
template<typename scalar_t>
void gather_typed(const Tensor& src,
                  const Tensor& index,
                  const Tensor& dst,
                  int64_t dim) {
    const int64_t ndim = index.dim();
    const int64_t dpos = core::WrapDim(dim, ndim);
    const int64_t n = index.numel();
    const auto* idx = index.data_ptr<int64_t>();

    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        cstrides{};
    const auto cstr = core::DefaultStrides(index.sizes());
    std::copy_n(cstr.data(), ndim, cstrides.data());

    const int64_t* src_st = src.strides().data();
    const int64_t* dst_st = dst.strides().data();
    const int64_t src_dim_size = src.size(dpos);

    const auto* src_p = static_cast<const scalar_t*>(src.data_ptr());
    auto* dst_p = static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));

    core::ParallelFor(dst.device(), n, [&](int64_t lin) {
        int64_t src_off = 0;
        int64_t dst_off = 0;
        int64_t rem = lin;
        for (int64_t d = 0; d < ndim; ++d) {
            const int64_t c = rem / cstrides[d];
            rem %= cstrides[d];
            if (d != dpos) {
                src_off += c * src_st[d];
            }
            dst_off += c * dst_st[d];
        }
        int64_t iv = idx[lin];
        if (iv < 0) {
            iv += src_dim_size;
        }
        src_off += iv * src_st[dpos];
        dst_p[dst_off] = src_p[src_off];
    });
}

// Scatter (overwrite): dst[..., index[i], ...] = src[i].
template<typename scalar_t>
void scatter_typed(const Tensor& src,
                   const Tensor& index,
                   const Tensor& dst,
                   int64_t dim) {
    const int64_t ndim = index.dim();
    const int64_t dpos = core::WrapDim(dim, ndim);
    const int64_t n = index.numel();
    const auto* idx = index.data_ptr<int64_t>();

    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        cstrides{};
    const auto cstr = core::DefaultStrides(index.sizes());
    std::copy_n(cstr.data(), ndim, cstrides.data());

    const int64_t* src_st = src.strides().data();
    const int64_t* dst_st = dst.strides().data();
    const int64_t dst_dim_size = dst.size(dpos);

    const auto* src_p = static_cast<const scalar_t*>(src.data_ptr());
    auto* dst_p = static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));

    core::ParallelFor(dst.device(), n, [&](int64_t lin) {
        int64_t src_off = 0;
        int64_t dst_off = 0;
        int64_t rem = lin;
        for (int64_t d = 0; d < ndim; ++d) {
            const int64_t c = rem / cstrides[d];
            rem %= cstrides[d];
            if (d != dpos) {
                dst_off += c * dst_st[d];
            }
            src_off += c * src_st[d];
        }
        int64_t iv = idx[lin];
        if (iv < 0) {
            iv += dst_dim_size;
        }
        dst_off += iv * dst_st[dpos];
        dst_p[dst_off] = src_p[src_off];
    });
}

// Scatter (accumulate): dst[..., index[i], ...] += src[i]. Serial: duplicate
// indices must sum and there are no CPU atomics here. (v1; a `#pragma omp
// atomic` parallel path is a future optimization.)
template<typename scalar_t>
void scatter_add_typed(const Tensor& src,
                       const Tensor& index,
                       const Tensor& dst,
                       int64_t dim) {
    const int64_t ndim = index.dim();
    const int64_t dpos = core::WrapDim(dim, ndim);
    const int64_t n = index.numel();
    const auto* idx = index.data_ptr<int64_t>();

    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        cstrides{};
    const auto cstr = core::DefaultStrides(index.sizes());
    std::copy_n(cstr.data(), ndim, cstrides.data());

    const int64_t* src_st = src.strides().data();
    const int64_t* dst_st = dst.strides().data();
    const int64_t dst_dim_size = dst.size(dpos);

    const auto* src_p = static_cast<const scalar_t*>(src.data_ptr());
    auto* dst_p = static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));

    for (int64_t lin = 0; lin < n; ++lin) {
        int64_t src_off = 0;
        int64_t dst_off = 0;
        int64_t rem = lin;
        for (int64_t d = 0; d < ndim; ++d) {
            const int64_t c = rem / cstrides[d];
            rem %= cstrides[d];
            if (d != dpos) {
                dst_off += c * dst_st[d];
            }
            src_off += c * src_st[d];
        }
        int64_t iv = idx[lin];
        if (iv < 0) {
            iv += dst_dim_size;
        }
        dst_off += iv * dst_st[dpos];
        dst_p[dst_off] += src_p[src_off];
    }
}

}  // namespace

void GatherCPU(const Tensor& src,
               const Tensor& index,
               const Tensor& dst,
               int64_t dim) {
    ZT_CHECK(index.scalar_type() == ScalarType::Long,
             "gather: index must be int64");
    ZT_CHECK(index.is_contiguous(), "gather: index must be contiguous");
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        gather_typed<scalar_t>(src, index, dst, dim);
    });
}

void ScatterCPU(const Tensor& src,
                const Tensor& index,
                const Tensor& dst,
                int64_t dim,
                bool accumulate) {
    ZT_CHECK(index.scalar_type() == ScalarType::Long,
             "scatter: index must be int64");
    ZT_CHECK(index.is_contiguous(), "scatter: index must be contiguous");
    if (!accumulate) {
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
            scatter_typed<scalar_t>(src, index, dst, dim);
        });
        return;
    }
    // Accumulation is restricted to the native-atomic dtypes (Float/Double/
    // Int32/Int64), matching kernel::IndexAdd / Reduction's atomic_supported.
    const auto dt = src.scalar_type();
    if (dt == ScalarType::Float) {
        scatter_add_typed<float>(src, index, dst, dim);
    } else if (dt == ScalarType::Double) {
        scatter_add_typed<double>(src, index, dst, dim);
    } else if (dt == ScalarType::Int) {
        scatter_add_typed<std::int32_t>(src, index, dst, dim);
    } else if (dt == ScalarType::Long) {
        scatter_add_typed<std::int64_t>(src, index, dst, dim);
    } else {
        ZT_LOG_ERROR(
            "scatter_add: dtype {} not supported (use "
            "Float/Double/Int32/Int64)",
            toString(dt));
    }
}

}  // namespace zt::kernel
