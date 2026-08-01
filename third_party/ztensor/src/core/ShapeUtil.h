// ztensor/core/ShapeUtil.h
//
// Host-side shape arithmetic: dim wrapping, default (row-major) strides, shape
// broadcasting, and reduction-shape computation. Ported from Open3D's
// open3d/core/ShapeUtil.{h,cpp}, restricted to the helpers the Indexer and the
// op dispatchers need. Header-only (small, host-only).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ztensor/zt/ArrayRef.h"
#include "ztensor/zt/utility/Log.h"

namespace zt::core {

// Wrap a (possibly negative) dim index into [0, max_dim). Throws on
// out-of-range. Mirrors Open3D shape_util::WrapDim.
inline int64_t WrapDim(int64_t dim, int64_t max_dim) {
    if (max_dim <= 0) {
        ZT_LOG_ERROR("WrapDim: max_dim {} must be > 0", max_dim);
    }
    const int64_t min = -max_dim;
    const int64_t max = max_dim - 1;
    if (dim < min || dim > max) {
        ZT_LOG_ERROR("WrapDim: dim {} out of range for ndim {}", dim, max_dim);
    }
    if (dim < 0) {
        dim += max_dim;
    }
    return dim;
}

// Row-major (C-contiguous) strides for `shape`, in elements. Matches the
// convention used by Tensor's own contiguous-stride helper.
inline std::vector<int64_t> DefaultStrides(IntArrayRef shape) {
    const std::size_t ndim = shape.size();
    std::vector<int64_t> strides(ndim, 0);
    if (ndim == 0) {
        return strides;
    }
    strides[ndim - 1] = 1;
    for (std::size_t i = ndim - 1; i-- > 0;) {
        const int64_t next = shape[i + 1] > 0 ? shape[i + 1] : 1;
        strides[i] = strides[i + 1] * next;
    }
    return strides;
}

// NumPy-style broadcast of two shapes. Throws if the shapes are not compatible
// (a dim where both differ and neither is 1).
inline std::vector<int64_t> BroadcastedShape(IntArrayRef lhs, IntArrayRef rhs) {
    const std::size_t nl = lhs.size();
    const std::size_t nr = rhs.size();
    const std::size_t no = std::max(nl, nr);
    std::vector<int64_t> out(no);
    for (std::size_t i = 0; i < no; ++i) {
        const int64_t a = (i + nl < no) ? 1 : lhs[i - (no - nl)];
        const int64_t b = (i + nr < no) ? 1 : rhs[i - (no - nr)];
        if (a == 1) {
            out[i] = b;
        } else if (b == 1 || a == b) {
            out[i] = a;
        } else {
            ZT_LOG_ERROR(
                "BroadcastedShape: incompatible shapes at dim {} ({} vs {})",
                i,
                a,
                b);
        }
    }
    return out;
}

// Output shape of reducing `src` along `dims`. With keepdim=true the reduced
// axes become size 1; with keepdim=false they are dropped. Dims are wrapped and
// must not repeat.
inline std::vector<int64_t> ReductionShape(IntArrayRef src,
                                           IntArrayRef dims,
                                           bool keepdim) {
    const auto ndim = static_cast<int64_t>(src.size());
    if (keepdim) {
        std::vector<int64_t> out(src.begin(), src.end());
        for (const int64_t d : dims) {
            out[WrapDim(d, ndim)] = 1;
        }
        return out;
    }
    std::vector<bool> reduced(ndim, false);
    for (const int64_t d : dims) {
        const auto w = WrapDim(d, ndim);
        ZT_CHECK(!reduced[w], "ReductionShape: dim {} reduced twice", w);
        reduced[w] = true;
    }
    std::vector<int64_t> out;
    out.reserve(static_cast<std::size_t>(
        ndim > 0 ? ndim - static_cast<int64_t>(dims.size()) : 0));
    for (int64_t i = 0; i < ndim; ++i) {
        if (!reduced[i]) {
            out.push_back(src[i]);
        }
    }
    return out;
}

}  // namespace zt::core
