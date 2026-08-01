// ztensor/kernel/ScatterCUDA.cu
//
// CUDA dim-based gather/scatter (DESIGN §8.6.D). Mirrors ScatterCPU: decode
// each linear index position to logical coords once, then remap the `dim`
// coordinate to the index value. The decode strides / operand strides are
// copied into fixed-size std::array captured by value into the __device__
// lambda (POD, so it lives in device local memory). The AdvancedIndexer is not
// used here — a single int64 index along one axis is simpler to handle
// directly than to route through the multi-key preprocessor.
//
// scatter_add uses atomicAdd, restricted to the native-atomic dtypes
// (int32/int64/float/double), matching IndexAddCUDA / ReductionCUDA.

#include <array>
#include <cstdint>

#include "ztensor/zt/utility/Log.h"

#include "core/AdvancedIndexing.h"  // ADV_INDEX_MAX_DIMS
#include "ztensor/zt/cuda/Guard.h"
#include "core/Dispatch.h"
#include "core/ParallelFor.h"
#include "core/ShapeUtil.h"  // WrapDim, DefaultStrides
#include "kernel/Scatter.h"

namespace zt {
namespace kernel {
namespace {

// atomic_add for the native-atomic dtypes (same set as IndexAddCUDA /
// ReductionCUDA's atomic_supported).
__device__ inline void scatter_atomic_add(std::int32_t* addr, std::int32_t v) {
    atomicAdd(addr, v);
}
__device__ inline void scatter_atomic_add(std::int64_t* addr, std::int64_t v) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(v));
}
__device__ inline void scatter_atomic_add(float* addr, float v) {
    atomicAdd(addr, v);
}
__device__ inline void scatter_atomic_add(double* addr, double v) {
    atomicAdd(addr, v);
}

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

    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        src_st{};
    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        dst_st{};
    std::copy_n(src.strides().data(), ndim, src_st.data());
    std::copy_n(dst.strides().data(), ndim, dst_st.data());
    const int64_t src_dim_size = src.size(dpos);

    const auto* src_p = static_cast<const scalar_t*>(src.data_ptr());
    auto* dst_p = static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));

    core::ParallelFor(dst.device(), n, [=] __device__(int64_t lin) {
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

    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        src_st{};
    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        dst_st{};
    std::copy_n(src.strides().data(), ndim, src_st.data());
    std::copy_n(dst.strides().data(), ndim, dst_st.data());
    const int64_t dst_dim_size = dst.size(dpos);

    const auto* src_p = static_cast<const scalar_t*>(src.data_ptr());
    auto* dst_p = static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));

    core::ParallelFor(dst.device(), n, [=] __device__(int64_t lin) {
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

    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        src_st{};
    std::array<int64_t, static_cast<std::size_t>(core::ADV_INDEX_MAX_DIMS)>
        dst_st{};
    std::copy_n(src.strides().data(), ndim, src_st.data());
    std::copy_n(dst.strides().data(), ndim, dst_st.data());
    const int64_t dst_dim_size = dst.size(dpos);

    const auto* src_p = static_cast<const scalar_t*>(src.data_ptr());
    auto* dst_p = static_cast<scalar_t*>(const_cast<void*>(dst.data_ptr()));

    core::ParallelFor(dst.device(), n, [=] __device__(int64_t lin) {
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
        scatter_atomic_add(dst_p + dst_off, src_p[src_off]);
    });
}

}  // namespace

void GatherCUDA(const Tensor& src,
                const Tensor& index,
                const Tensor& dst,
                int64_t dim) {
    CUDAScopedDevice scoped(dst.device());
    ZT_CHECK(index.scalar_type() == ScalarType::Long,
             "gather: index must be int64");
    ZT_CHECK(index.is_contiguous(), "gather: index must be contiguous");
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        gather_typed<scalar_t>(src, index, dst, dim);
    });
}

void ScatterCUDA(const Tensor& src,
                 const Tensor& index,
                 const Tensor& dst,
                 int64_t dim,
                 bool accumulate) {
    CUDAScopedDevice scoped(dst.device());
    ZT_CHECK(index.scalar_type() == ScalarType::Long,
             "scatter: index must be int64");
    ZT_CHECK(index.is_contiguous(), "scatter: index must be contiguous");
    if (!accumulate) {
        ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
            scatter_typed<scalar_t>(src, index, dst, dim);
        });
        return;
    }
    // Accumulation: same dtype scope as ScatterCPU / IndexAdd (native
    // atomicAdd).
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

}  // namespace kernel
}  // namespace zt
