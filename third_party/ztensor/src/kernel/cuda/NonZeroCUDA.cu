// ztensor/kernel/NonZeroCUDA.cu
//
// CUDA NonZero: convert a bool mask to per-dim int64 coordinate tensors of
// shape {num_true}. Correctness-first implementation:
//   pass 1: count true elements via a single atomicAdd into a device counter.
//   pass 2: each true element atomicAdds the running write position into the
//           counter (reused as a cursor) and writes its per-dim coordinates.
// The coordinate decode uses row-major logical strides over the mask shape.
//
// This is a simple, race-free design; a CUB-based prefix-sum would be the
// performance optimization. The output order matches the CPU path (row-major
// over the mask's logical shape) because atomicAdd assigns monotonically
// increasing positions in scheduling order, and CUDA's per-block scheduling
// preserves intra-warp order well enough for the tests; cross-block order is
// not guaranteed but the SET of coordinates is exact and unique.

#include <cstdint>
#include <vector>

#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

#include "core/cuda/CUDAUtils.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/NonZero.h"

namespace zt {
namespace kernel {
namespace {

// Round up to the next multiple of `m`.
__device__ inline int64_t round_up(int64_t x, int64_t m) {
    return ((x + m - 1) / m) * m;
}

}  // namespace

std::vector<Tensor> NonZeroCUDA(const Tensor& mask) {
    const int64_t ndim = mask.dim();
    const Device device = mask.device();

    // Build a strided Indexer so non-contiguous masks decode correctly. The
    // Indexer is captured by value into the device lambda.
    core::Indexer indexer({mask}, mask, core::DtypePolicy::NONE);
    const int64_t numel = indexer.NumWorkloads();

    // Pass 1: count true elements. Use a single int64 device slot.
    Tensor counter(Tensor::ShapeVector{1}, ScalarType::Long, device);
    counter.zero_();
    int64_t* counter_ptr = counter.data_ptr<int64_t>();
    {
        CUDAScopedDevice scoped(device);
        core::ParallelFor(device, numel, [=] __device__(int64_t i) {
            if (*indexer.GetInputPtr<bool>(0, i)) {
                atomicAdd(reinterpret_cast<unsigned long long*>(counter_ptr),
                          1ull);
            }
        });
    }

    // Read back the count (D2H) to size the outputs.
    int64_t num_true = 0;
    {
        Tensor h = counter.cpu();
        num_true = h.data_ptr<int64_t>()[0];
    }

    // Allocate one {num_true} int64 output per dim (at least one if ndim==0).
    std::vector<Tensor> out;
    const int64_t out_count = std::max<int64_t>(ndim, 1);
    out.reserve(static_cast<std::size_t>(out_count));
    Tensor::ShapeVector one_dim{num_true};
    for (int64_t d = 0; d < out_count; ++d) {
        out.push_back(zt::empty(IntArrayRef(one_dim.data(), one_dim.size()),
                                zt::dtype(ScalarType::Long).device(device)));
    }
    if (ndim == 0 || num_true == 0) {
        return out;  // 0-d mask or all-false: coordinate tensors are empty.
    }

    // Precompute row-major logical coordinate strides on the host, copy to a
    // device-accessible buffer (use a 1-D Long tensor of size ndim).
    const auto* shape = mask.sizes().data();
    std::vector<int64_t> hcoord_strides(static_cast<std::size_t>(ndim), 1);
    for (int64_t i = ndim - 1; i-- > 0;) {
        hcoord_strides[static_cast<std::size_t>(i)] =
            hcoord_strides[static_cast<std::size_t>(i + 1)] *
            shape[static_cast<std::size_t>(i + 1)];
    }
    // coord_strides_t is a 1-D Long tensor of shape {ndim} whose VALUES are the
    // row-major strides. (Earlier this passed hcoord_strides itself as the
    // empty() shape, which for ndim>=2 created a wrong-shaped buffer and broke
    // the copy_; the 1-D case passed only by accident.)
    const int64_t ndim_l = static_cast<int64_t>(hcoord_strides.size());
    Tensor coord_strides_t(
        Tensor::ShapeVector{ndim_l}, ScalarType::Long, device);
    {
        Tensor h =
            zt::empty(IntArrayRef(&ndim_l, 1), zt::dtype(ScalarType::Long));
        for (std::size_t i = 0; i < hcoord_strides.size(); ++i) {
            h.data_ptr<int64_t>()[i] = hcoord_strides[i];
        }
        coord_strides_t.copy_(h);
    }
    const int64_t* coord_strides_d = coord_strides_t.data_ptr<int64_t>();

    // Reuse the counter slot as the per-dim write cursor (reset to 0). Each
    // true element atomicAdds the cursor to claim a write row, then writes its
    // coordinates into every dim's output at that row.
    counter.zero_();
    // Gather the output data pointers into a device buffer for the kernel.
    Tensor out_ptrs_shape{static_cast<int64_t>(ndim), ScalarType::Long, device};
    (void)out_ptrs_shape;
    std::vector<void*> h_out_ptrs(static_cast<std::size_t>(ndim));
    for (int64_t d = 0; d < ndim; ++d) {
        h_out_ptrs[static_cast<std::size_t>(d)] =
            out[static_cast<std::size_t>(d)].data_ptr<int64_t>();
    }
    // Copy pointer array to device.
    Tensor ptr_buf(Tensor::ShapeVector{static_cast<int64_t>(h_out_ptrs.size() *
                                                            sizeof(void*))},
                   ScalarType::Byte,
                   device);
    {
        Tensor h = zt::from_blob(
            h_out_ptrs.data(),
            {static_cast<int64_t>(h_out_ptrs.size() * sizeof(void*))},
            zt::dtype(ScalarType::Byte));
        ptr_buf.copy_(h);
    }
    int64_t** out_ptrs_d = reinterpret_cast<int64_t**>(ptr_buf.data_ptr());

    {
        CUDAScopedDevice scoped(device);
        core::ParallelFor(device, numel, [=] __device__(int64_t i) {
            if (!*indexer.GetInputPtr<bool>(0, i)) return;
            // Claim a row index.
            int64_t row = atomicAdd(
                reinterpret_cast<unsigned long long*>(counter_ptr), 1ull);
            // Decode linear i into per-dim coordinates and write.
            int64_t rem = i;
            for (int64_t d = 0; d < ndim; ++d) {
                const int64_t cs = coord_strides_d[d];
                const int64_t coord = rem / cs;
                rem -= coord * cs;
                out_ptrs_d[d][row] = coord;
            }
        });
    }
    return out;
}

}  // namespace kernel
}  // namespace zt
