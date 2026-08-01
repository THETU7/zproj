// ztensor/factory/TensorFactories.cpp
//
// Tensor factory functions: empty / zeros / ones / full / eye / arange /
// from_blob / *_like. Modeled on PyTorch's TensorFactories.cpp, restricted to
// the CPU path (CUDA arrives in phase 4).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/utility/Log.h"

#include "core/MemoryManager.h"
#include "core/ShapeUtil.h"
#include "kernel/Rand.h"

namespace zt {
namespace {

// Resolve a TensorOptions against defaults the way PyTorch does: take the
// explicitly-set fields of `options`, fall back to Float / CPU.
TensorOptions resolved_options(TensorOptions options) {
    if (!options.has_dtype()) options = options.dtype(ScalarType::Float);
    if (!options.has_device()) options = options.device(Device(kCPU));
    if (!options.has_layout()) options = options.layout(Layout::Strided);
    return options;
}

}  // namespace

// ---- empty ----------------------------------------------------------------
Tensor empty(IntArrayRef size, TensorOptions options) {
    options = resolved_options(options);
    Tensor::ShapeVector shape(size.begin(), size.end());
    return Tensor(shape, options.dtype(), options.device());
}

Tensor empty_like(const Tensor& self, TensorOptions options) {
    if (!options.has_dtype()) options = options.dtype(self.scalar_type());
    if (!options.has_device()) options = options.device(self.device());
    return empty(self.sizes(), options);
}

// ---- zeros / ones / full --------------------------------------------------
Tensor zeros(IntArrayRef size, TensorOptions options) {
    Tensor t = empty(size, options);
    t.zero_();
    return t;
}
Tensor ones(IntArrayRef size, TensorOptions options) {
    Tensor t = empty(size, options);
    t.fill_(Scalar(1));
    return t;
}
Tensor full(IntArrayRef size, Scalar fill_value, TensorOptions options) {
    Tensor t = empty(size, options);
    t.fill_(fill_value);
    return t;
}

Tensor zeros_like(const Tensor& self, TensorOptions options) {
    if (!options.has_dtype()) options = options.dtype(self.scalar_type());
    if (!options.has_device()) options = options.device(self.device());
    return zeros(self.sizes(), options);
}
Tensor ones_like(const Tensor& self, TensorOptions options) {
    if (!options.has_dtype()) options = options.dtype(self.scalar_type());
    if (!options.has_device()) options = options.device(self.device());
    return ones(self.sizes(), options);
}
Tensor full_like(const Tensor& self, Scalar fill_value, TensorOptions options) {
    if (!options.has_dtype()) options = options.dtype(self.scalar_type());
    if (!options.has_device()) options = options.device(self.device());
    return full(self.sizes(), fill_value, options);
}

// ---- arange ---------------------------------------------------------------
// Defined first; the (end, options) overload forwards to it. We do NOT give
// `step` a default value here (only in the header), so the two arange
// overloads are unambiguous at the call site.
//
// The sequence is materialized on CPU (host pointer writes) and then moved to
// the requested device via Tensor::to(): correct for any device, and avoids a
// per-backend arange kernel. The transfer is a single coarse Memcpy when the
// target is CUDA.
Tensor arange(Scalar start, Scalar end, Scalar step, TensorOptions options) {
    const Device dst_device =
        options.has_device() ? options.device() : Device(kCPU);
    options = resolved_options(options);
    ZT_CHECK(step.toDouble() != 0.0, "arange: step must be non-zero");
    const double s = start.toDouble();
    const double e = end.toDouble();
    const double k = step.toDouble();
    const int64_t n = static_cast<int64_t>(std::ceil((e - s) / k));
    const int64_t count = n > 0 ? n : 0;
    Tensor::ShapeVector shape{count};
    // Always build on CPU first; move below.
    Tensor t(shape, options.dtype(), Device(kCPU));
    if (count == 0) return t.to(dst_device, /*copy=*/false);
    auto fill_arr = [&](auto* base) {
        using T = typename std::remove_pointer<decltype(base)>::type;
        for (int64_t i = 0; i < count; ++i) {
            base[i] = static_cast<T>(s + static_cast<double>(i) * k);
        }
    };
    switch (options.dtype()) {
        case ScalarType::Bool: {
            auto* p = t.data_ptr<bool>();
            for (int64_t i = 0; i < count; ++i)
                p[i] = static_cast<bool>(s + static_cast<double>(i) * k);
            break;
        }
        case ScalarType::Byte:
            fill_arr(t.data_ptr<std::uint8_t>());
            break;
        case ScalarType::Char:
            fill_arr(t.data_ptr<std::int8_t>());
            break;
        case ScalarType::Short:
            fill_arr(t.data_ptr<std::int16_t>());
            break;
        case ScalarType::Int:
            fill_arr(t.data_ptr<std::int32_t>());
            break;
        case ScalarType::Long:
            fill_arr(t.data_ptr<std::int64_t>());
            break;
        case ScalarType::Float:
            fill_arr(t.data_ptr<float>());
            break;
        case ScalarType::Double:
            fill_arr(t.data_ptr<double>());
            break;
        default:
            ZT_LOG_ERROR("arange: unsupported dtype {}",
                         toString(options.dtype()));
    }
    return t.to(dst_device, /*copy=*/false);
}

Tensor arange(Scalar end, TensorOptions options) {
    options = resolved_options(options);
    return arange(Scalar(0), end, Scalar(1), options);
}

// ---- eye ------------------------------------------------------------------
// Built on CPU (host pointer writes for the diagonal) then moved to the
// requested device. Correct for any device without a per-backend kernel.
Tensor eye(int64_t n, TensorOptions options) {
    const Device dst_device =
        options.has_device() ? options.device() : Device(kCPU);
    options = resolved_options(options);
    Tensor::ShapeVector shape{n, n};
    Tensor t(shape, options.dtype(), Device(kCPU));
    t.zero_();
    // Set the diagonal to 1 (handles any dtype via Scalar dispatch).
    const std::size_t esize = t.element_size();
    const int64_t d = std::min(t.size(0), t.size(1));
    auto set_one = [&](void* p) {
        switch (options.dtype()) {
            case ScalarType::Bool:
                *static_cast<bool*>(p) = true;
                break;
            case ScalarType::Byte:
                *static_cast<std::uint8_t*>(p) = 1;
                break;
            case ScalarType::Char:
                *static_cast<std::int8_t*>(p) = 1;
                break;
            case ScalarType::Short:
                *static_cast<std::int16_t*>(p) = 1;
                break;
            case ScalarType::Int:
                *static_cast<std::int32_t*>(p) = 1;
                break;
            case ScalarType::Long:
                *static_cast<std::int64_t*>(p) = 1;
                break;
            case ScalarType::Float:
                *static_cast<float*>(p) = 1.0f;
                break;
            case ScalarType::Double:
                *static_cast<double*>(p) = 1.0;
                break;
            default:
                ZT_LOG_ERROR("eye: unsupported dtype {}",
                             toString(options.dtype()));
        }
    };
    for (int64_t i = 0; i < d; ++i) {
        auto* p = static_cast<std::uint8_t*>(t.data_ptr()) +
                  (i * t.stride(0) + i * t.stride(1)) * esize;
        set_one(p);
    }
    return t.to(dst_device, /*copy=*/false);
}

// ---- linspace / logspace (DESIGN §8.6.C) ----------------------------------
// Same pattern as arange: build the sequence on the CPU via host pointer
// writes, then move to the requested device in one coarse Memcpy.

// 4-arg form with explicit `endpoint`. The 3-arg form (declared in the header
// with `endpoint` defaulted to true) forwards to this.
Tensor linspace(Scalar start,
                Scalar end,
                int64_t steps,
                bool endpoint,
                TensorOptions options) {
    const Device dst_device =
        options.has_device() ? options.device() : Device(kCPU);
    options = resolved_options(options);
    ZT_CHECK(steps >= 0, "linspace: number of steps must be non-negative");
    Tensor::ShapeVector shape{steps};
    Tensor t(shape, options.dtype(), Device(kCPU));
    if (steps == 0) return t.to(dst_device, /*copy=*/false);
    const double s = start.toDouble();
    const double e = end.toDouble();
    // With endpoint the gap is (e - s)/(steps-1); without it (e - s)/steps.
    // Special-case steps == 1 to avoid division by zero (single value = start).
    const double denom = (steps == 1)
                             ? 1.0
                             : (endpoint ? static_cast<double>(steps - 1)
                                         : static_cast<double>(steps));
    const double step = (e - s) / denom;
    auto fill_arr = [&](auto* base) {
        using T = typename std::remove_pointer<decltype(base)>::type;
        for (int64_t i = 0; i < steps; ++i) {
            base[i] = static_cast<T>(s + static_cast<double>(i) * step);
        }
    };
    switch (options.dtype()) {
        case ScalarType::Bool: {
            auto* p = t.data_ptr<bool>();
            for (int64_t i = 0; i < steps; ++i)
                p[i] = static_cast<bool>(s + static_cast<double>(i) * step);
            break;
        }
        case ScalarType::Byte:
            fill_arr(t.data_ptr<std::uint8_t>());
            break;
        case ScalarType::Char:
            fill_arr(t.data_ptr<std::int8_t>());
            break;
        case ScalarType::Short:
            fill_arr(t.data_ptr<std::int16_t>());
            break;
        case ScalarType::Int:
            fill_arr(t.data_ptr<std::int32_t>());
            break;
        case ScalarType::Long:
            fill_arr(t.data_ptr<std::int64_t>());
            break;
        case ScalarType::Float:
            fill_arr(t.data_ptr<float>());
            break;
        case ScalarType::Double:
            fill_arr(t.data_ptr<double>());
            break;
        default:
            ZT_LOG_ERROR("linspace: unsupported dtype {}",
                         toString(options.dtype()));
    }
    return t.to(dst_device, /*copy=*/false);
}

Tensor linspace(Scalar start,
                Scalar end,
                int64_t steps,
                TensorOptions options) {
    return linspace(start, end, steps, /*endpoint=*/true, options);
}

Tensor logspace(Scalar start,
                Scalar end,
                int64_t steps,
                double base,
                TensorOptions options) {
    const Device dst_device =
        options.has_device() ? options.device() : Device(kCPU);
    options = resolved_options(options);
    ZT_CHECK(steps >= 0, "logspace: number of steps must be non-negative");
    ZT_CHECK(base > 0.0, "logspace: base must be positive");
    Tensor::ShapeVector shape{steps};
    Tensor t(shape, options.dtype(), Device(kCPU));
    if (steps == 0) return t.to(dst_device, /*copy=*/false);
    const double s = start.toDouble();
    const double e = end.toDouble();
    // logspace mirrors linspace with endpoint=true: the exponent runs from
    // `start` to `end` inclusive, then each entry is base ** exponent.
    const double denom = (steps == 1) ? 1.0 : static_cast<double>(steps - 1);
    const double step = (e - s) / denom;
    auto fill_arr = [&](auto* base_ptr) {
        using T = typename std::remove_pointer<decltype(base_ptr)>::type;
        for (int64_t i = 0; i < steps; ++i) {
            const double exponent = s + static_cast<double>(i) * step;
            base_ptr[i] = static_cast<T>(std::pow(base, exponent));
        }
    };
    switch (options.dtype()) {
        case ScalarType::Bool: {
            auto* p = t.data_ptr<bool>();
            for (int64_t i = 0; i < steps; ++i) {
                const double exponent = s + static_cast<double>(i) * step;
                p[i] = static_cast<bool>(std::pow(base, exponent));
            }
            break;
        }
        case ScalarType::Byte:
            fill_arr(t.data_ptr<std::uint8_t>());
            break;
        case ScalarType::Char:
            fill_arr(t.data_ptr<std::int8_t>());
            break;
        case ScalarType::Short:
            fill_arr(t.data_ptr<std::int16_t>());
            break;
        case ScalarType::Int:
            fill_arr(t.data_ptr<std::int32_t>());
            break;
        case ScalarType::Long:
            fill_arr(t.data_ptr<std::int64_t>());
            break;
        case ScalarType::Float:
            fill_arr(t.data_ptr<float>());
            break;
        case ScalarType::Double:
            fill_arr(t.data_ptr<double>());
            break;
        default:
            ZT_LOG_ERROR("logspace: unsupported dtype {}",
                         toString(options.dtype()));
    }
    return t.to(dst_device, /*copy=*/false);
}

// ---- from_blob ------------------------------------------------------------
// Most general overload: defined first so the 3- and 4-arg convenience
// overloads can forward to it unambiguously.
Tensor from_blob(void* data,
                 IntArrayRef sizes,
                 IntArrayRef strides,
                 std::function<void(void*)> deleter,
                 TensorOptions options) {
    options = resolved_options(options);
    Tensor::ShapeVector shape(sizes.begin(), sizes.end());
    Tensor::ShapeVector str(strides.begin(), strides.end());
    // When the caller passes no deleter, the Blob must NOT free the buffer
    // (external memory whose lifetime is managed elsewhere). We install a
    // no-op deleter to distinguish this case from an internally-allocated
    // Blob (whose deleter_ stays null and is freed via MemoryManager::Free).
    if (!deleter) {
        deleter = [](void*) { /* external memory: do nothing */ };
    }
    auto blob = std::make_shared<Blob>(options.device(), data, deleter);
    return Tensor(shape, str, data, options.dtype(), blob);
}

Tensor from_blob(void* data,
                 IntArrayRef sizes,
                 IntArrayRef strides,
                 TensorOptions options) {
    return from_blob(
        data, sizes, strides, std::function<void(void*)>{nullptr}, options);
}

Tensor from_blob(void* data, IntArrayRef sizes, TensorOptions options) {
    options = resolved_options(options);
    Tensor::ShapeVector shape(sizes.begin(), sizes.end());
    auto strides = Tensor::ShapeVector(shape.size(), 1);
    // Compute row-major strides.
    if (!shape.empty()) {
        strides[shape.size() - 1] = 1;
        for (std::size_t i = shape.size() - 1; i-- > 0;) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return from_blob(data,
                     sizes,
                     IntArrayRef(strides.data(), strides.size()),
                     std::function<void(void*)>{nullptr},
                     options);
}

// ---- linear algebra (DESIGN §8.5.A) -----------------------------------------
// Thin delegates to the Tensor member methods; they exist so call sites can
// use the free-function PyTorch form (`zt::mm(a, b)`) as well as the method
// form (`a.mm(b)`).

Tensor mm(const Tensor& self, const Tensor& other) { return self.mm(other); }
Tensor bmm(const Tensor& self, const Tensor& other) { return self.bmm(other); }
Tensor matmul(const Tensor& self, const Tensor& other) {
    return self.matmul(other);
}
Tensor addmm(const Tensor& C,
             const Tensor& A,
             const Tensor& B,
             Scalar beta,
             Scalar alpha) {
    return C.addmm(A, B, beta, alpha);
}

// ---- random (DESIGN §8.5.C) -------------------------------------------------

Tensor rand(IntArrayRef size, TensorOptions options, const Generator& gen) {
    options = resolved_options(options);
    Tensor t = empty(size, options);
    kernel::Rand(t, /*from=*/0.0, /*to=*/1.0, gen);
    return t;
}

Tensor randn(IntArrayRef size, TensorOptions options, const Generator& gen) {
    options = resolved_options(options);
    Tensor t = empty(size, options);
    kernel::RandN(t, /*mean=*/0.0, /*stddev=*/1.0, gen);
    return t;
}

Tensor randint(int64_t low,
               int64_t high,
               IntArrayRef size,
               TensorOptions options,
               const Generator& gen) {
    // Default to Long if no dtype specified (PyTorch convention).
    if (!options.has_dtype()) options = options.dtype(ScalarType::Long);
    options = resolved_options(options);
    Tensor t = empty(size, options);
    kernel::RandInt(t, low, high, gen);
    return t;
}

Tensor& uniform_(Tensor& self, double from, double to, const Generator& gen) {
    kernel::Uniform_(self, from, to, gen);
    return self;
}

Tensor& normal_(Tensor& self,
                double mean,
                double stddev,
                const Generator& gen) {
    kernel::Normal_(self, mean, stddev, gen);
    return self;
}

Tensor& random_(Tensor& self, int64_t from, int64_t to, const Generator& gen) {
    kernel::Random_(self, from, to, gen);
    return self;
}

// ---- construction / concatenation (DESIGN §8.6.C) -------------------------

// Common validation + output allocation for cat. All inputs must share the
// source dtype and device; their shapes must match on every axis except dim.
// Returns the allocated output tensor (uninitialized).
Tensor cat_prepare(ArrayRef<Tensor> tensors, int64_t dim) {
    ZT_CHECK(!tensors.empty(),
             "cat: there is no single tensor to return from an empty list");
    const Tensor& first = tensors[0];
    const auto ndim = first.dim();
    ZT_CHECK(ndim > 0, "cat: zero-dimensional tensors cannot be concatenated");
    const ScalarType dtype = first.scalar_type();
    const Device device = first.device();
    const int64_t wdim = core::WrapDim(dim, ndim);
    Tensor::ShapeVector out_shape(first.sizes().begin(), first.sizes().end());
    int64_t total = first.size(wdim);
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        const Tensor& t = tensors[i];
        ZT_CHECK(t.scalar_type() == dtype,
                 "cat: dtype mismatch ({} vs {})",
                 toString(dtype),
                 toString(t.scalar_type()));
        ZT_CHECK(t.device() == device,
                 "cat: device mismatch ({} vs {})",
                 device.string(),
                 t.device().string());
        ZT_CHECK(
            t.dim() == ndim, "cat: rank mismatch ({} vs {})", t.dim(), ndim);
        for (int64_t d = 0; d < ndim; ++d) {
            if (d == wdim) continue;
            ZT_CHECK(t.size(d) == out_shape[d],
                     "cat: size mismatch on dim {} ({} vs {})",
                     d,
                     t.size(d),
                     out_shape[d]);
        }
        total += t.size(wdim);
    }
    out_shape[wdim] = total;
    return Tensor(out_shape, dtype, device);
}

Tensor cat(ArrayRef<Tensor> tensors, int64_t dim) {
    Tensor out = cat_prepare(tensors, dim);
    const Tensor& first = tensors[0];
    const int64_t wdim = core::WrapDim(dim, first.dim());
    int64_t offset = 0;
    for (const Tensor& t : tensors) {
        const int64_t sz = t.size(wdim);
        if (sz > 0) out.narrow(wdim, offset, sz).copy_(t);
        offset += sz;
    }
    return out;
}

Tensor stack(ArrayRef<Tensor> tensors, int64_t dim) {
    ZT_CHECK(!tensors.empty(),
             "stack: there is no single tensor to return from an empty list");
    const Tensor& first = tensors[0];
    const auto ndim = first.dim();
    const int64_t wdim = core::WrapDim(dim, ndim + 1);
    // Validate shapes/dtypes/devices up front (so a rank-mismatch surfaces
    // before we unsqueeze and hit it inside cat with a confusing message).
    const ScalarType dtype = first.scalar_type();
    const Device device = first.device();
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        const Tensor& t = tensors[i];
        ZT_CHECK(t.scalar_type() == dtype,
                 "stack: dtype mismatch ({} vs {})",
                 toString(dtype),
                 toString(t.scalar_type()));
        ZT_CHECK(t.device() == device,
                 "stack: device mismatch ({} vs {})",
                 device.string(),
                 t.device().string());
        ZT_CHECK(t.sizes() == first.sizes(),
                 "stack: shape mismatch {} vs {}",
                 t.sizes().size(),
                 first.sizes().size());
    }
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const Tensor& t : tensors) unsqueezed.push_back(t.unsqueeze(wdim));
    return cat(ArrayRef<Tensor>(unsqueezed.data(), unsqueezed.size()), wdim);
}

Tensor hstack(ArrayRef<Tensor> tensors) {
    ZT_CHECK(!tensors.empty(), "hstack: empty input list");
    const Tensor& first = tensors[0];
    if (first.dim() == 1) return cat(tensors, 0);
    return cat(tensors, 1);
}

Tensor vstack(ArrayRef<Tensor> tensors) {
    ZT_CHECK(!tensors.empty(), "vstack: empty input list");
    // PyTorch semantics: 0-D and 1-D inputs are first reshaped to 2-D
    // (1, N) before concatenating along dim 0.
    std::vector<Tensor> reshaped;
    reshaped.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        if (t.dim() <= 1) {
            reshaped.push_back(t.reshape({1, t.numel()}));
        } else {
            reshaped.push_back(t);
        }
    }
    return cat(ArrayRef<Tensor>(reshaped.data(), reshaped.size()), 0);
}

Tensor dstack(ArrayRef<Tensor> tensors) {
    ZT_CHECK(!tensors.empty(), "dstack: empty input list");
    // NumPy semantics: treat inputs as at least 3-D by appending a trailing
    // unit dim (0-D -> (1,1,1), 1-D -> (1,N,1), 2-D -> (M,N,1)), then
    // CONCATENATE along dim 2. (cat, not stack: stack would insert a brand
    // new axis and yield (M,N,K,1) instead of (M,N,K).)
    std::vector<Tensor> reshaped;
    reshaped.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        if (t.dim() == 0) {
            reshaped.push_back(t.reshape({1, 1, 1}));
        } else if (t.dim() == 1) {
            reshaped.push_back(t.reshape({1, t.size(0), 1}));
        } else if (t.dim() == 2) {
            reshaped.push_back(t.reshape({t.size(0), t.size(1), 1}));
        } else {
            reshaped.push_back(t);
        }
    }
    return cat(ArrayRef<Tensor>(reshaped.data(), reshaped.size()), 2);
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
    ZT_CHECK(split_size >= 0, "split: split_size must be non-negative");
    const int64_t wdim = core::WrapDim(dim, self.dim());
    const int64_t len = self.size(wdim);
    std::vector<Tensor> out;
    if (split_size == 0) {
        ZT_CHECK(len == 0, "split: split_size 0 requires empty dimension");
        return out;
    }
    for (int64_t offset = 0; offset < len; offset += split_size) {
        const int64_t sz = std::min(split_size, len - offset);
        out.push_back(self.slice(wdim, offset, offset + sz));
    }
    return out;
}

std::vector<Tensor> split(const Tensor& self,
                          IntArrayRef split_sizes,
                          int64_t dim) {
    const int64_t wdim = core::WrapDim(dim, self.dim());
    const int64_t len = self.size(wdim);
    std::vector<Tensor> out;
    out.reserve(split_sizes.size());
    int64_t offset = 0;
    for (const int64_t sz : split_sizes) {
        ZT_CHECK(sz >= 0, "split: negative piece size {}", sz);
        ZT_CHECK(offset + sz <= len,
                 "split: sizes sum {} exceeds dim length {}",
                 offset + sz,
                 len);
        out.push_back(self.slice(wdim, offset, offset + sz));
        offset += sz;
    }
    return out;
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
    ZT_CHECK(chunks > 0, "chunk: chunks must be greater than 0");
    const int64_t wdim = core::WrapDim(dim, self.dim());
    const int64_t len = self.size(wdim);
    if (len == 0) {
        // PyTorch returns `chunks` empty views along dim in this case.
        std::vector<Tensor> out;
        out.reserve(static_cast<std::size_t>(chunks));
        for (int64_t i = 0; i < chunks; ++i)
            out.push_back(self.slice(wdim, 0, 0));
        return out;
    }
    const int64_t split_size = (len + chunks - 1) / chunks;  // ceil
    return split(self, split_size, wdim);
}

}  // namespace zt
