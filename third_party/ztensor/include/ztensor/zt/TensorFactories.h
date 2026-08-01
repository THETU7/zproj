// ztensor/zt/TensorFactories.h
//
// Free-function tensor factories. Modeled on PyTorch's
// aten/src/ATen/native/TensorFactories.cpp, restricted to the subset ztensor
// supports in phase 2.
//
// All factories take a TensorOptions whose unspecified fields fall back to
// Float32 / CPU / Strided, matching PyTorch semantics.

#pragma once

#include <cstdint>
#include <functional>

#include "ztensor/zt/Generator.h"
#include "ztensor/zt/Scalar.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorOptions.h"

namespace zt {

// ---- allocation / fill ----
Tensor empty(IntArrayRef size, TensorOptions options = {});
Tensor zeros(IntArrayRef size, TensorOptions options = {});
Tensor ones(IntArrayRef size, TensorOptions options = {});
Tensor full(IntArrayRef size, Scalar fill_value, TensorOptions options = {});

// ---- sequences ----
Tensor arange(Scalar end, TensorOptions options = {});
Tensor arange(Scalar start,
              Scalar end,
              Scalar step = 1,
              TensorOptions options = {});
Tensor eye(int64_t n, TensorOptions options = {});

// ---- sequences (DESIGN §8.6.C) ----
// linspace: `steps` evenly-spaced values in [start, end] (or [start, end) when
// `endpoint=false`). Built on CPU then moved to the requested device, same as
// arange. logspace: base ** linspace(...). dtype defaults to Float.
Tensor linspace(Scalar start,
                Scalar end,
                int64_t steps,
                TensorOptions options = {});
Tensor linspace(Scalar start,
                Scalar end,
                int64_t steps,
                bool endpoint,
                TensorOptions options = {});
Tensor logspace(Scalar start,
                Scalar end,
                int64_t steps,
                double base = 10.0,
                TensorOptions options = {});

// ---- like ----
Tensor empty_like(const Tensor& self, TensorOptions options = {});
Tensor zeros_like(const Tensor& self, TensorOptions options = {});
Tensor ones_like(const Tensor& self, TensorOptions options = {});
Tensor full_like(const Tensor& self,
                 Scalar fill_value,
                 TensorOptions options = {});

// ---- external memory ----
// from_blob wraps an existing buffer. Without a deleter the caller retains
// ownership; with one, the Blob invokes it on final release.
Tensor from_blob(void* data, IntArrayRef sizes, TensorOptions options = {});
Tensor from_blob(void* data,
                 IntArrayRef sizes,
                 IntArrayRef strides,
                 TensorOptions options = {});
Tensor from_blob(void* data,
                 IntArrayRef sizes,
                 IntArrayRef strides,
                 std::function<void(void*)> deleter,
                 TensorOptions options = {});

// ---- random (DESIGN §8.5.C) ----
// Allocate and fill with random samples. dtype defaults to Float.
Tensor rand(IntArrayRef size,
            TensorOptions options = {},
            const Generator& gen = Generator());
Tensor randn(IntArrayRef size,
             TensorOptions options = {},
             const Generator& gen = Generator());
Tensor randint(int64_t low,
               int64_t high,
               IntArrayRef size,
               TensorOptions options = {},
               const Generator& gen = Generator());

// In-place random fills on an existing tensor.
Tensor& uniform_(Tensor& self,
                 double from,
                 double to,
                 const Generator& gen = Generator());
Tensor& normal_(Tensor& self,
                double mean,
                double stddev,
                const Generator& gen = Generator());
Tensor& random_(Tensor& self,
                int64_t from,
                int64_t to,
                const Generator& gen = Generator());

// ---- linear algebra (DESIGN §8.5.A) ----
// PyTorch-style free functions. Members on Tensor also exist; these delegate
// to them. Only Float/Double/Half are supported.
Tensor mm(const Tensor& self, const Tensor& other);
Tensor bmm(const Tensor& self, const Tensor& other);
Tensor matmul(const Tensor& self, const Tensor& other);
// addmm(C, A, B, beta, alpha) = beta*C + alpha*(A@B).
Tensor addmm(const Tensor& C,
             const Tensor& A,
             const Tensor& B,
             Scalar beta = 1,
             Scalar alpha = 1);

// ---- construction / concatenation (DESIGN §8.6.C) ----
// All inputs must share dtype and device; cat/stack allocate one new output
// tensor and copy each input into its slot (cross-device inputs are rejected;
// use Tensor::to() first). split/chunk return zero-copy views sharing the
// source blob (same lifetime model as Tensor::slice).
Tensor cat(ArrayRef<Tensor> tensors, int64_t dim = 0);
Tensor stack(ArrayRef<Tensor> tensors, int64_t dim = 0);
// Sugary 2-D / 3-D stacking helpers (PyTorch semantics):
//   hstack: 1-D -> cat(dim=0); >=2-D -> cat(dim=1).
//   vstack: 0-D -> reshape({1}) then; 1-D -> stack(dim=0); >=2-D -> cat(dim=0).
//   dstack: 0-D/1-D -> reshape to 3-D then; >=2-D -> stack(dim=2).
Tensor hstack(ArrayRef<Tensor> tensors);
Tensor vstack(ArrayRef<Tensor> tensors);
Tensor dstack(ArrayRef<Tensor> tensors);
// Split along `dim`. `split_size` is the (max) size of each piece; the last
// piece may be smaller. The IntArrayRef overload takes explicit per-piece
// sizes (their sum must equal size(dim)).
std::vector<Tensor> split(const Tensor& self,
                          int64_t split_size,
                          int64_t dim = 0);
std::vector<Tensor> split(const Tensor& self,
                          IntArrayRef split_sizes,
                          int64_t dim = 0);
// Split into `chunks` approximately equal pieces (PyTorch semantics).
std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim = 0);

}  // namespace zt
