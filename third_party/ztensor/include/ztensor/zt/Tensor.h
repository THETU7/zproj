// ztensor/zt/Tensor.h
//
// zt::Tensor: a multi-dimensional view of a Blob. Modeled on Open3D's
// open3d::core::Tensor (a {shape, strides, data_ptr, dtype, shared_ptr<Blob>}
// tuple) and exposing a PyTorch-flavored public API.
//
// Phase 2 scope: shape queries, shape-only views (view/reshape/permute/...),
// contiguous()/clone()/copy_()/to(), and the empty/zeros/ones/full/eye/
// arange/from_blob factories. Arithmetic operators arrive in phase 3.
//
// Ownership semantics (Open3D parity):
//   * Tensor is a non-owning view; Blob is the sole owner.
//   * `a = b` is a SHALLOW copy (shares Blob).
//   * Assignment to an rvalue Tensor, e.g. `a[i] = b` (indexing lands later),
//     is a DEEP copy. This is expressed with ref-qualifiers (`&` vs `&&`).

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ztensor/zt/ArrayRef.h"
#include "ztensor/zt/Blob.h"
#include "ztensor/zt/Device.h"
#include "ztensor/zt/dlpack.h"
#include "ztensor/zt/MemoryFormat.h"
#include "ztensor/zt/Scalar.h"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/SmallVector.h"
#include "ztensor/zt/TensorKey.h"
#include "ztensor/zt/TensorOptions.h"
#include "ztensor/zt/utility/Log.h"

namespace zt {

class Tensor {
public:
    // ── shape / strides storage ────────────────────────────────────────────
    using ShapeVector = SmallVector<int64_t, 4>;

    // ── constructors ───────────────────────────────────────────────────────

    // Default-constructed tensor: no Blob, scalar_type Undefined, shape {0}.
    Tensor() = default;

    // Copy / move are shallow (share Blob); they must be defaulted explicitly
    // because we declare user-provided copy-assignment operators below.
    Tensor(const Tensor&) noexcept = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) & noexcept = default;

    // Allocate a new contiguous tensor (uninitialized values).
    Tensor(const ShapeVector& shape, ScalarType dtype, const Device& device);

    // Permissive IntArrayRef overload (so {2, 3} works).
    Tensor(IntArrayRef shape, ScalarType dtype, const Device& device);

    // Fully specified view over an existing Blob: highest freedom, used by all
    // view operations (view/reshape/transpose/...).
    Tensor(const ShapeVector& shape,
           const ShapeVector& strides,
           void* data_ptr,
           ScalarType dtype,
           std::shared_ptr<Blob> blob);

    // ── query: shape / dtype / device ──────────────────────────────────────
    int64_t dim() const noexcept { return static_cast<int64_t>(shape_.size()); }
    int64_t ndimension() const noexcept { return dim(); }
    IntArrayRef sizes() const noexcept {
        return IntArrayRef(shape_.data(), shape_.size());
    }
    IntArrayRef strides() const noexcept {
        return IntArrayRef(strides_.data(), strides_.size());
    }
    int64_t size(int64_t dim) const;
    int64_t stride(int64_t dim) const;
    int64_t numel() const noexcept;
    std::size_t nbytes() const noexcept;  // bytes spanned by the view
    std::size_t element_size() const noexcept {
        return zt::elementSize(dtype_);
    }
    int64_t ItemRefOffset() const;  // byte offset of data_ptr_ into the blob

    // The Blob backing this view (shared with all views of the same storage).
    // Exposed so internal ops that build a new strided view of the SAME blob
    // (e.g. advanced-indexing restrihe) can do so outside Tensor.cpp.
    std::shared_ptr<Blob> GetBlob() const noexcept { return blob_; }

    ScalarType scalar_type() const noexcept { return dtype_; }
    const Device& device() const noexcept;
    bool is_cpu() const noexcept { return device().is_cpu(); }
    bool is_cuda() const noexcept { return device().is_cuda(); }
    bool defined() const noexcept { return blob_ != nullptr; }
    TensorOptions options() const noexcept;

    // ── data access ────────────────────────────────────────────────────────
    void* data_ptr() noexcept { return data_ptr_; }
    const void* data_ptr() const noexcept { return data_ptr_; }
    template<typename T>
    T* data_ptr() noexcept {
        return static_cast<T*>(data_ptr_);
    }
    template<typename T>
    const T* data_ptr() const noexcept {
        return static_cast<const T*>(data_ptr_);
    }

    // Read a 0-d (or size-1) tensor as a scalar.
    template<typename T>
    T item() const;

    // ── shape-only views (zero-copy, share Blob) ───────────────────────────
    Tensor contiguous(MemoryFormat mf = MemoryFormat::Contiguous) const;
    bool is_contiguous(MemoryFormat mf = MemoryFormat::Contiguous) const;

    Tensor view(IntArrayRef sizes) const;
    Tensor reshape(IntArrayRef sizes) const;
    Tensor permute(IntArrayRef dims) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor transpose() const;  // 2-D special case
    Tensor squeeze() const;
    Tensor squeeze(int64_t dim) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor flatten(int64_t start = 0, int64_t end = -1) const;
    Tensor slice(int64_t dim,
                 int64_t start,
                 int64_t end,
                 int64_t step = 1) const;
    Tensor expand(IntArrayRef sizes) const;
    Tensor broadcast_to(IntArrayRef sizes) const { return expand(sizes); }
    Tensor clone() const;

    // ── §8.6.C construction / repetition ──────────────────────────────────
    // Narrow: zero-copy view of the half-open range [start, start+length)
    // along `dim`. Like PyTorch's narrow / narrow_copy semantics for the view
    // half (no copy). Used by cat() to slice the output into per-input slots.
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    // Repeat: tile the whole tensor `repeats[i]` times along axis i. Unlike
    // expand (size-1-only, zero-copy), repeat always materializes a new blob.
    Tensor repeat(IntArrayRef repeats) const;
    // NumPy tile semantics: `reps` may be longer than ndim (prepend new axes)
    // or shorter (treated as 1 for the missing leading axes).
    Tensor tile(IntArrayRef reps) const;
    // Split along `dim` into chunks. Returns zero-copy views sharing the
    // blob (same lifetime model as slice()).
    std::vector<Tensor> split(int64_t split_size, int64_t dim = 0) const;
    std::vector<Tensor> split(IntArrayRef split_sizes, int64_t dim = 0) const;
    // Split into `chunks` approximately equal pieces along `dim` (PyTorch
    // semantics: the first floor(len/chunks) pieces have size ceil(len/chunks)
    // and any remainder goes to the trailing smaller pieces).
    std::vector<Tensor> chunk(int64_t chunks, int64_t dim = 0) const;

    // ── conversion ─────────────────────────────────────────────────────────
    Tensor to(TensorOptions options, bool copy = false) const;
    Tensor to(ScalarType dtype, bool copy = false) const;
    Tensor to(const Device& device, ScalarType dtype, bool copy = false) const;
    // Move to `device`, preserving dtype.
    Tensor to(const Device& device, bool copy = false) const;
    Tensor cpu() const { return to(Device(kCPU), dtype_, /*copy=*/false); }
    Tensor cuda() const { return to(Device(kCUDA), dtype_, /*copy=*/false); }

    // ── inplace writes ─────────────────────────────────────────────────────
    // lvalue assignment = shallow copy (view-sharing), matching NumPy/PyTorch.
    Tensor& operator=(const Tensor& other) &;
    // rvalue assignment = deep copy of `other`'s data into `*this`'s buffer.
    // Used by `a[i] = b` style expressions once indexing is implemented.
    Tensor& operator=(const Tensor& other) &&;
    // Fill from a scalar (deep).
    Tensor& operator=(Scalar v) &&;

    // Copy `src` into `*this` (deep). Requires same shape & dtype.
    Tensor& copy_(const Tensor& src, bool non_blocking = false);
    // Fill the whole view with `v` (deep).
    Tensor& fill_(Scalar v);
    Tensor& zero_();

    // ── element-wise arithmetic (out-of-place) ─────────────────────────────
    // Dtypes follow PyTorch/NumPy: arithmetic promotes the two operands via
    // promoteTypes, except `div` of two non-floating (integer/bool) tensors ->
    // Float (NumPy true division). `other` may be a Tensor (broadcast) or a
    // Scalar (wrapped as a 0-d tensor). `alpha` scales `other` for add/sub.
    Tensor add(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor sub(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor neg() const;
    Tensor add(Scalar other, const Scalar& alpha = 1) const;
    Tensor sub(Scalar other, const Scalar& alpha = 1) const;
    Tensor mul(Scalar other) const;
    Tensor div(Scalar other) const;

    // ── element-wise arithmetic (inplace) ──────────────────────────────────
    // Computed out-of-place into a temporary and copied back (so dtype
    // promotion + cast are handled by copy_). The broadcast result shape must
    // match self's shape, else this throws.
    Tensor& add_(const Tensor& other, const Scalar& alpha = 1);
    Tensor& sub_(const Tensor& other, const Scalar& alpha = 1);
    Tensor& mul_(const Tensor& other);
    Tensor& div_(const Tensor& other);
    Tensor& neg_();
    Tensor& add_(Scalar other, const Scalar& alpha = 1);
    Tensor& sub_(Scalar other, const Scalar& alpha = 1);
    Tensor& mul_(Scalar other);
    Tensor& div_(Scalar other);

    // Compound assignment (member: mutates *this, returns *this).
    Tensor& operator+=(const Tensor& other) { return add_(other); }
    Tensor& operator-=(const Tensor& other) { return sub_(other); }
    Tensor& operator*=(const Tensor& other) { return mul_(other); }
    Tensor& operator/=(const Tensor& other) { return div_(other); }
    Tensor& operator+=(Scalar other) { return add_(other); }
    Tensor& operator-=(Scalar other) { return sub_(other); }
    Tensor& operator*=(Scalar other) { return mul_(other); }
    Tensor& operator/=(Scalar other) { return div_(other); }

    // ── comparisons (Bool output) ──────────────────────────────────────────
    // Inputs are promoted to a common dtype, compared element-wise (with
    // broadcasting), and the result is a Bool tensor.
    Tensor eq(const Tensor& other) const;
    Tensor ne(const Tensor& other) const;
    Tensor lt(const Tensor& other) const;
    Tensor le(const Tensor& other) const;
    Tensor gt(const Tensor& other) const;
    Tensor ge(const Tensor& other) const;
    Tensor eq(Scalar other) const;
    Tensor ne(Scalar other) const;
    Tensor lt(Scalar other) const;
    Tensor le(Scalar other) const;
    Tensor gt(Scalar other) const;
    Tensor ge(Scalar other) const;

    // ── unary math (out-of-place, §8.6.A) ──────────────────────────────────
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor rsqrt() const;
    Tensor exp() const;
    Tensor expm1() const;
    Tensor log() const;
    Tensor log2() const;
    Tensor log10() const;
    Tensor log1p() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor tan() const;
    Tensor sinh() const;
    Tensor cosh() const;
    Tensor tanh() const;
    Tensor asin() const;
    Tensor acos() const;
    Tensor atan() const;
    Tensor sigmoid() const;
    Tensor reciprocal() const;
    Tensor frac() const;
    Tensor floor() const;
    Tensor ceil() const;
    Tensor round() const;
    Tensor trunc() const;
    Tensor sign() const;
    Tensor logical_not() const;
    Tensor bitwise_not() const;

    // ── unary math (inplace, §8.6.A) ──────────────────────────────────────
    Tensor& abs_();
    Tensor& sqrt_();
    Tensor& rsqrt_();
    Tensor& exp_();
    Tensor& expm1_();
    Tensor& log_();
    Tensor& log2_();
    Tensor& log10_();
    Tensor& log1p_();
    Tensor& sin_();
    Tensor& cos_();
    Tensor& tan_();
    Tensor& sinh_();
    Tensor& cosh_();
    Tensor& tanh_();
    Tensor& asin_();
    Tensor& acos_();
    Tensor& atan_();
    Tensor& sigmoid_();
    Tensor& reciprocal_();
    Tensor& frac_();
    Tensor& floor_();
    Tensor& ceil_();
    Tensor& round_();
    Tensor& trunc_();
    Tensor& sign_();
    Tensor& logical_not_();
    Tensor& bitwise_not_();

    // ── binary math (out-of-place, §8.6.A) ────────────────────────────────
    Tensor pow(const Tensor& other) const;
    Tensor fmod(const Tensor& other) const;
    Tensor remainder(const Tensor& other) const;
    Tensor maximum(const Tensor& other) const;
    Tensor minimum(const Tensor& other) const;
    Tensor atan2(const Tensor& other) const;
    Tensor hypot(const Tensor& other) const;
    Tensor logical_and(const Tensor& other) const;
    Tensor logical_or(const Tensor& other) const;
    Tensor logical_xor(const Tensor& other) const;
    Tensor bitwise_and(const Tensor& other) const;
    Tensor bitwise_or(const Tensor& other) const;
    Tensor bitwise_xor(const Tensor& other) const;
    Tensor lshift(const Tensor& other) const;
    Tensor rshift(const Tensor& other) const;
    // Scalar overloads
    Tensor pow(Scalar other) const;
    Tensor fmod(Scalar other) const;
    Tensor remainder(Scalar other) const;
    Tensor maximum(Scalar other) const;
    Tensor minimum(Scalar other) const;
    Tensor atan2(Scalar other) const;
    Tensor hypot(Scalar other) const;
    Tensor logical_and(Scalar other) const;
    Tensor logical_or(Scalar other) const;
    Tensor logical_xor(Scalar other) const;
    Tensor bitwise_and(Scalar other) const;
    Tensor bitwise_or(Scalar other) const;
    Tensor bitwise_xor(Scalar other) const;
    Tensor lshift(Scalar other) const;
    Tensor rshift(Scalar other) const;

    // ── binary math (inplace, §8.6.A) ─────────────────────────────────────
    Tensor& pow_(const Tensor& other);
    Tensor& fmod_(const Tensor& other);
    Tensor& remainder_(const Tensor& other);
    Tensor& maximum_(const Tensor& other);
    Tensor& minimum_(const Tensor& other);
    Tensor& logical_and_(const Tensor& other);
    Tensor& logical_or_(const Tensor& other);
    Tensor& logical_xor_(const Tensor& other);
    Tensor& bitwise_and_(const Tensor& other);
    Tensor& bitwise_or_(const Tensor& other);
    Tensor& bitwise_xor_(const Tensor& other);
    Tensor& lshift_(const Tensor& other);
    Tensor& rshift_(const Tensor& other);
    // Scalar inplace
    Tensor& pow_(Scalar other);
    Tensor& fmod_(Scalar other);
    Tensor& remainder_(Scalar other);
    Tensor& maximum_(Scalar other);
    Tensor& minimum_(Scalar other);

    // ── ternary / clamp (§8.6.A) ──────────────────────────────────────────
    /// Element-wise `cond ? a : b`.  `cond` must be Bool; `a` and `b` are
    /// promoted to a common dtype and broadcast with `cond`.
    static Tensor where(const Tensor& cond, const Tensor& a, const Tensor& b);

    /// Clamp all elements into [min, max].  Either bound may be std::nullopt
    /// to skip that side.  Implemented via maximum/minimum (no new kernel).
    Tensor clamp(const std::optional<Scalar>& min,
                 const std::optional<Scalar>& max) const;
    Tensor clamp(Scalar min, Scalar max) const;
    Tensor clamp_min(Scalar min) const;
    Tensor clamp_max(Scalar max) const;
    Tensor& clamp_(Scalar min, Scalar max);
    Tensor& clamp_min_(Scalar min);
    Tensor& clamp_max_(Scalar max);

    // ── indexing ───────────────────────────────────────────────────────────
    // Read: `t.index({Index(1), Slice(0,3), Tensor(idx)})`. With no Tensor
    // keys present the result is a zero-copy view (basic indexing: int +
    // slice); with any Tensor key it is a materialized gather (advanced
    // indexing). Boolean Tensor keys are masks, expanded to int64 internally.
    Tensor index(const std::vector<TensorKey>& keys) const;

    // Write: in-place scatter of `value` into the indexed locations.
    // `value` broadcasts to the indexed result shape. Non-atomic — duplicate
    // indices give last-writer-wins (matches PyTorch `index_put_`).
    Tensor& index_put_(const std::vector<TensorKey>& keys, const Tensor& value);
    Tensor& index_put_(const std::vector<TensorKey>& keys, Scalar value);

    // Single-key convenience overloads (each forwards to index()).
    Tensor operator[](int64_t i) const;
    Tensor operator[](const SliceSpec& s) const;
    Tensor operator[](const Tensor& idx) const;

    // ── §8.6.D index enhancements ──────────────────────────────────────────
    // Accumulating scatter (TensorKey form): `self[keys] += value`. Duplicate
    // indices sum (atomic on CUDA, serial on CPU) — unlike index_put_'s
    // last-writer-wins. Dtype scope: Float/Double/Int32/Int64 (native
    // atomicAdd on CUDA); other dtypes throw.
    Tensor& index_add_(const std::vector<TensorKey>& keys, const Tensor& value);

    // Dim-based scatter: `self[..., index[i], ...] = src[i]` along `dim`.
    // `index` and `src` share a shape; `self` matches except along `dim`.
    // scatter_ overwrites (last-writer-wins on duplicates); scatter_add_
    // accumulates (atomic). Out-of-range indices throw (PyTorch parity).
    Tensor& scatter_(int64_t dim, const Tensor& index, const Tensor& src);
    Tensor& scatter_(int64_t dim, const Tensor& index, Scalar value);
    Tensor& scatter_add_(int64_t dim, const Tensor& index, const Tensor& src);

    // Dim-based gather (inverse of scatter): out[i] = src[..., index[i], ...]
    // along `dim`. `out` and `index` share a shape; `src` matches except along
    // `dim`.
    Tensor gather(int64_t dim, const Tensor& index) const;

    // TensorKey-form write helpers (thin wrappers over index_put_).
    Tensor& index_fill_(const std::vector<TensorKey>& keys, Scalar value);
    Tensor& index_fill_(const std::vector<TensorKey>& keys,
                        const Tensor& value);
    Tensor& index_copy_(const std::vector<TensorKey>& keys,
                        const Tensor& value);

    // Boolean-mask write/select. `mask` broadcasts to this tensor's shape.
    Tensor& masked_fill_(const Tensor& mask, Scalar value);
    Tensor masked_select(const Tensor& mask) const;

    // Coordinates of every non-zero (true) element, packed as a {num_true,
    // ndim} int64 tensor (PyTorch nonzero). Non-Bool inputs are taken as
    // "!= 0". An empty result has shape {0, ndim}.
    Tensor nonzero() const;

    // ── reductions ─────────────────────────────────────────────────────────
    // Full reduction (all axes) collapses to a 0-d scalar; the `dims` overloads
    // reduce specific axes. keepdim keeps reduced axes as size 1. Output dtype:
    // sum/min/max/prod preserve the input dtype; mean of an integer tensor is
    // Float (else the input dtype); argmin/argmax always return Long (int64
    // indices); all/any require Bool input; var/std/norm always return Float.
    Tensor sum() const;
    Tensor mean() const;
    Tensor max() const;
    Tensor min() const;
    Tensor sum(IntArrayRef dims, bool keepdim = false) const;
    Tensor mean(IntArrayRef dims, bool keepdim = false) const;
    Tensor max(IntArrayRef dims, bool keepdim = false) const;
    Tensor min(IntArrayRef dims, bool keepdim = false) const;

    // ── §8.6.B reduction extensions ──────────────────────────────────────
    Tensor prod() const;
    Tensor prod(IntArrayRef dims, bool keepdim = false) const;

    /// Argmin / argmax: return int64 indices of the extreme value along the
    /// reduced axes.  Tie-breaking selects the smallest flat index within the
    /// reduced subspace.
    Tensor argmin() const;
    Tensor argmin(IntArrayRef dims, bool keepdim = false) const;
    Tensor argmax() const;
    Tensor argmax(IntArrayRef dims, bool keepdim = false) const;

    /// NaN-skipping min/max.  Only meaningful for floating-point types;
    /// integral types fall through to the regular min/max.
    Tensor nanmin() const;
    Tensor nanmin(IntArrayRef dims, bool keepdim = false) const;
    Tensor nanmax() const;
    Tensor nanmax(IntArrayRef dims, bool keepdim = false) const;

    /// Boolean reductions: input is cast to Bool; output is Bool.
    Tensor all() const;
    Tensor all(IntArrayRef dims, bool keepdim = false) const;
    Tensor any() const;
    Tensor any(IntArrayRef dims, bool keepdim = false) const;

    /// Count of non-zero elements.  Full reduction returns a 0-d Long scalar.
    Tensor count_nonzero() const;
    Tensor count_nonzero(IntArrayRef dims, bool keepdim = false) const;

    // ── §8.6.B statistical functions ─────────────────────────────────────
    /// Variance with optional Bessel correction (unbiased = ddof=1).
    /// Integer inputs are promoted to Float.
    Tensor var(IntArrayRef dims,
               bool keepdim = false,
               bool unbiased = true) const;
    Tensor var(bool unbiased = true) const;

    /// Standard deviation: sqrt(var(...)).
    Tensor std(IntArrayRef dims,
               bool keepdim = false,
               bool unbiased = true) const;
    Tensor std(bool unbiased = true) const;

    /// p-norm along specified axes.  p=0 (count nonzero), p=1 (L1),
    /// p=2 (L2 / Frobenius), p=-1 (infinity / max-norm).
    Tensor norm(const Scalar& p, IntArrayRef dims, bool keepdim = false) const;
    Tensor norm(const Scalar& p = Scalar(2)) const;

    // ── §8.6.B scan (cumulative operations along an axis) ────────────────
    /// Inclusive cumulative sum along `dim`.  Output has the same shape and
    /// dtype as the input.
    Tensor cumsum(int64_t dim) const;

    /// Inclusive cumulative product along `dim`.
    Tensor cumprod(int64_t dim) const;

    /// Inclusive cumulative max/min along `dim`.
    /// Returns (values, indices) pair.  Tie-breaking prefers the smallest
    /// index.
    std::pair<Tensor, Tensor> cummax(int64_t dim) const;
    std::pair<Tensor, Tensor> cummin(int64_t dim) const;

    // ── linear algebra (DESIGN §8.5.A) ─────────────────────────────────────
    // Only Float / Double / Half are supported. BFloat16 and integer matmul
    // are rejected. Half is computed in float on the CPU (cast back) and
    // natively via cuBLAS on CUDA.
    //
    // mm:    2D·2D -> 2D. (self {m,k}, other {k,n} -> {m,n}.)
    // bmm:   3D·3D -> 3D, batch dim must match (no broadcast).
    // matmul: N-d, broadcasts batch dims (all but the last 2). 1D operands
    //         are unsqueezed to 2D and the injected dim squeezed back.
    Tensor mm(const Tensor& other) const;
    Tensor bmm(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    // out = beta*self + alpha*(A@B). self is the bias (C); A {m,k}, B {k,n}.
    // Output dtype follows promote(self, A, B).
    Tensor addmm(const Tensor& A,
                 const Tensor& B,
                 Scalar beta = 1,
                 Scalar alpha = 1) const;

    // ── random (DESIGN §8.5.C) ───────────────────────────────────────────
    // Factory members that fill `*this` with random samples. Each has a
    // free-function counterpart declared in TensorFactories.h.
    //
    // uniform_([from, to)): fill with uniform samples, preserve this dtype.
    // normal_(mean, stddev): fill with normal samples, preserve this dtype.
    // random_(from, to): fill with uniform integers, preserve this dtype.
    Tensor& uniform_(double from, double to);
    Tensor& normal_(double mean, double stddev);
    Tensor& random_(int64_t from, int64_t to);

    // ── DLPack (DESIGN §8.5.D) ───────────────────────────────────────────
    // Zero-copy export/import via the DLPack C ABI. Strides are converted
    // between ztensor's element-count convention and DLPack's byte convention
    // automatically.
    struct DLManagedTensorVersioned* ToDLPack() const;
    static Tensor FromDLPack(const struct DLManagedTensorVersioned* dlmt);
    static Tensor FromDLPack(const struct DLManagedTensor* dlmt);  // v0 compat

    // ── CUDA memory pool (DESIGN §8.5.E E-4) ──────────────────────────────
    // Static helpers for diagnostics and lifecycle management of the four-tier
    // CUDA cache allocator.  No-ops when BUILD_CUDA_MODULE is off.
#ifdef BUILD_CUDA_MODULE
    static void PrintMemoryPoolStats();
    static void TrimMemoryPool();
    static void ShutdownMemoryPool();
#endif  // BUILD_CUDA_MODULE

    // Per-tensor stream operations.  cudaStream_t is only visible under
    // NVCC, so these are guarded with __CUDACC__ rather than
    // BUILD_CUDA_MODULE (same discipline as CUDAUtils.h).
#ifdef __CUDACC__
    // Notify the memory pool that `*this`'s backing memory will be used on
    // `stream` in addition to its current home stream.  Must be called
    // before work is enqueued on that stream.
    void record_stream(cudaStream_t stream);

    // Move `*this`'s backing memory home stream to `stream`.  Future reads/
    // writes are expected to happen there.  The old home becomes a recorded
    // extra use.
    void set_stream(cudaStream_t stream);
#endif  // __CUDACC__

    // ── string repr ────────────────────────────────────────────────────────
    std::string string() const;

private:
    ShapeVector shape_ = {0};
    ShapeVector strides_ = {1};
    void* data_ptr_ = nullptr;
    ScalarType dtype_ = ScalarType::Undefined;
    std::shared_ptr<Blob> blob_;  // may be nullptr (default-constructed).

    friend Tensor empty(IntArrayRef, TensorOptions);
    friend Tensor from_blob(void*,
                            IntArrayRef,
                            IntArrayRef,
                            std::function<void(void*)>,
                            TensorOptions);
};

// Static_asserts on sizeof would go here; intentionally omitted so the layout
// can evolve.

// ---------------------------------------------------------------------------
// Templates (defined inline).
// ---------------------------------------------------------------------------

template<typename T>
T Tensor::item() const {
    ZT_CHECK(numel() == 1,
             "item(): expected a single-element tensor, got numel={}",
             numel());
    ZT_CHECK(CppTypeToScalarType<T>::value == dtype_,
             "item(): dtype mismatch (requested {})",
             toString(CppTypeToScalarType<T>::value));
    return *data_ptr<T>();
}

// ---------------------------------------------------------------------------
// Free operators (PyTorch-style triple overloads).
//
// `==`/`!=`/`<`/... return a Bool TENSOR (element-wise), matching NumPy/PyTorch
// — NOT a single bool. Every operator here is a thin delegate to a member
// method, except `operator/(Scalar, Tensor)`, which needs a 0-d numerator
// tensor and is therefore declared here but defined out-of-line in Tensor.cpp.
// ---------------------------------------------------------------------------

// Unary negation.
inline Tensor operator-(const Tensor& self) { return self.neg(); }

// Arithmetic: Tensor <-> Tensor.
inline Tensor operator+(const Tensor& a, const Tensor& b) { return a.add(b); }
inline Tensor operator-(const Tensor& a, const Tensor& b) { return a.sub(b); }
inline Tensor operator*(const Tensor& a, const Tensor& b) { return a.mul(b); }
inline Tensor operator/(const Tensor& a, const Tensor& b) { return a.div(b); }

// Arithmetic: Tensor <-> Scalar.
inline Tensor operator+(const Tensor& a, Scalar b) { return a.add(b); }
inline Tensor operator-(const Tensor& a, Scalar b) { return a.sub(b); }
inline Tensor operator*(const Tensor& a, Scalar b) { return a.mul(b); }
inline Tensor operator/(const Tensor& a, Scalar b) { return a.div(b); }

// Arithmetic: Scalar <-> Tensor (+ and * are commutative; - flips via
// (-b)+a; / is defined out-of-line below).
inline Tensor operator+(Scalar a, const Tensor& b) { return b.add(a); }
inline Tensor operator*(Scalar a, const Tensor& b) { return b.mul(a); }
inline Tensor operator-(Scalar a, const Tensor& b) { return b.neg().add(a); }
Tensor operator/(Scalar a, const Tensor& b);  // defined in Tensor.cpp

// Comparisons: Tensor <-> Tensor.
inline Tensor operator==(const Tensor& a, const Tensor& b) { return a.eq(b); }
inline Tensor operator!=(const Tensor& a, const Tensor& b) { return a.ne(b); }
inline Tensor operator<(const Tensor& a, const Tensor& b) { return a.lt(b); }
inline Tensor operator<=(const Tensor& a, const Tensor& b) { return a.le(b); }
inline Tensor operator>(const Tensor& a, const Tensor& b) { return a.gt(b); }
inline Tensor operator>=(const Tensor& a, const Tensor& b) { return a.ge(b); }

// Comparisons: Tensor <-> Scalar.
inline Tensor operator==(const Tensor& a, Scalar b) { return a.eq(b); }
inline Tensor operator!=(const Tensor& a, Scalar b) { return a.ne(b); }
inline Tensor operator<(const Tensor& a, Scalar b) { return a.lt(b); }
inline Tensor operator<=(const Tensor& a, Scalar b) { return a.le(b); }
inline Tensor operator>(const Tensor& a, Scalar b) { return a.gt(b); }
inline Tensor operator>=(const Tensor& a, Scalar b) { return a.ge(b); }

// Comparisons: Scalar <-> Tensor. eq/ne are symmetric; the ordered comparisons
// reverse (a<b  <=>  b>a, etc.).
inline Tensor operator==(Scalar a, const Tensor& b) { return b.eq(a); }
inline Tensor operator!=(Scalar a, const Tensor& b) { return b.ne(a); }
inline Tensor operator<(Scalar a, const Tensor& b) { return b.gt(a); }
inline Tensor operator<=(Scalar a, const Tensor& b) { return b.ge(a); }
inline Tensor operator>(Scalar a, const Tensor& b) { return b.lt(a); }
inline Tensor operator>=(Scalar a, const Tensor& b) { return b.le(a); }

// pow: Tensor ↔ Tensor / Scalar (triple overload, §8.6.A).
inline Tensor pow(const Tensor& a, const Tensor& b) { return a.pow(b); }
inline Tensor pow(const Tensor& a, Scalar b) { return a.pow(b); }
// pow(Scalar, Tensor): construct a 0-d tensor for the base so
// `pow(3.0, t)` correctly computes 3^t (not t^3).
Tensor pow(Scalar a, const Tensor& b);  // defined in Tensor.cpp

// Free function form of nonzero (§8.6.D): identical to Tensor::nonzero().
inline Tensor nonzero(const Tensor& self) { return self.nonzero(); }
// argwhere is an alias for nonzero (PyTorch parity).
inline Tensor argwhere(const Tensor& condition) { return condition.nonzero(); }

}  // namespace zt
