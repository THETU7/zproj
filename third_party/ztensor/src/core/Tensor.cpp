// ztensor/core/Tensor.cpp
//
// Implementation of zt::Tensor. Phase 2 covers shape queries, shape-only
// views (which never touch data, only the {shape,strides,data_ptr} tuple),
// contiguous()/clone()/copy_/fill_/zero_, and to() conversions.

#include "ztensor/zt/Tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ztensor/zt/dlpack.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

#include "core/AdvancedIndexing.h"
#include "core/Dispatch.h"
#include "core/MemoryManager.h"
#include "core/ShapeUtil.h"
#include "kernel/ArgReduce.h"
#include "kernel/BinaryEW.h"
#include "kernel/CheckIndexBounds.h"
#include "kernel/Copy.h"
#include "kernel/Fill.h"
#include "kernel/Index.h"
#include "kernel/MatMul.h"
#include "kernel/NonZero.h"
#include "kernel/Rand.h"
#include "kernel/Reduction.h"
#include "kernel/Scan.h"
#include "kernel/Scatter.h"
#include "kernel/TernaryEW.h"
#include "kernel/UnaryEW.h"

namespace zt {
namespace {

// Row-major (C-contiguous) strides for `shape`, in elements (NOT bytes).
Tensor::ShapeVector contiguous_strides(const Tensor::ShapeVector& shape) {
    const auto ndim = shape.size();
    Tensor::ShapeVector strides(ndim, 1);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    for (std::size_t i = ndim - 1; i-- > 0;) {
        const int64_t next = shape[i + 1] > 0 ? shape[i + 1] : 1;
        strides[i] = strides[i + 1] * next;
    }
    return strides;
}

int64_t numel_of(const Tensor::ShapeVector& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// Normalize a (possibly negative) dim index into [0, ndim).
int64_t wrap_dim(int64_t dim, int64_t ndim) {
    ZT_CHECK(ndim > 0, "wrap_dim: tensor has no dimensions");
    if (dim < 0) dim += ndim;
    ZT_CHECK(dim >= 0 && dim < ndim,
             "wrap_dim: dim {} out of range for ndim {}",
             dim,
             ndim);
    return dim;
}

// Resolve a -1 in `view_shape` against numel of the original.
Tensor::ShapeVector resolve_view_shape(IntArrayRef view_shape, int64_t numel) {
    int64_t known = 1;
    int slot = -1;
    for (std::size_t i = 0; i < view_shape.size(); ++i) {
        const int64_t d = view_shape[i];
        if (d == -1) {
            ZT_CHECK(slot == -1, "view: at most one -1 allowed");
            slot = static_cast<int64_t>(i);
        } else {
            ZT_CHECK(d >= 0, "view: negative dim {} not allowed", d);
            known *= d;
        }
    }
    Tensor::ShapeVector out(view_shape.size());
    for (std::size_t i = 0; i < view_shape.size(); ++i) {
        out[i] = view_shape[i];
    }
    if (slot >= 0) {
        ZT_CHECK(known != 0 && numel % known == 0,
                 "view: cannot infer -1; numel={} not divisible by {}",
                 numel,
                 known);
        out[slot] = numel / known;
    }
    return out;
}

// ── phase-3 element-wise helpers ────────────────────────────────────────────

// A 0-d tensor holding `s`, placed on `device`, with dtype = s.type(). Feeds a
// Scalar into the broadcasting BinaryEW path (the Scalar-operand overloads).
// An empty ShapeVector selects the allocator ctor's 0-d branch (numel == 1).
Tensor scalar_to_tensor(const Scalar& s, const Device& device) {
    Tensor t(Tensor::ShapeVector{}, s.type(), device);
    t.fill_(s);
    return t;
}

// Output dtype for an arithmetic BinaryEW. promoteTypes, except Div of two
// non-floating (integer/bool) operands -> Float (NumPy true division).
ScalarType arith_out_dtype(kernel::BinaryEWOpCode op,
                           ScalarType a,
                           ScalarType b) {
    if (op == kernel::BinaryEWOpCode::Div && !isFloatingType(a) &&
        !isFloatingType(b)) {
        return ScalarType::Float;
    }
    return promoteTypes(a, b);
}

// Broadcasted shape of two tensors as a ShapeVector.
Tensor::ShapeVector broadcast_shape(const Tensor& a, const Tensor& b) {
    const auto v = core::BroadcastedShape(a.sizes(), b.sizes());
    return Tensor::ShapeVector(v.begin(), v.end());
}

// Run an arithmetic BinaryEW: promote inputs, allocate the broadcast result,
// dispatch the kernel. The allocator ctor takes a ShapeVector directly (no
// IntArrayRef conversion needed).
Tensor binary_arith(const Tensor& self,
                    const Tensor& other,
                    kernel::BinaryEWOpCode op) {
    ZT_CHECK(self.device() == other.device(),
             "binary op: device mismatch ({} vs {})",
             self.device().string(),
             other.device().string());
    const ScalarType out_dtype =
        arith_out_dtype(op, self.scalar_type(), other.scalar_type());
    const Tensor lhs = self.to(out_dtype);
    const Tensor rhs = other.to(out_dtype);
    const Tensor::ShapeVector shape = broadcast_shape(lhs, rhs);
    Tensor out(shape, out_dtype, self.device());
    kernel::BinaryEW(lhs, rhs, out, op);
    return out;
}

// Run a comparison BinaryEW: promote inputs to a common dtype, allocate a Bool
// broadcast result, dispatch the kernel.
Tensor binary_cmp(const Tensor& self,
                  const Tensor& other,
                  kernel::BinaryEWOpCode op) {
    ZT_CHECK(self.device() == other.device(),
             "comparison: device mismatch ({} vs {})",
             self.device().string(),
             other.device().string());
    const ScalarType in_dtype =
        promoteTypes(self.scalar_type(), other.scalar_type());
    const Tensor lhs = self.to(in_dtype);
    const Tensor rhs = other.to(in_dtype);
    const Tensor::ShapeVector shape = broadcast_shape(lhs, rhs);
    Tensor out(shape, ScalarType::Bool, self.device());
    kernel::BinaryEW(lhs, rhs, out, op);
    return out;
}

// Run a unary element-wise op (e.g. Neg): same shape & dtype as self.
Tensor unary_ew(const Tensor& self, kernel::UnaryEWOpCode op) {
    Tensor out = empty_like(self);
    kernel::UnaryEW(self, out, op);
    return out;
}

// ── phase-3 reduction helpers ───────────────────────────────────────────────

// Wrap user-supplied reduction dims into [0, ndim). ReductionShape wraps them
// again, but that is idempotent for already-nonnegative values.
std::vector<int64_t> resolve_reduction_dims(IntArrayRef dims, int64_t ndim) {
    std::vector<int64_t> out;
    out.reserve(dims.size());
    for (const auto d : dims) {
        out.push_back(wrap_dim(d, ndim));
    }
    return out;
}

// [0, ndim): every axis. Used by the no-arg reductions (full collapse).
std::vector<int64_t> all_dims(int64_t ndim) {
    std::vector<int64_t> out(static_cast<std::size_t>(ndim));
    std::iota(out.begin(), out.end(), int64_t{0});
    return out;
}

// Run a reduction: pre-cast src to out_dtype, allocate a keepdim-shaped dst,
// dispatch the kernel (or copy when nothing is reduced), then squeeze the
// reduced axes if keepdim=false. The caller picks out_dtype per the rules
// (sum/min/max preserve the input dtype; mean of an integer tensor is Float,
// else the input dtype).
Tensor do_reduction(const Tensor& self,
                    IntArrayRef dims,
                    bool keepdim,
                    ScalarType out_dtype,
                    kernel::ReductionOpCode op) {
    const int64_t ndim = self.dim();
    const std::vector<int64_t> rdims = resolve_reduction_dims(dims, ndim);

    // The kernel's dst must be keepdim-shaped; we squeeze afterwards if asked.
    const std::vector<int64_t> keep_vec =
        core::ReductionShape(self.sizes(), IntArrayRef(rdims), true);
    const Tensor::ShapeVector keep_shape(keep_vec.begin(), keep_vec.end());

    const Tensor src = self.to(out_dtype);
    Tensor dst(keep_shape, out_dtype, self.device());
    if (rdims.empty()) {
        dst.copy_(src);  // no axes to reduce: dtype-cast + copy.
    } else {
        kernel::Reduction(src, dst, IntArrayRef(rdims), op);
    }

    if (keepdim) {
        return dst;
    }
    const std::vector<int64_t> flat_vec =
        core::ReductionShape(self.sizes(), IntArrayRef(rdims), false);
    return dst.reshape(flat_vec);
}

// ── phase-5.A matmul helpers (DESIGN §8.5.A) ────────────────────────────────

// Output dtype for matmul. promoteTypes of the operands, but clamped to the
// supported matmul set {Float, Double, Half, BFloat16}. Integer operands are
// rejected at the entry of each public method (below).
ScalarType matmul_out_dtype(ScalarType a, ScalarType b) {
    return promoteTypes(a, b);
}

// Reject dtypes the matmul kernel does not handle.
void check_matmul_dtype(ScalarType dt) {
    if (dt != ScalarType::Float && dt != ScalarType::Double &&
        dt != ScalarType::Half && dt != ScalarType::BFloat16) {
        ZT_LOG_ERROR(
            "matmul: unsupported dtype {} (only Float/Double/Half/"
            "BFloat16)",
            toString(dt));
    }
}

// Broadcast a pair of batch-shape vectors (leading dims of A and B). Returns
// the broadcasted batch shape and asserts the per-dim broadcastability.
std::vector<int64_t> broadcast_batch(IntArrayRef a_batch, IntArrayRef b_batch) {
    const auto na = a_batch.size();
    const auto nb = b_batch.size();
    const std::size_t n = std::max(na, nb);
    std::vector<int64_t> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int64_t ad = i + na < n ? 1 : a_batch[i + na - n];
        const int64_t bd = i + nb < n ? 1 : b_batch[i + nb - n];
        if (ad != bd && ad != 1 && bd != 1) {
            ZT_LOG_ERROR(
                "matmul: batch dims not broadcastable ({} vs {})", ad, bd);
        }
        out.push_back(std::max(ad, bd));
    }
    return out;
}

// 2D matmul dispatcher entry point: shape-validate, dtype-check, allocate C,
// call the kernel.
Tensor do_mm(const Tensor& A, const Tensor& B) {
    ZT_CHECK(A.dim() == 2 && B.dim() == 2,
             "mm: expected 2D inputs, got {} and {}",
             A.dim(),
             B.dim());
    ZT_CHECK(A.size(1) == B.size(0),
             "mm: incompatible shapes ({}x{} and {}x{})",
             A.size(0),
             A.size(1),
             B.size(0),
             B.size(1));
    ZT_CHECK(A.device() == B.device(),
             "mm: device mismatch ({} vs {})",
             A.device().string(),
             B.device().string());
    const ScalarType out_dt =
        matmul_out_dtype(A.scalar_type(), B.scalar_type());
    check_matmul_dtype(out_dt);
    const Tensor A2 = A.to(out_dt);
    const Tensor B2 = B.to(out_dt);
    Tensor C(Tensor::ShapeVector{A2.size(0), B2.size(1)}, out_dt, A2.device());
    // Empty contraction dim (k == 0): the GEMM is degenerate and both Eigen
    // (CPU) and cuBLAS (CUDA) Map/arg handling on a zero-sized inner dim is
    // ill-defined. The canonical result (per PyTorch/NumPy) is a zeros tensor
    // of shape {m, n}, so short-circuit the kernel and fill zero.
    if (A2.size(1) == 0) {
        kernel::Fill(C, Scalar(static_cast<int64_t>(0)));
        return C;
    }
    kernel::MatMul(A2, B2, C);
    return C;
}

// 3D batched matmul (no batch broadcast): {b,m,k} @ {b,k,n} -> {b,m,n}.
Tensor do_bmm(const Tensor& A, const Tensor& B) {
    ZT_CHECK(A.dim() == 3 && B.dim() == 3,
             "bmm: expected 3D inputs, got {} and {}",
             A.dim(),
             B.dim());
    ZT_CHECK(A.size(0) == B.size(0),
             "bmm: batch mismatch ({} vs {})",
             A.size(0),
             B.size(0));
    ZT_CHECK(A.size(2) == B.size(1),
             "bmm: incompatible shapes ({}x{}x{} and {}x{}x{})",
             A.size(0),
             A.size(1),
             A.size(2),
             B.size(0),
             B.size(1),
             B.size(2));
    ZT_CHECK(A.device() == B.device(),
             "bmm: device mismatch ({} vs {})",
             A.device().string(),
             B.device().string());
    const ScalarType out_dt =
        matmul_out_dtype(A.scalar_type(), B.scalar_type());
    check_matmul_dtype(out_dt);
    const Tensor A3 = A.to(out_dt);
    const Tensor B3 = B.to(out_dt);
    Tensor C(Tensor::ShapeVector{A3.size(0), A3.size(1), B3.size(2)},
             out_dt,
             A3.device());
    // Empty contraction dim (k == 0): canonical zeros, see do_mm.
    if (A3.size(2) == 0) {
        kernel::Fill(C, Scalar(static_cast<int64_t>(0)));
        return C;
    }
    kernel::MatMul(A3, B3, C);
    return C;
}

// N-d matmul with batch broadcast. Pads 1D operands to 2D, broadcasts batch
// dims, flattens batch into one (broadcasting via expand+contiguous), runs a
// batched GEMM, then restores the batch shape and squeezes injected 1D dims.
//
//   1D A {k}     -> {1,k}  (squeeze row at the end)
//   1D B {k}     -> {k,1}  (squeeze col at the end)
//   batch dims of A and B broadcast like NumPy.
Tensor do_matmul(const Tensor& self, const Tensor& other) {
    ZT_CHECK(self.dim() >= 1 && other.dim() >= 1,
             "matmul: expected at least 1D inputs");
    ZT_CHECK(self.device() == other.device(),
             "matmul: device mismatch ({} vs {})",
             self.device().string(),
             other.device().string());

    Tensor A = self;
    Tensor B = other;
    const bool a_was_1d = A.dim() == 1;
    const bool b_was_1d = B.dim() == 1;
    if (a_was_1d) A = A.unsqueeze(0);   // {k} -> {1,k}
    if (b_was_1d) B = B.unsqueeze(-1);  // {k} -> {k,1}

    const int64_t m = A.size(A.dim() - 2);
    const int64_t kA = A.size(A.dim() - 1);
    const int64_t kB = B.size(B.dim() - 2);
    const int64_t n = B.size(B.dim() - 1);
    ZT_CHECK(
        kA == kB, "matmul: incompatible contraction dims ({} and {})", kA, kB);

    // Broadcast the batch dims (everything except the trailing two).
    const IntArrayRef a_batch(A.sizes().data(), A.dim() - 2);
    const IntArrayRef b_batch(B.sizes().data(), B.dim() - 2);
    const std::vector<int64_t> batch = broadcast_batch(a_batch, b_batch);

    // Materialize broadcasted operands: expand batch dims and contiguize so the
    // trailing-2 matrix dims are packed. Reshape to a flat 3D {B,m,k}/{B,k,n}.
    std::vector<int64_t> a_full(batch.begin(), batch.end());
    a_full.push_back(m);
    a_full.push_back(kA);
    std::vector<int64_t> b_full(batch.begin(), batch.end());
    b_full.push_back(kB);
    b_full.push_back(n);
    const Tensor Ab = A.expand(a_full).contiguous();
    const Tensor Bb = B.expand(b_full).contiguous();

    int64_t batch_prod = 1;
    for (auto d : batch) batch_prod *= d;

    // Reshape to 3D and run a batched GEMM.
    const Tensor A3 = Ab.reshape({batch_prod, m, kA});
    const Tensor B3 = Bb.reshape({batch_prod, kA, n});
    const ScalarType out_dt =
        matmul_out_dtype(Ab.scalar_type(), Bb.scalar_type());
    Tensor C3(Tensor::ShapeVector{batch_prod, m, n}, out_dt, Ab.device());
    // Empty contraction dim (k == 0): canonical zeros, see do_mm.
    if (kA == 0) {
        kernel::Fill(C3, Scalar(static_cast<int64_t>(0)));
    } else {
        kernel::MatMul(A3, B3, C3);
    }

    // Reshape back to broadcast batch + {m,n}.
    std::vector<int64_t> c_shape(batch.begin(), batch.end());
    if (!a_was_1d) c_shape.push_back(m);
    if (!b_was_1d) c_shape.push_back(n);
    return C3.reshape(c_shape);
}

// addmm: out = beta*C + alpha*(A@B). C is the bias; A {m,k}; B {k,n}; out
// {m,n}.
Tensor do_addmm(const Tensor& C,
                const Tensor& A,
                const Tensor& B,
                Scalar beta,
                Scalar alpha) {
    ZT_CHECK(C.dim() == 2 && A.dim() == 2 && B.dim() == 2,
             "addmm: expected 2D inputs, got C ndim {}, A ndim {}, B ndim {}",
             C.dim(),
             A.dim(),
             B.dim());
    ZT_CHECK(A.size(1) == B.size(0),
             "addmm: A@B incompatible ({}x{} and {}x{})",
             A.size(0),
             A.size(1),
             B.size(0),
             B.size(1));
    ZT_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1),
             "addmm: C shape {}x{} does not match A@B {}x{}",
             C.size(0),
             C.size(1),
             A.size(0),
             B.size(1));
    ZT_CHECK(C.device() == A.device() && A.device() == B.device(),
             "addmm: device mismatch ({} / {} / {})",
             C.device().string(),
             A.device().string(),
             B.device().string());
    const ScalarType out_dt = matmul_out_dtype(
        C.scalar_type(), matmul_out_dtype(A.scalar_type(), B.scalar_type()));
    check_matmul_dtype(out_dt);
    const Tensor C2 = C.to(out_dt);
    const Tensor A2 = A.to(out_dt);
    const Tensor B2 = B.to(out_dt);
    Tensor out(
        Tensor::ShapeVector{C2.size(0), C2.size(1)}, out_dt, C2.device());
    // Empty contraction dim (k == 0): A@B contributes nothing, so out = beta*C.
    // The cuBLAS/Eigen sgemm path on a zero inner dim is ill-defined, so
    // compute the bias contribution directly and skip AddMM.
    if (A2.size(1) == 0) {
        const double b = beta.toDouble();
        if (b == 0.0) {
            kernel::Fill(out, Scalar(static_cast<int64_t>(0)));
        } else if (b == 1.0) {
            out.copy_(C2);
        } else {
            out = C2.mul(Scalar(b));
        }
        return out;
    }
    kernel::AddMM(C2, A2, B2, out, beta.toDouble(), alpha.toDouble());
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

Tensor::Tensor(const ShapeVector& shape, ScalarType dtype, const Device& device)
    : shape_(shape), dtype_(dtype) {
    check_device_supported(device);
    strides_ = contiguous_strides(shape_);
    const std::size_t bytes =
        static_cast<std::size_t>(numel()) * element_size();
    blob_ = std::make_shared<Blob>(bytes, device);
    data_ptr_ = blob_->GetDataPtr();
}

Tensor::Tensor(IntArrayRef shape, ScalarType dtype, const Device& device)
    : Tensor(ShapeVector(shape.begin(), shape.end()), dtype, device) {}

Tensor::Tensor(const ShapeVector& shape,
               const ShapeVector& strides,
               void* data_ptr,
               ScalarType dtype,
               std::shared_ptr<Blob> blob)
    : shape_(shape),
      strides_(strides),
      data_ptr_(data_ptr),
      dtype_(dtype),
      blob_(std::move(blob)) {}

// ---------------------------------------------------------------------------
// Shape queries
// ---------------------------------------------------------------------------

int64_t Tensor::size(int64_t dim) const {
    return shape_[wrap_dim(dim, this->dim())];
}
int64_t Tensor::stride(int64_t dim) const {
    return strides_[wrap_dim(dim, this->dim())];
}

int64_t Tensor::numel() const noexcept { return numel_of(shape_); }

std::size_t Tensor::nbytes() const noexcept {
    if (shape_.empty() || numel() == 0) return 0;
    // Spanned bytes = element_size * sum over strided extent. For contiguous
    // tensors this equals numel * element_size; for arbitrary strides we use
    // the max byte reached.
    const auto esize = element_size();
    // Use the bytes that contiguous() would touch for safety: numel * esize.
    return static_cast<std::size_t>(numel()) * esize;
}

int64_t Tensor::ItemRefOffset() const {
    if (!blob_) return 0;
    const auto base = static_cast<const std::uint8_t*>(blob_->GetDataPtr());
    return static_cast<int64_t>(static_cast<const std::uint8_t*>(data_ptr_) -
                                base);
}

const Device& Tensor::device() const noexcept {
    static const Device kDefaultCPU = Device(kCPU);
    return blob_ ? blob_->GetDevice() : kDefaultCPU;
}

TensorOptions Tensor::options() const noexcept {
    return TensorOptions().dtype(dtype_).device(device());
}

// ---------------------------------------------------------------------------
// is_contiguous / contiguous
// ---------------------------------------------------------------------------

bool Tensor::is_contiguous(MemoryFormat /*mf*/) const {
    // A tensor is row-major contiguous iff strides[i] == product(shape[i+1:]).
    // Size-1 dims tolerate any stride; size-0 dims also tolerate any stride
    // (they hold no elements) and — crucially — do not collapse the running
    // stride product to zero, which would otherwise falsely mark every
    // subsequent (non-zero) dim as non-contiguous and send reshape() into an
    // infinite contiguous().reshape() recursion on empty tensors.
    const auto ndim = shape_.size();
    if (ndim == 0) return true;
    int64_t acc = 1;
    for (std::size_t i = ndim; i-- > 0;) {
        if (shape_[i] <= 1) continue;  // size-0 and size-1: any stride is fine
        if (strides_[i] != acc) return false;
        acc *= shape_[i];
    }
    return true;
}

Tensor Tensor::contiguous(MemoryFormat /*mf*/) const {
    if (is_contiguous()) return *this;
    Tensor out(shape_, dtype_, device());
    out.copy_(*this);
    return out;
}

Tensor Tensor::clone() const {
    Tensor out(shape_, dtype_, device());
    out.copy_(*this);
    return out;
}

// ---------------------------------------------------------------------------
// Shape-only views
// ---------------------------------------------------------------------------

Tensor Tensor::view(IntArrayRef sizes) const {
    ZT_CHECK(is_contiguous(),
             "view(): tensor must be contiguous; call .contiguous() first");
    const auto new_shape = resolve_view_shape(sizes, numel());
    ZT_CHECK(numel_of(new_shape) == numel(),
             "view: shape {} is incompatible with numel {}",
             new_shape.size(),
             numel());
    auto new_strides = contiguous_strides(new_shape);
    return Tensor(new_shape, new_strides, data_ptr_, dtype_, blob_);
}

Tensor Tensor::reshape(IntArrayRef sizes) const {
    const auto new_shape = resolve_view_shape(sizes, numel());
    ZT_CHECK(numel_of(new_shape) == numel(), "reshape: incompatible numel");
    if (is_contiguous()) {
        auto new_strides = contiguous_strides(new_shape);
        return Tensor(new_shape, new_strides, data_ptr_, dtype_, blob_);
    }
    return contiguous().reshape(sizes);
}

Tensor Tensor::permute(IntArrayRef dims) const {
    const auto ndim = this->dim();
    ZT_CHECK(static_cast<int64_t>(dims.size()) == ndim,
             "permute: expected {} dims, got {}",
             ndim,
             dims.size());
    ShapeVector seen(ndim, 0);
    ShapeVector new_shape(ndim), new_strides(ndim);
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const int64_t d = wrap_dim(dims[i], ndim);
        ZT_CHECK(seen[d] == 0, "permute: dim {} repeated", d);
        seen[d] = 1;
        new_shape[i] = shape_[d];
        new_strides[i] = strides_[d];
    }
    return Tensor(new_shape, new_strides, data_ptr_, dtype_, blob_);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    const auto ndim = this->dim();
    dim0 = wrap_dim(dim0, ndim);
    dim1 = wrap_dim(dim1, ndim);
    auto new_shape = shape_;
    auto new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return Tensor(new_shape, new_strides, data_ptr_, dtype_, blob_);
}

Tensor Tensor::transpose() const {
    ZT_CHECK(dim() == 2, "transpose(): no-arg form requires a 2-D tensor");
    return transpose(0, 1);
}

Tensor Tensor::squeeze() const {
    ShapeVector ns, nst;
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != 1) {
            ns.push_back(shape_[i]);
            nst.push_back(strides_[i]);
        }
    }
    if (ns.empty()) {
        ns.push_back(1);
        nst.push_back(1);
    }  // 0-d corner case -> keep as is
    // Produce a 0-d tensor when all dims were 1.
    if (ns.size() == 1 && shape_.size() > 1 && numel() == 1) {
        // Keep a scalar (shape {}) view.
        return Tensor(ShapeVector{}, ShapeVector{}, data_ptr_, dtype_, blob_);
    }
    return Tensor(ns, nst, data_ptr_, dtype_, blob_);
}

Tensor Tensor::squeeze(int64_t dim) const {
    const auto ndim = this->dim();
    dim = wrap_dim(dim, ndim);
    ZT_CHECK(shape_[dim] == 1,
             "squeeze(): dim {} has size {}, not 1",
             dim,
             shape_[dim]);
    ShapeVector ns, nst;
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) continue;
        ns.push_back(shape_[i]);
        nst.push_back(strides_[i]);
    }
    return Tensor(ns, nst, data_ptr_, dtype_, blob_);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    const auto ndim = this->dim();
    int64_t d = dim;
    if (d < 0) d += static_cast<int64_t>(ndim) + 1;
    ZT_CHECK(d >= 0 && d <= static_cast<int64_t>(ndim),
             "unsqueeze: dim {} out of range for ndim {}",
             dim,
             ndim);
    ShapeVector ns, nst;
    for (std::size_t i = 0; i <= shape_.size(); ++i) {
        if (static_cast<int64_t>(i) == d) {
            ns.push_back(1);
            nst.push_back(0);  // stride of a size-1 dim is irrelevant
        }
        if (i < shape_.size()) {
            ns.push_back(shape_[i]);
            nst.push_back(strides_[i]);
        }
    }
    return Tensor(ns, nst, data_ptr_, dtype_, blob_);
}

Tensor Tensor::flatten(int64_t start, int64_t end) const {
    const auto ndim = this->dim();
    if (ndim == 0) return *this;
    start = wrap_dim(start, ndim);
    end = wrap_dim(end, ndim);
    ZT_CHECK(start <= end, "flatten: start {} > end {}", start, end);
    ShapeVector ns, nst;
    for (int64_t i = 0; i < start; ++i) {
        ns.push_back(shape_[i]);
        nst.push_back(strides_[i]);
    }
    int64_t flat = 1;
    for (int64_t i = start; i <= end; ++i) flat *= shape_[i];
    ns.push_back(flat);
    nst.push_back(strides_[end]);
    for (int64_t i = end + 1; i < ndim; ++i) {
        ns.push_back(shape_[i]);
        nst.push_back(strides_[i]);
    }
    return Tensor(ns, nst, data_ptr_, dtype_, blob_);
}

Tensor Tensor::slice(int64_t dim,
                     int64_t start,
                     int64_t end,
                     int64_t step) const {
    ZT_CHECK(step != 0, "slice: step must be non-zero");
    const auto ndim = this->dim();
    dim = wrap_dim(dim, ndim);
    const int64_t len = shape_[dim];
    int64_t s, e;
    if (step > 0) {
        auto clamp = [len](int64_t x) {
            if (x < 0) x += len;
            if (x < 0) return static_cast<int64_t>(0);
            if (x > len) return len;
            return x;
        };
        s = clamp(start);
        e = clamp(end);
        if (e < s) e = s;
    } else {
        // NumPy clamping for negative steps: both bounds live in [-1, len-1].
        // A stop of -1 includes index 0 (t[::-1] == all elements); a start of
        // -1 yields an empty slice. The old step-agnostic clamp mapped a stop
        // of -1 to 0, silently dropping the first element of reversed slices.
        s = start < 0 ? start + len : start;
        e = end < 0 ? end + len : end;
        if (s < -1) s = -1;
        if (s > len - 1) s = len - 1;
        if (e < -1) e = -1;
        if (e > len - 1) e = len - 1;
        if (e > s) e = s;
    }
    const int64_t new_len =
        step > 0 ? (e - s + step - 1) / step : (s - e + (-step) - 1) / (-step);
    ShapeVector ns = shape_;
    ShapeVector nst = strides_;
    ns[dim] = new_len > 0 ? new_len : 0;
    nst[dim] = strides_[dim] * step;
    auto* base = static_cast<std::uint8_t*>(data_ptr_) +
                 s * strides_[dim] * element_size();
    return Tensor(ns, nst, base, dtype_, blob_);
}

Tensor Tensor::expand(IntArrayRef sizes) const {
    const auto ndim = this->dim();
    ZT_CHECK(static_cast<int64_t>(sizes.size()) >= ndim,
             "expand: target rank {} < source rank {}",
             sizes.size(),
             ndim);
    const auto extra = sizes.size() - ndim;
    ShapeVector ns, nst;
    for (std::size_t i = 0; i < extra; ++i) {
        ns.push_back(sizes[i]);
        nst.push_back(0);
    }
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        const int64_t want = sizes[extra + i];
        // -1 keeps the existing extent (PyTorch expand contract).
        ZT_CHECK(want == shape_[i] || shape_[i] == 1 || want == -1,
                 "expand: dim {} has size {}, cannot expand to {}",
                 i,
                 shape_[i],
                 want);
        const int64_t final_size = want == -1 ? shape_[i] : want;
        ns.push_back(final_size);
        nst.push_back(shape_[i] == 1 ? 0 : strides_[i]);
    }
    return Tensor(ns, nst, data_ptr_, dtype_, blob_);
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    ZT_CHECK(length >= 0, "narrow: length {} must be non-negative", length);
    const auto ndim = this->dim();
    dim = wrap_dim(dim, ndim);
    const int64_t len = shape_[dim];
    ZT_CHECK(start >= 0 && start + length <= len,
             "narrow: range [{}, {}) out of bounds for size {} along dim {}",
             start,
             start + length,
             len,
             dim);
    ShapeVector ns = shape_;
    ShapeVector nst = strides_;
    ns[dim] = length;
    // strides unchanged — only the base pointer shifts.
    auto* base = static_cast<std::uint8_t*>(data_ptr_) +
                 start * strides_[dim] * element_size();
    return Tensor(ns, nst, base, dtype_, blob_);
}

// ---------------------------------------------------------------------------
// to()
// ---------------------------------------------------------------------------

Tensor Tensor::to(ScalarType dtype, bool copy) const {
    return to(device(), dtype, copy);
}

Tensor Tensor::to(const Device& target_device, bool copy) const {
    return to(target_device, dtype_, copy);
}

Tensor Tensor::to(const Device& target_device,
                  ScalarType target_dtype,
                  bool copy) const {
    if (!copy && target_dtype == dtype_ && target_device == device()) {
        return *this;
    }
    if (!defined()) {
        return *this;
    }
    Tensor out(shape_, target_dtype, target_device);
    if (target_dtype == dtype_ && target_device == device()) {
        out.copy_(*this);
    } else if (target_dtype == dtype_) {
        // device change only
        out.copy_(*this);
    } else {
        // dtype change: copy with element-wise cast.
        out.copy_(*this);
    }
    return out;
}

Tensor Tensor::to(TensorOptions options, bool copy) const {
    return to(options.device(), options.dtype(), copy);
}

// ---------------------------------------------------------------------------
// Assignment (shallow vs deep) + copy_/fill_/zero_
// ---------------------------------------------------------------------------

Tensor& Tensor::operator=(const Tensor& other) & {
    if (this != &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        data_ptr_ = other.data_ptr_;
        dtype_ = other.dtype_;
        blob_ = other.blob_;  // shallow: share Blob
    }
    return *this;
}

Tensor& Tensor::operator=(const Tensor& other) && {
    // rvalue form: deep-copy data.
    copy_(other);
    return *this;
}

Tensor& Tensor::operator=(Scalar v) && {
    fill_(v);
    return *this;
}

// Copy `src` into `*this` (deep). Same shape required; dtype and/or device may
// differ (the data is cast / transferred element-wise via kernel::Copy).
//
// Same-dtype + both-contiguous + both-same-device is a single coarse bytewise
// Memcpy fast path (the common case for freshly-allocated contiguous buffers
// and for cross-device transfers of contiguous tensors). Every other case —
// dtype cast, non-contiguous strides, cross-device — is handled by
// kernel::Copy, which dispatches per-device and stages cross-device transfers
// through a contiguous buffer so a GPU kernel never dereferences host memory.
Tensor& Tensor::copy_(const Tensor& src, bool /*non_blocking*/) {
    ZT_CHECK(defined(), "copy_: destination is undefined");
    ZT_CHECK(src.defined(), "copy_: source is undefined");
    ZT_CHECK(shape_ == src.shape_,
             "copy_: shape mismatch (dst ndim {} vs src ndim {})",
             shape_.size(),
             src.shape_.size());

    // Fast path: same dtype, both contiguous -> one bytewise blit (works across
    // devices too: MemoryManager::Memcpy routes H2D/D2H/D2D internally).
    if (dtype_ == src.dtype_ && src.is_contiguous() && is_contiguous()) {
        MemoryManager::Memcpy(
            device(),
            data_ptr_,
            src.device(),
            src.data_ptr_,
            static_cast<std::size_t>(numel()) * element_size());
        return *this;
    }

    // Everything else: per-element copy with cast + strides (+ cross-device
    // staging for CPU<->CUDA under BUILD_CUDA_MODULE).
    kernel::Copy(src, *this);
    return *this;
}

Tensor& Tensor::fill_(Scalar v) {
    ZT_CHECK(defined(), "fill_: tensor is undefined");
    kernel::Fill(*this, v);
    return *this;
}

Tensor& Tensor::zero_() {
    ZT_CHECK(defined(), "zero_: tensor is undefined");
    kernel::Fill(*this, Scalar(0));
    return *this;
}

// ---------------------------------------------------------------------------
// Element-wise arithmetic + comparisons (phase 3)
// ---------------------------------------------------------------------------

Tensor Tensor::add(const Tensor& other, const Scalar& alpha) const {
    // alpha scales `other`: skip the mul when alpha == 1 (the common case).
    const Tensor rhs = (alpha.toDouble() == 1.0) ? other : other.mul(alpha);
    return binary_arith(*this, rhs, kernel::BinaryEWOpCode::Add);
}
Tensor Tensor::sub(const Tensor& other, const Scalar& alpha) const {
    const Tensor rhs = (alpha.toDouble() == 1.0) ? other : other.mul(alpha);
    return binary_arith(*this, rhs, kernel::BinaryEWOpCode::Sub);
}
Tensor Tensor::mul(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Mul);
}
Tensor Tensor::div(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Div);
}
Tensor Tensor::neg() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Neg);
}

// Scalar-operand overloads: wrap the scalar as a 0-d tensor and reuse the
// broadcasting Tensor-operand path.
Tensor Tensor::add(Scalar other, const Scalar& alpha) const {
    return add(scalar_to_tensor(other, device()), alpha);
}
Tensor Tensor::sub(Scalar other, const Scalar& alpha) const {
    return sub(scalar_to_tensor(other, device()), alpha);
}
Tensor Tensor::mul(Scalar other) const {
    return mul(scalar_to_tensor(other, device()));
}
Tensor Tensor::div(Scalar other) const {
    return div(scalar_to_tensor(other, device()));
}

// Inplace (Tensor operand): compute out-of-place, then copy_ back. copy_
// handles the dtype cast when the promoted result is wider than self.
Tensor& Tensor::add_(const Tensor& other, const Scalar& alpha) {
    const Tensor rhs = (alpha.toDouble() == 1.0) ? other : other.mul(alpha);
    const Tensor result = binary_arith(*this, rhs, kernel::BinaryEWOpCode::Add);
    ZT_CHECK(result.sizes() == sizes(),
             "add_: broadcast result is incompatible with the inplace target");
    copy_(result);
    return *this;
}
Tensor& Tensor::sub_(const Tensor& other, const Scalar& alpha) {
    const Tensor rhs = (alpha.toDouble() == 1.0) ? other : other.mul(alpha);
    const Tensor result = binary_arith(*this, rhs, kernel::BinaryEWOpCode::Sub);
    ZT_CHECK(result.sizes() == sizes(),
             "sub_: broadcast result is incompatible with the inplace target");
    copy_(result);
    return *this;
}
Tensor& Tensor::mul_(const Tensor& other) {
    const Tensor result =
        binary_arith(*this, other, kernel::BinaryEWOpCode::Mul);
    ZT_CHECK(result.sizes() == sizes(),
             "mul_: broadcast result is incompatible with the inplace target");
    copy_(result);
    return *this;
}
Tensor& Tensor::div_(const Tensor& other) {
    const Tensor result =
        binary_arith(*this, other, kernel::BinaryEWOpCode::Div);
    ZT_CHECK(result.sizes() == sizes(),
             "div_: broadcast result is incompatible with the inplace target");
    copy_(result);
    return *this;
}
Tensor& Tensor::neg_() {
    copy_(unary_ew(*this, kernel::UnaryEWOpCode::Neg));
    return *this;
}

// Inplace (Scalar operand).
Tensor& Tensor::add_(Scalar other, const Scalar& alpha) {
    return add_(scalar_to_tensor(other, device()), alpha);
}
Tensor& Tensor::sub_(Scalar other, const Scalar& alpha) {
    return sub_(scalar_to_tensor(other, device()), alpha);
}
Tensor& Tensor::mul_(Scalar other) {
    return mul_(scalar_to_tensor(other, device()));
}
Tensor& Tensor::div_(Scalar other) {
    return div_(scalar_to_tensor(other, device()));
}

// Comparisons.
Tensor Tensor::eq(const Tensor& other) const {
    return binary_cmp(*this, other, kernel::BinaryEWOpCode::Eq);
}
Tensor Tensor::ne(const Tensor& other) const {
    return binary_cmp(*this, other, kernel::BinaryEWOpCode::Ne);
}
Tensor Tensor::lt(const Tensor& other) const {
    return binary_cmp(*this, other, kernel::BinaryEWOpCode::Lt);
}
Tensor Tensor::le(const Tensor& other) const {
    return binary_cmp(*this, other, kernel::BinaryEWOpCode::Le);
}
Tensor Tensor::gt(const Tensor& other) const {
    return binary_cmp(*this, other, kernel::BinaryEWOpCode::Gt);
}
Tensor Tensor::ge(const Tensor& other) const {
    return binary_cmp(*this, other, kernel::BinaryEWOpCode::Ge);
}

// Comparison Scalar overloads.
Tensor Tensor::eq(Scalar other) const {
    return eq(scalar_to_tensor(other, device()));
}
Tensor Tensor::ne(Scalar other) const {
    return ne(scalar_to_tensor(other, device()));
}
Tensor Tensor::lt(Scalar other) const {
    return lt(scalar_to_tensor(other, device()));
}
Tensor Tensor::le(Scalar other) const {
    return le(scalar_to_tensor(other, device()));
}
Tensor Tensor::gt(Scalar other) const {
    return gt(scalar_to_tensor(other, device()));
}
Tensor Tensor::ge(Scalar other) const {
    return ge(scalar_to_tensor(other, device()));
}

// Free operator declared in Tensor.h; defined here because it needs
// scalar_to_tensor (anon-namespace) to build the 0-d numerator.
Tensor operator/(Scalar a, const Tensor& b) {
    return binary_arith(
        scalar_to_tensor(a, b.device()), b, kernel::BinaryEWOpCode::Div);
}

Tensor pow(Scalar a, const Tensor& b) {
    return scalar_to_tensor(a, b.device()).pow(b);
}

// ── §8.6.A helpers ─────────────────────────────────────────────────────────

// Run a logical BinaryEW (And/Or/Xor): Bool output, INPUT_SAME_OUTPUT_BOOL.
Tensor binary_logical(const Tensor& self,
                      const Tensor& other,
                      kernel::BinaryEWOpCode op) {
    ZT_CHECK(self.device() == other.device(),
             "logical op: device mismatch ({} vs {})",
             self.device().string(),
             other.device().string());
    const ScalarType in_dtype =
        promoteTypes(self.scalar_type(), other.scalar_type());
    const Tensor lhs = self.to(in_dtype);
    const Tensor rhs = other.to(in_dtype);
    const Tensor::ShapeVector shape = broadcast_shape(lhs, rhs);
    Tensor out(shape, ScalarType::Bool, self.device());
    kernel::BinaryEW(lhs, rhs, out, op);
    return out;
}

// Run a bitwise BinaryEW: ALL_SAME, integral only.
Tensor binary_bitwise(const Tensor& self,
                      const Tensor& other,
                      kernel::BinaryEWOpCode op) {
    ZT_CHECK(self.device() == other.device(),
             "bitwise op: device mismatch ({} vs {})",
             self.device().string(),
             other.device().string());
    const ScalarType out_dtype =
        promoteTypes(self.scalar_type(), other.scalar_type());
    const Tensor lhs = self.to(out_dtype);
    const Tensor rhs = other.to(out_dtype);
    const Tensor::ShapeVector shape = broadcast_shape(lhs, rhs);
    Tensor out(shape, out_dtype, self.device());
    kernel::BinaryEW(lhs, rhs, out, op);
    return out;
}

// Run a ternary element-wise op (Where): cond (Bool) + a + b → same dtype as a.
Tensor ternary_ew(const Tensor& cond,
                  const Tensor& a,
                  const Tensor& b,
                  kernel::TernaryEWOpCode op) {
    ZT_CHECK(cond.scalar_type() == ScalarType::Bool,
             "ternary op: cond must be Bool, got {}",
             toString(cond.scalar_type()));
    const ScalarType out_dtype = promoteTypes(a.scalar_type(), b.scalar_type());
    const Tensor pa = a.to(out_dtype);
    const Tensor pb = b.to(out_dtype);
    // Broadcast all three together.
    const Tensor::ShapeVector shape_ab = broadcast_shape(pa, pb);
    // Full 3-way broadcast: broadcast cond against shape_ab.
    const auto v_cond = core::BroadcastedShape(
        cond.sizes(), IntArrayRef(shape_ab.data(), shape_ab.size()));
    const Tensor::ShapeVector shape(v_cond.begin(), v_cond.end());
    // Convert to IntArrayRef for expand().
    const IntArrayRef shape_ref(shape.data(), shape.size());
    // Expand cond/a/b to the full output shape.
    Tensor c_exp = cond.expand(shape_ref).contiguous();
    Tensor a_exp = pa.expand(shape_ref).contiguous();
    Tensor b_exp = pb.expand(shape_ref).contiguous();
    Tensor out(shape, out_dtype, a.device());
    kernel::TernaryEW(c_exp, a_exp, b_exp, out, op);
    return out;
}

// ── §8.6.A unary math (out-of-place) ───────────────────────────────────────

Tensor Tensor::abs() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Abs);
}
Tensor Tensor::sqrt() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Sqrt);
}
Tensor Tensor::rsqrt() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Rsqrt);
}
Tensor Tensor::exp() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Exp);
}
Tensor Tensor::expm1() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Expm1);
}
Tensor Tensor::log() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Log);
}
Tensor Tensor::log2() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Log2);
}
Tensor Tensor::log10() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Log10);
}
Tensor Tensor::log1p() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Log1p);
}
Tensor Tensor::sin() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Sin);
}
Tensor Tensor::cos() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Cos);
}
Tensor Tensor::tan() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Tan);
}
Tensor Tensor::sinh() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Sinh);
}
Tensor Tensor::cosh() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Cosh);
}
Tensor Tensor::tanh() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Tanh);
}
Tensor Tensor::asin() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Asin);
}
Tensor Tensor::acos() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Acos);
}
Tensor Tensor::atan() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Atan);
}
Tensor Tensor::sigmoid() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Sigmoid);
}
Tensor Tensor::reciprocal() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Reciprocal);
}
Tensor Tensor::frac() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Frac);
}
Tensor Tensor::floor() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Floor);
}
Tensor Tensor::ceil() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Ceil);
}
Tensor Tensor::round() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Round);
}
Tensor Tensor::trunc() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Trunc);
}
Tensor Tensor::sign() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::Sign);
}
Tensor Tensor::logical_not() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::LogicalNot);
}
Tensor Tensor::bitwise_not() const {
    return unary_ew(*this, kernel::UnaryEWOpCode::BitwiseNot);
}

// ── §8.6.A unary math (inplace) ────────────────────────────────────────────

Tensor& Tensor::abs_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Abs));
}
Tensor& Tensor::sqrt_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Sqrt));
}
Tensor& Tensor::rsqrt_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Rsqrt));
}
Tensor& Tensor::exp_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Exp));
}
Tensor& Tensor::expm1_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Expm1));
}
Tensor& Tensor::log_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Log));
}
Tensor& Tensor::log2_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Log2));
}
Tensor& Tensor::log10_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Log10));
}
Tensor& Tensor::log1p_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Log1p));
}
Tensor& Tensor::sin_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Sin));
}
Tensor& Tensor::cos_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Cos));
}
Tensor& Tensor::tan_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Tan));
}
Tensor& Tensor::sinh_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Sinh));
}
Tensor& Tensor::cosh_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Cosh));
}
Tensor& Tensor::tanh_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Tanh));
}
Tensor& Tensor::asin_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Asin));
}
Tensor& Tensor::acos_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Acos));
}
Tensor& Tensor::atan_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Atan));
}
Tensor& Tensor::sigmoid_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Sigmoid));
}
Tensor& Tensor::reciprocal_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Reciprocal));
}
Tensor& Tensor::frac_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Frac));
}
Tensor& Tensor::floor_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Floor));
}
Tensor& Tensor::ceil_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Ceil));
}
Tensor& Tensor::round_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Round));
}
Tensor& Tensor::trunc_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Trunc));
}
Tensor& Tensor::sign_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::Sign));
}
Tensor& Tensor::logical_not_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::LogicalNot));
}
Tensor& Tensor::bitwise_not_() {
    return copy_(unary_ew(*this, kernel::UnaryEWOpCode::BitwiseNot));
}

// ── §8.6.A binary math (out-of-place, Tensor operand) ──────────────────────

Tensor Tensor::pow(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Pow);
}
Tensor Tensor::fmod(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Fmod);
}
Tensor Tensor::remainder(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Remainder);
}
Tensor Tensor::maximum(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Maximum);
}
Tensor Tensor::minimum(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Minimum);
}
Tensor Tensor::atan2(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Atan2);
}
Tensor Tensor::hypot(const Tensor& other) const {
    return binary_arith(*this, other, kernel::BinaryEWOpCode::Hypot);
}
Tensor Tensor::logical_and(const Tensor& other) const {
    return binary_logical(*this, other, kernel::BinaryEWOpCode::LogicalAnd);
}
Tensor Tensor::logical_or(const Tensor& other) const {
    return binary_logical(*this, other, kernel::BinaryEWOpCode::LogicalOr);
}
Tensor Tensor::logical_xor(const Tensor& other) const {
    return binary_logical(*this, other, kernel::BinaryEWOpCode::LogicalXor);
}
Tensor Tensor::bitwise_and(const Tensor& other) const {
    return binary_bitwise(*this, other, kernel::BinaryEWOpCode::BitwiseAnd);
}
Tensor Tensor::bitwise_or(const Tensor& other) const {
    return binary_bitwise(*this, other, kernel::BinaryEWOpCode::BitwiseOr);
}
Tensor Tensor::bitwise_xor(const Tensor& other) const {
    return binary_bitwise(*this, other, kernel::BinaryEWOpCode::BitwiseXor);
}
Tensor Tensor::lshift(const Tensor& other) const {
    return binary_bitwise(*this, other, kernel::BinaryEWOpCode::Lshift);
}
Tensor Tensor::rshift(const Tensor& other) const {
    return binary_bitwise(*this, other, kernel::BinaryEWOpCode::Rshift);
}

// ── §8.6.A binary math (out-of-place, Scalar operand) ─────────────────────

Tensor Tensor::pow(Scalar other) const {
    return pow(scalar_to_tensor(other, device()));
}
Tensor Tensor::fmod(Scalar other) const {
    return fmod(scalar_to_tensor(other, device()));
}
Tensor Tensor::remainder(Scalar other) const {
    return remainder(scalar_to_tensor(other, device()));
}
Tensor Tensor::maximum(Scalar other) const {
    return maximum(scalar_to_tensor(other, device()));
}
Tensor Tensor::minimum(Scalar other) const {
    return minimum(scalar_to_tensor(other, device()));
}
Tensor Tensor::atan2(Scalar other) const {
    return atan2(scalar_to_tensor(other, device()));
}
Tensor Tensor::hypot(Scalar other) const {
    return hypot(scalar_to_tensor(other, device()));
}
Tensor Tensor::logical_and(Scalar other) const {
    return logical_and(scalar_to_tensor(other, device()));
}
Tensor Tensor::logical_or(Scalar other) const {
    return logical_or(scalar_to_tensor(other, device()));
}
Tensor Tensor::logical_xor(Scalar other) const {
    return logical_xor(scalar_to_tensor(other, device()));
}
Tensor Tensor::bitwise_and(Scalar other) const {
    return bitwise_and(scalar_to_tensor(other, device()));
}
Tensor Tensor::bitwise_or(Scalar other) const {
    return bitwise_or(scalar_to_tensor(other, device()));
}
Tensor Tensor::bitwise_xor(Scalar other) const {
    return bitwise_xor(scalar_to_tensor(other, device()));
}
Tensor Tensor::lshift(Scalar other) const {
    return lshift(scalar_to_tensor(other, device()));
}
Tensor Tensor::rshift(Scalar other) const {
    return rshift(scalar_to_tensor(other, device()));
}

// ── §8.6.A binary math (inplace, Tensor operand) ──────────────────────────

Tensor& Tensor::pow_(const Tensor& other) { return copy_(pow(other)); }
Tensor& Tensor::fmod_(const Tensor& other) { return copy_(fmod(other)); }
Tensor& Tensor::remainder_(const Tensor& other) {
    return copy_(remainder(other));
}
Tensor& Tensor::maximum_(const Tensor& other) { return copy_(maximum(other)); }
Tensor& Tensor::minimum_(const Tensor& other) { return copy_(minimum(other)); }
Tensor& Tensor::logical_and_(const Tensor& other) {
    return copy_(logical_and(other));
}
Tensor& Tensor::logical_or_(const Tensor& other) {
    return copy_(logical_or(other));
}
Tensor& Tensor::logical_xor_(const Tensor& other) {
    return copy_(logical_xor(other));
}
Tensor& Tensor::bitwise_and_(const Tensor& other) {
    return copy_(bitwise_and(other));
}
Tensor& Tensor::bitwise_or_(const Tensor& other) {
    return copy_(bitwise_or(other));
}
Tensor& Tensor::bitwise_xor_(const Tensor& other) {
    return copy_(bitwise_xor(other));
}
Tensor& Tensor::lshift_(const Tensor& other) { return copy_(lshift(other)); }
Tensor& Tensor::rshift_(const Tensor& other) { return copy_(rshift(other)); }

// ── §8.6.A binary math (inplace, Scalar operand) ──────────────────────────

Tensor& Tensor::pow_(Scalar other) {
    return pow_(scalar_to_tensor(other, device()));
}
Tensor& Tensor::fmod_(Scalar other) {
    return fmod_(scalar_to_tensor(other, device()));
}
Tensor& Tensor::remainder_(Scalar other) {
    return remainder_(scalar_to_tensor(other, device()));
}
Tensor& Tensor::maximum_(Scalar other) {
    return maximum_(scalar_to_tensor(other, device()));
}
Tensor& Tensor::minimum_(Scalar other) {
    return minimum_(scalar_to_tensor(other, device()));
}

// ── §8.6.A ternary / clamp ─────────────────────────────────────────────────

Tensor Tensor::where(const Tensor& cond, const Tensor& a, const Tensor& b) {
    return ternary_ew(cond, a, b, kernel::TernaryEWOpCode::Where);
}

Tensor Tensor::clamp(const std::optional<Scalar>& min,
                     const std::optional<Scalar>& max) const {
    Tensor result = clone();
    if (min.has_value())
        result = result.maximum(scalar_to_tensor(*min, device()));
    if (max.has_value())
        result = result.minimum(scalar_to_tensor(*max, device()));
    return result;
}

Tensor Tensor::clamp(Scalar min, Scalar max) const {
    return clamp(std::optional<Scalar>(min), std::optional<Scalar>(max));
}

Tensor Tensor::clamp_min(Scalar min) const {
    return clamp(std::optional<Scalar>(min), std::nullopt);
}

Tensor Tensor::clamp_max(Scalar max) const {
    return clamp(std::nullopt, std::optional<Scalar>(max));
}

Tensor& Tensor::clamp_(Scalar min, Scalar max) {
    maximum_(scalar_to_tensor(min, device()));
    minimum_(scalar_to_tensor(max, device()));
    return *this;
}

Tensor& Tensor::clamp_min_(Scalar min) {
    return maximum_(scalar_to_tensor(min, device()));
}

Tensor& Tensor::clamp_max_(Scalar max) {
    return minimum_(scalar_to_tensor(max, device()));
}

// ---------------------------------------------------------------------------
// Reductions
//
// Dtype rules (PyTorch/NumPy-friendly): sum/min/max preserve the input dtype;
// mean of an integer tensor is Float, else the input dtype. The no-arg forms
// collapse every axis to a 0-d scalar.
// ---------------------------------------------------------------------------

Tensor Tensor::sum() const {
    return do_reduction(*this,
                        IntArrayRef(all_dims(dim())),
                        /*keepdim=*/false,
                        scalar_type(),
                        kernel::ReductionOpCode::Sum);
}
Tensor Tensor::mean() const {
    const ScalarType dt =
        isFloatingType(scalar_type()) ? scalar_type() : ScalarType::Float;
    return do_reduction(*this,
                        IntArrayRef(all_dims(dim())),
                        /*keepdim=*/false,
                        dt,
                        kernel::ReductionOpCode::Mean);
}
Tensor Tensor::max() const {
    return do_reduction(*this,
                        IntArrayRef(all_dims(dim())),
                        /*keepdim=*/false,
                        scalar_type(),
                        kernel::ReductionOpCode::Max);
}
Tensor Tensor::min() const {
    return do_reduction(*this,
                        IntArrayRef(all_dims(dim())),
                        /*keepdim=*/false,
                        scalar_type(),
                        kernel::ReductionOpCode::Min);
}

Tensor Tensor::sum(IntArrayRef dims, bool keepdim) const {
    return do_reduction(
        *this, dims, keepdim, scalar_type(), kernel::ReductionOpCode::Sum);
}
Tensor Tensor::mean(IntArrayRef dims, bool keepdim) const {
    const ScalarType dt =
        isFloatingType(scalar_type()) ? scalar_type() : ScalarType::Float;
    return do_reduction(
        *this, dims, keepdim, dt, kernel::ReductionOpCode::Mean);
}
Tensor Tensor::max(IntArrayRef dims, bool keepdim) const {
    return do_reduction(
        *this, dims, keepdim, scalar_type(), kernel::ReductionOpCode::Max);
}
Tensor Tensor::min(IntArrayRef dims, bool keepdim) const {
    return do_reduction(
        *this, dims, keepdim, scalar_type(), kernel::ReductionOpCode::Min);
}

// ── §8.6.B extended reductions ───────────────────────────────────────────────

Tensor Tensor::prod() const {
    return do_reduction(*this,
                        IntArrayRef(all_dims(dim())),
                        /*keepdim=*/false,
                        scalar_type(),
                        kernel::ReductionOpCode::Prod);
}
Tensor Tensor::prod(IntArrayRef dims, bool keepdim) const {
    return do_reduction(
        *this, dims, keepdim, scalar_type(), kernel::ReductionOpCode::Prod);
}

// argmin/argmax: output is always Long (int64).
namespace {
Tensor do_arg_reduction(const Tensor& self,
                        IntArrayRef dims,
                        bool keepdim,
                        kernel::ArgReduceOp op) {
    const int64_t ndim = self.dim();
    const std::vector<int64_t> rdims = resolve_reduction_dims(dims, ndim);
    const std::vector<int64_t> keep_vec =
        core::ReductionShape(self.sizes(), IntArrayRef(rdims), true);
    const Tensor::ShapeVector keep_shape(keep_vec.begin(), keep_vec.end());
    Tensor dst(keep_shape, ScalarType::Long, self.device());
    if (rdims.empty()) {
        dst.zero_();  // no reduction: every element is arg of itself → idx 0
    } else {
        kernel::ArgReduce(self, dst, IntArrayRef(rdims), op);
    }
    if (keepdim) {
        return dst;
    }
    const std::vector<int64_t> flat_vec =
        core::ReductionShape(self.sizes(), IntArrayRef(rdims), false);
    return dst.reshape(flat_vec);
}
}  // namespace

Tensor Tensor::argmin() const {
    return do_arg_reduction(*this,
                            IntArrayRef(all_dims(dim())),
                            false,
                            kernel::ArgReduceOp::ArgMin);
}
Tensor Tensor::argmin(IntArrayRef dims, bool keepdim) const {
    return do_arg_reduction(*this, dims, keepdim, kernel::ArgReduceOp::ArgMin);
}
Tensor Tensor::argmax() const {
    return do_arg_reduction(*this,
                            IntArrayRef(all_dims(dim())),
                            false,
                            kernel::ArgReduceOp::ArgMax);
}
Tensor Tensor::argmax(IntArrayRef dims, bool keepdim) const {
    return do_arg_reduction(*this, dims, keepdim, kernel::ArgReduceOp::ArgMax);
}

// nanmin/nanmax: for floating types use the NaN-skipping kernel; for integral
// types fall through to regular min/max.
Tensor Tensor::nanmin() const {
    if (isFloatingType(scalar_type())) {
        return do_reduction(*this,
                            IntArrayRef(all_dims(dim())),
                            false,
                            scalar_type(),
                            kernel::ReductionOpCode::NanMin);
    }
    return min();
}
Tensor Tensor::nanmin(IntArrayRef dims, bool keepdim) const {
    if (isFloatingType(scalar_type())) {
        return do_reduction(*this,
                            dims,
                            keepdim,
                            scalar_type(),
                            kernel::ReductionOpCode::NanMin);
    }
    return min(dims, keepdim);
}
Tensor Tensor::nanmax() const {
    if (isFloatingType(scalar_type())) {
        return do_reduction(*this,
                            IntArrayRef(all_dims(dim())),
                            false,
                            scalar_type(),
                            kernel::ReductionOpCode::NanMax);
    }
    return max();
}
Tensor Tensor::nanmax(IntArrayRef dims, bool keepdim) const {
    if (isFloatingType(scalar_type())) {
        return do_reduction(*this,
                            dims,
                            keepdim,
                            scalar_type(),
                            kernel::ReductionOpCode::NanMax);
    }
    return max(dims, keepdim);
}

// all/any: cast input to Bool, reduce, output is Bool.
Tensor Tensor::all() const {
    Tensor src = this->to(ScalarType::Bool);
    return do_reduction(src,
                        IntArrayRef(all_dims(dim())),
                        false,
                        ScalarType::Bool,
                        kernel::ReductionOpCode::All);
}
Tensor Tensor::all(IntArrayRef dims, bool keepdim) const {
    Tensor src = this->to(ScalarType::Bool);
    return do_reduction(
        src, dims, keepdim, ScalarType::Bool, kernel::ReductionOpCode::All);
}
Tensor Tensor::any() const {
    Tensor src = this->to(ScalarType::Bool);
    return do_reduction(src,
                        IntArrayRef(all_dims(dim())),
                        false,
                        ScalarType::Bool,
                        kernel::ReductionOpCode::Any);
}
Tensor Tensor::any(IntArrayRef dims, bool keepdim) const {
    Tensor src = this->to(ScalarType::Bool);
    return do_reduction(
        src, dims, keepdim, ScalarType::Bool, kernel::ReductionOpCode::Any);
}

// count_nonzero: (src != 0).to(Long).sum()
Tensor Tensor::count_nonzero() const {
    return this->ne(Scalar(0)).to(ScalarType::Long).sum();
}
Tensor Tensor::count_nonzero(IntArrayRef dims, bool keepdim) const {
    return this->ne(Scalar(0)).to(ScalarType::Long).sum(dims, keepdim);
}

// ── §8.6.B statistical functions (pure Tensor composition) ───────────────────

Tensor Tensor::var(IntArrayRef dims, bool keepdim, bool unbiased) const {
    ScalarType dt =
        isFloatingType(scalar_type()) ? scalar_type() : ScalarType::Float;
    Tensor src = (scalar_type() == dt) ? *this : this->to(dt);
    // mean_x with keepdim for broadcasting back
    Tensor mean_x = src.mean(dims, /*keepdim=*/true);
    Tensor diff = src.sub(mean_x);
    Tensor sq = diff.mul(diff);
    // Compute N = product of reduced dim sizes
    const int64_t ndim = src.dim();
    const std::vector<int64_t> rdims = resolve_reduction_dims(dims, ndim);
    int64_t n = 1;
    for (int64_t d : rdims) {
        n *= src.size(d);
    }
    int64_t divisor = unbiased ? (n > 1 ? n - 1 : 1) : n;
    if (divisor <= 0) {
        divisor = 1;
    }
    double inv = 1.0 / static_cast<double>(divisor);
    Tensor sum_sq = sq.sum(dims, /*keepdim=*/true);
    // Use the same dtype as the source for the scalar to avoid promotion.
    Tensor result;
    if (dt == ScalarType::Double) {
        result = sum_sq.mul(Scalar(inv));
    } else {
        result = sum_sq.mul(Scalar(static_cast<float>(inv)));
    }
    if (!keepdim) {
        const std::vector<int64_t> flat_vec =
            core::ReductionShape(src.sizes(), IntArrayRef(rdims), false);
        result = result.reshape(flat_vec);
    }
    return result;
}
Tensor Tensor::var(bool unbiased) const {
    return var(IntArrayRef(all_dims(dim())), false, unbiased);
}

Tensor Tensor::std(IntArrayRef dims, bool keepdim, bool unbiased) const {
    return var(dims, keepdim, unbiased).sqrt();
}
Tensor Tensor::std(bool unbiased) const { return var(unbiased).sqrt(); }

Tensor Tensor::norm(const Scalar& p, IntArrayRef dims, bool keepdim) const {
    ScalarType dt =
        isFloatingType(scalar_type()) ? scalar_type() : ScalarType::Float;
    Tensor src = (scalar_type() == dt) ? *this : this->to(dt);
    // Infinity norms (PyTorch parity): +inf -> max |x|, -inf -> min |x|.
    // Check before converting to int64, which is undefined for non-finite
    // doubles (a bare inf used to fall through to the p==0 branch, silently
    // turning norm(inf) into the L0 count).
    const double p_d = p.toDouble();
    if (std::isinf(p_d)) {
        Tensor abs_x = src.abs();
        return p_d > 0 ? abs_x.max(dims, keepdim) : abs_x.min(dims, keepdim);
    }
    int p_int = static_cast<int>(p.toInt64());
    if (p_int == 0) {
        return src.ne(Scalar(0)).to(ScalarType::Long).sum(dims, keepdim).to(dt);
    }
    Tensor abs_x = src.abs();
    if (p_int == 1) {
        return abs_x.sum(dims, keepdim);
    }
    if (p_int == 2) {
        return abs_x.pow(Scalar(2)).sum(dims, keepdim).sqrt();
    }
    if (p_int == -1) {
        // Infinity norm: max of absolute values.
        return abs_x.max(dims, keepdim);
    }
    ZT_LOG_ERROR("norm: unsupported p={} (only 0, 1, 2, -1=inf)", p_int);
}
Tensor Tensor::norm(const Scalar& p) const {
    return norm(p, IntArrayRef(all_dims(dim())), /*keepdim=*/false);
}

// ── §8.6.B scan methods ──────────────────────────────────────────────────────

namespace {

// Inclusive argmax/argmin scan along `dim` of a *contiguous* CPU tensor.
// Returns the int64 index tensor tracking the position of the running
// extreme value; tie-breaking keeps the earliest (smallest) index.
//
// Pre: `src` is row-major contiguous, so flat element indices are safe to
// walk via decoded (row, j) coordinates. Typed comparison (no memcmp) so
// +0.0 / -0.0 compare equal and NaN behaviour is the C++ default.
template<typename T>
Tensor cum_extreme_indices_cpu(const Tensor& src, int64_t dim, bool is_max) {
    const int64_t ndim = src.dim();
    const int64_t scan_len = src.size(dim);
    const int64_t num_rows = src.numel() / scan_len;
    // `src` is contiguous → strides are the standard row-major ones, so we
    // can decode a flat offset from (row, j) directly.
    const int64_t scan_stride = src.stride(dim);
    std::vector<int64_t> ns_strides;
    std::vector<int64_t> ns_sizes;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            ns_strides.push_back(src.stride(d));
            ns_sizes.push_back(src.size(d));
        }
    }
    const int64_t ns_ndim = static_cast<int64_t>(ns_sizes.size());

    Tensor indices = zt::empty_like(src, zt::dtype(ScalarType::Long));
    auto* idx_ptr = indices.data_ptr<int64_t>();
    const auto* sptr = src.data_ptr<T>();

#pragma omp parallel for
    for (int64_t row = 0; row < num_rows; ++row) {
        int64_t base = 0;
        int64_t r = row;
        for (int64_t d = ns_ndim - 1; d >= 0; --d) {
            const auto sz = ns_sizes[static_cast<std::size_t>(d)];
            base += (r % sz) * ns_strides[static_cast<std::size_t>(d)];
            r /= sz;
        }
        int64_t best_idx = 0;
        idx_ptr[base] = 0;
        for (int64_t j = 1; j < scan_len; ++j) {
            const int64_t off = base + (j * scan_stride);
            const int64_t best_off = base + (best_idx * scan_stride);
            const T cur = sptr[off];
            const T best = sptr[best_off];
            const bool better = is_max ? (cur > best) : (cur < best);
            if (better) {
                best_idx = j;
            }
            idx_ptr[off] = best_idx;
        }
    }
    return indices;
}

// Dispatch cum_extreme_indices_cpu on `src.scalar_type()` (Bool included so
// cummax/cummin on a Bool tensor works — comparison is well-defined).
Tensor cum_extreme_indices(const Tensor& src, int64_t dim, bool is_max) {
    return ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE_WITH_BOOL(src.scalar_type(), [&] {
        return cum_extreme_indices_cpu<scalar_t>(src, dim, is_max);
    });
}

}  // namespace

Tensor Tensor::cumsum(int64_t dim) const {
    // Normalize to contiguous: ScanCPU/CUDA index `dst` (allocated by
    // empty_like, hence contiguous) using `src.stride(...)`, so a
    // non-contiguous `*this` (e.g. a transpose) would otherwise corrupt the
    // scan. `.contiguous()` is a zero-cost nop when already contiguous.
    Tensor src = this->contiguous();
    const int64_t wdim = wrap_dim(dim, src.dim());
    Tensor dst = empty_like(src);
    kernel::Scan(src, dst, wdim, kernel::ScanOpCode::CumSum);
    return dst;
}
Tensor Tensor::cumprod(int64_t dim) const {
    Tensor src = this->contiguous();
    const int64_t wdim = wrap_dim(dim, src.dim());
    Tensor dst = empty_like(src);
    kernel::Scan(src, dst, wdim, kernel::ScanOpCode::CumProd);
    return dst;
}
std::pair<Tensor, Tensor> Tensor::cummax(int64_t dim) const {
    Tensor src = this->contiguous();
    const int64_t wdim = wrap_dim(dim, src.dim());
    Tensor values = empty_like(src);
    kernel::Scan(src, values, wdim, kernel::ScanOpCode::CumMax);
    Tensor indices;
    if (src.is_cpu()) {
        // Typed comparison (no memcmp): correct for ±0.0 and follows C++ NaN
        // ordering. Indices share the scan loop's strides, which now match
        // the contiguous `values`/`src`.
        indices = cum_extreme_indices(src, wdim, /*is_max=*/true);
    } else {
        // CUDA path: stage through CPU (matches the existing CUDA Scan
        // strategy — native CUB impl is a tracked TODO).
        Tensor src_cpu = src.cpu();
        auto [val_cpu, idx_cpu] = src_cpu.cummax(wdim);
        indices = zt::empty_like(values, zt::dtype(ScalarType::Long));
        indices.copy_(idx_cpu);
    }
    return {values, indices};
}
std::pair<Tensor, Tensor> Tensor::cummin(int64_t dim) const {
    Tensor src = this->contiguous();
    const int64_t wdim = wrap_dim(dim, src.dim());
    Tensor values = empty_like(src);
    kernel::Scan(src, values, wdim, kernel::ScanOpCode::CumMin);
    Tensor indices;
    if (src.is_cpu()) {
        indices = cum_extreme_indices(src, wdim, /*is_max=*/false);
    } else {
        Tensor src_cpu = src.cpu();
        auto [val_cpu, idx_cpu] = src_cpu.cummin(wdim);
        indices = zt::empty_like(values, zt::dtype(ScalarType::Long));
        indices.copy_(idx_cpu);
    }
    return {values, indices};
}

// ---------------------------------------------------------------------------
// construction / repetition (DESIGN §8.6.C)
// ---------------------------------------------------------------------------

// repeat() does a real copy along every axis (unlike expand, which only
// broadcasts size-1 dims via stride 0). The output always owns a fresh blob.
//
// Trick (matches PyTorch's implementation): build a 2*rank view of the source
// whose axes are interleaved as [reps0, src0, reps1, src1, ...] and whose
// strides are [0, st0, 0, st1, ...] — i.e. a stride-0 "reps" axis immediately
// followed by its corresponding source axis. Making that view contiguous
// materializes one full copy per tile, in the row-major order PyTorch
// expects (each source row is tiled along its own axis before the next axis
// repeats the block). A final reshape collapses each (ri, si) pair into ri*si.
Tensor Tensor::repeat(IntArrayRef repeats) const {
    const auto ndim = this->dim();
    ZT_CHECK(static_cast<int64_t>(repeats.size()) >= ndim,
             "repeat: repeats rank {} < tensor ndim {}",
             repeats.size(),
             ndim);
    for (std::size_t i = 0; i < repeats.size(); ++i) {
        ZT_CHECK(repeats[i] >= 0,
                 "repeat: count {} must be non-negative",
                 repeats[i]);
    }
    // Prepend size-1 axes so the source rank matches `repeats`.
    const std::size_t extra = repeats.size() - ndim;
    Tensor src = *this;
    for (std::size_t i = 0; i < extra; ++i) src = src.unsqueeze(0);
    const std::size_t rank = repeats.size();

    // Interleaved [reps_i, src_i, ...] view with reps-stride 0.
    ShapeVector view_shape;
    ShapeVector view_strides;
    view_shape.reserve(2 * rank);
    view_strides.reserve(2 * rank);
    for (std::size_t i = 0; i < rank; ++i) {
        view_shape.push_back(repeats[i]);  // reps axis
        view_strides.push_back(0);         // stride 0 -> reuse
        view_shape.push_back(src.size(static_cast<int64_t>(i)));  // src axis
        view_strides.push_back(src.stride(static_cast<int64_t>(i)));
    }
    Tensor tiled(
        view_shape, view_strides, src.data_ptr(), dtype_, src.GetBlob());
    // Materialize (single contiguous copy) and collapse the interleaved pairs.
    ShapeVector out_shape;
    out_shape.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i)
        out_shape.push_back(repeats[i] * src.size(static_cast<int64_t>(i)));
    return tiled.contiguous().reshape(
        IntArrayRef(out_shape.data(), out_shape.size()));
}

// NumPy tile: if reps has more entries than ndim, prepend new size-1 axes; if
// fewer, the missing leading entries are treated as 1. Then it's just repeat.
Tensor Tensor::tile(IntArrayRef reps) const {
    const auto ndim = this->dim();
    ShapeVector full_reps;
    if (static_cast<std::size_t>(ndim) >= reps.size()) {
        const std::size_t lead = static_cast<std::size_t>(ndim) - reps.size();
        for (std::size_t i = 0; i < lead; ++i) full_reps.push_back(1);
    }
    for (const auto r : reps) {
        ZT_CHECK(r >= 0, "tile: rep {} must be non-negative", r);
        full_reps.push_back(r);
    }
    // Prepend size-1 axes so the source rank matches full_reps.
    Tensor src = *this;
    while (static_cast<std::size_t>(src.dim()) < full_reps.size())
        src = src.unsqueeze(0);
    return src.repeat(IntArrayRef(full_reps.data(), full_reps.size()));
}

std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    return zt::split(*this, split_size, dim);
}

std::vector<Tensor> Tensor::split(IntArrayRef split_sizes, int64_t dim) const {
    return zt::split(*this, split_sizes, dim);
}

std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    return zt::chunk(*this, chunks, dim);
}

// ---------------------------------------------------------------------------
// linear algebra (DESIGN §8.5.A)
// ---------------------------------------------------------------------------

Tensor Tensor::mm(const Tensor& other) const {
    check_matmul_dtype(scalar_type());
    check_matmul_dtype(other.scalar_type());
    return do_mm(*this, other);
}

Tensor Tensor::bmm(const Tensor& other) const {
    check_matmul_dtype(scalar_type());
    check_matmul_dtype(other.scalar_type());
    return do_bmm(*this, other);
}

Tensor Tensor::matmul(const Tensor& other) const {
    check_matmul_dtype(scalar_type());
    check_matmul_dtype(other.scalar_type());
    return do_matmul(*this, other);
}

Tensor Tensor::addmm(const Tensor& A,
                     const Tensor& B,
                     Scalar beta,
                     Scalar alpha) const {
    check_matmul_dtype(scalar_type());
    check_matmul_dtype(A.scalar_type());
    check_matmul_dtype(B.scalar_type());
    return do_addmm(*this, A, B, beta, alpha);
}

// ---------------------------------------------------------------------------
// indexing
// ---------------------------------------------------------------------------

namespace {

// Build a new Tensor view that drops `dim`, with data_ptr advanced by idx along
// it. This is the primitive behind integer-indexing: t[i] removes axis 0.
// (Open3D names this IndexExtract.)
Tensor index_extract(const Tensor& t, int64_t dim, int64_t idx) {
    const int64_t ndim = t.dim();
    dim = wrap_dim(dim, ndim);
    const int64_t len = t.size(dim);
    int64_t wrapped = idx;
    if (wrapped < 0) wrapped += len;
    ZT_CHECK(wrapped >= 0 && wrapped < len,
             "index: integer {} out of range for dim {} (size {})",
             idx,
             dim,
             len);
    Tensor::ShapeVector ns, nst;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        ns.push_back(t.sizes()[static_cast<std::size_t>(d)]);
        nst.push_back(t.strides()[static_cast<std::size_t>(d)]);
    }
    auto* base = const_cast<std::uint8_t*>(
                     static_cast<const std::uint8_t*>(t.data_ptr())) +
                 wrapped * t.stride(dim) * t.element_size();
    return Tensor(ns, nst, base, t.scalar_type(), t.GetBlob());
}

// A 0-d int64 scalar tensor on `device`; the "this dim is a full slice"
// sentinel consumed by the advanced-indexing preprocessor.
Tensor slice_sentinel(const Device& device) {
    Tensor s(Tensor::ShapeVector{}, ScalarType::Long, device);
    return s;  // value irrelevant; only its 0-d shape matters.
}

// Expand a bool Tensor key into the per-dim int64 index tensors via NonZero,
// matching the dims the key occupies (starting at `dim_offset`). Returns the
// expanded index tensors (one per occupied dim) and the number of dims
// consumed.
std::vector<Tensor> expand_bool_key(const Tensor& key, int64_t dim_offset) {
    auto coords = kernel::NonZero(key);
    ZT_CHECK(static_cast<int64_t>(coords.size()) == key.dim(),
             "index: NonZero dim count mismatch");
    return coords;
}

// Number of source dims consumed by the keys before `pos`. A Bool mask at
// `pos` covers the next mask.dim() source dims (NumPy/PyTorch semantics), so
// this is where its prefix match must start.
int64_t bool_mask_dim_offset(const std::vector<TensorKey>& keys,
                             std::size_t pos) {
    int64_t off = 0;
    for (std::size_t j = 0; j < pos; ++j) {
        const auto& k = keys[j];
        if (k.IsIndex() || k.IsSlice()) {
            ++off;
        } else if (k.IsTensor()) {
            const Tensor& idx = k.GetTensor();
            off += (idx.scalar_type() == ScalarType::Bool) ? idx.dim() : 1;
        }
        // NewAxis consumes no source dim (rejected together with Tensor keys).
    }
    return off;
}

// A Bool mask must prefix-match the dims it covers (PyTorch: "The shape of the
// mask [..] at index .. does not match the shape of the indexed tensor").
// This is a shape contract, so it stays on in release builds. Without it the
// NonZero coords (valid offsets into the *mask* shape) would be fed to the
// gather/scatter kernel as offsets into the *tensor* shape, silently reading
// or writing out of bounds.
void check_bool_mask_shape(const Tensor& tensor,
                           const Tensor& mask,
                           int64_t dim_offset) {
    for (int64_t d = 0; d < mask.dim(); ++d) {
        ZT_CHECK(mask.size(d) == tensor.size(dim_offset + d),
                 "index: the shape of the boolean mask at dim {} (size {}) "
                 "does not match the shape of the indexed tensor at dim {} "
                 "(size {})",
                 d,
                 mask.size(d),
                 dim_offset + d,
                 tensor.size(dim_offset + d));
    }
}

// Expand any Ellipsis (`...`) key into the right number of full slices and
// validate the key list. NumPy semantics: a single ellipsis stands for
// `(ndim - consumed)` full slices, where `consumed` is the count of source
// dims the other keys select (an Index/Slice/int64-Tensor consumes one dim;
// a Bool mask consumes as many dims as it spans). NewAxis keys are left in
// place — the basic indexing walk unsqueezes them, and advanced indexing
// rejects them (NewAxis + Tensor keys is unsupported in this version; see
// DESIGN §8.6.D risk note on simplifying the advanced+NewAxis interaction).
std::vector<TensorKey> expand_ellipsis(const std::vector<TensorKey>& keys,
                                       int64_t ndim) {
    int64_t consumed = 0;
    int64_t ellipsis_count = 0;
    for (const auto& k : keys) {
        if (k.IsEllipsis()) {
            ++ellipsis_count;
        } else if (k.IsNewAxis()) {
            // Inserts a size-1 dim; consumes no source dim.
        } else if (k.IsIndex() || k.IsSlice()) {
            ++consumed;
        } else {  // Tensor (int64 index or bool mask)
            const Tensor& idx = k.GetTensor();
            consumed += (idx.scalar_type() == ScalarType::Bool) ? idx.dim() : 1;
        }
    }
    ZT_CHECK(ellipsis_count <= 1,
             "index: an index can only have a single ellipsis ('...')");
    ZT_CHECK(consumed <= ndim,
             "index: too many indices for tensor of dimension {}",
             ndim);

    std::vector<TensorKey> out;
    out.reserve(keys.size());
    const int64_t fill =
        ndim - consumed;  // full slices the ellipsis expands to
    for (const auto& k : keys) {
        if (k.IsEllipsis()) {
            for (int64_t j = 0; j < fill; ++j) {
                out.push_back(TensorKey::Slice(None, None, None));
            }
        } else {
            out.push_back(k);
        }
    }
    return out;
}

bool has_key(const std::vector<TensorKey>& keys,
             bool (*pred)(const TensorKey&)) {
    for (const auto& k : keys) {
        if (pred(k)) return true;
    }
    return false;
}

// Run the advanced-indexing gather (GET) over `t` with the given Tensor keys
// already expanded to int64. Returns the gathered result.
Tensor advanced_index_get(const Tensor& t, const std::vector<TensorKey>& keys) {
    // Step 1: build a basic-slice pre-pass view. Replace every Index/Tensor key
    // with a full Slice(None,None,None); pass Slice keys through. The result is
    // a pure view (no data move).
    std::vector<TensorKey> pre_keys;
    pre_keys.reserve(keys.size());
    for (const auto& k : keys) {
        if (k.IsSlice()) {
            pre_keys.push_back(k);
        } else {
            pre_keys.push_back(TensorKey::Slice(None, None, None));
        }
    }
    Tensor pre = t.index(pre_keys);  // basic path (no Tensor keys here)

    // Step 2: build the index_tensors list for the advanced engine.
    //   Index    -> {idx} (shape {1}, Int64)
    //   Slice    -> 0-d Int64 sentinel (this dim is fully sliced)
    //   Tensor   -> its (int64) tensor
    std::vector<Tensor> index_tensors;
    index_tensors.reserve(pre.dim());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const TensorKey& k = keys[i];
        if (k.IsIndex()) {
            Tensor s(Tensor::ShapeVector{1}, ScalarType::Long, t.device());
            s.data_ptr<int64_t>()[0] = k.GetIndex();
            index_tensors.push_back(s);
        } else if (k.IsSlice()) {
            index_tensors.push_back(slice_sentinel(t.device()));
        } else {
            Tensor idx = k.GetTensor();
            if (idx.scalar_type() == ScalarType::Bool) {
                // Bool mask: replace this single key with the NonZero coords
                // of the dims it covers. The number of output index tensors
                // equals idx.dim(); they occupy consecutive positions. The
                // mask must prefix-match the dims it covers (PyTorch parity).
                const int64_t dim_offset = bool_mask_dim_offset(keys, i);
                check_bool_mask_shape(pre, idx, dim_offset);
                auto coords = expand_bool_key(idx, dim_offset);
                for (auto& c : coords) {
                    if (c.device() != t.device()) c = c.to(t.device());
                    index_tensors.push_back(std::move(c));
                }
            } else {
                if (idx.scalar_type() != ScalarType::Long) {
                    idx = idx.to(ScalarType::Long);
                }
                if (idx.device() != t.device()) idx = idx.to(t.device());
                index_tensors.push_back(idx);
            }
        }
    }
    // Pad trailing dims (those without an explicit key) with slice sentinels.
    while (static_cast<int64_t>(index_tensors.size()) < pre.dim()) {
        index_tensors.push_back(slice_sentinel(t.device()));
    }

    core::AdvancedIndexPreprocessor aip(pre, index_tensors);
    const auto& out_shape = aip.GetOutputShape();
    Tensor::ShapeVector out_sv(out_shape.begin(), out_shape.end());
    Tensor dst(out_sv, pre.scalar_type(), pre.device());
    kernel::IndexGet(aip.GetTensor(),
                     dst,
                     aip.GetIndexTensors(),
                     aip.GetIndexedShape(),
                     aip.GetIndexedStrides());
    return dst;
}

// Stage an int64 `index` tensor for the dim-based scatter/gather kernels: cast
// to Int64, materialize a contiguous copy if needed (the kernels decode linear
// positions assuming row-major index), and place it on `dev`. Cheap (Tensor is
// a ref-counted view; only a non-contiguous or cross-device index copies).
Tensor stage_long_index(const Tensor& index, const Device& dev) {
    Tensor idx = (index.scalar_type() == ScalarType::Long)
                     ? index
                     : index.to(ScalarType::Long);
    if (!idx.is_contiguous()) {
        idx = idx.contiguous();
    }
    if (idx.device() != dev) {
        idx = idx.to(dev);
    }
    return idx;
}

// Validate that every entry of a staged int64 `index` is in [-size, size)
// (PyTorch parity). Negative indices wrap (handled in the kernels), but values
// outside that range would read/write out-of-bounds memory, so reject them
// BEFORE the scatter/gather kernel runs (the operand stays untouched on error).
//
// Debug-only (compiled out under NDEBUG): the scan is O(numel) and on CUDA it
// adds a stream sync, so release builds trust the caller instead — the kernels
// assume valid indices, matching the minimal-library performance goal. See
// kernel/CheckIndexBounds.{h,cpp} for the per-device implementations (CPU:
// parallel scan; CUDA: grid-stride kernel + O(1) result D2H).
void check_index_bounds(const Tensor& idx, int64_t dim_size, const char* op) {
#ifndef NDEBUG
    // dim_size == 0 is handled inside the scan: [-0, 0) is empty, so every
    // non-empty index is rejected (PyTorch parity).
    kernel::CheckIndexBounds(idx, dim_size, op);
#else
    (void)idx;
    (void)dim_size;
    (void)op;
#endif
}

}  // namespace

Tensor Tensor::index(const std::vector<TensorKey>& keys) const {
    ZT_CHECK(defined(), "index: tensor is undefined");
    // Normalize: expand Ellipsis into full slices (NewAxis left in place).
    auto expanded = expand_ellipsis(keys, dim());
    const bool has_tensor_key =
        has_key(expanded, [](const TensorKey& k) { return k.IsTensor(); });
    if (has_tensor_key) {
        ZT_CHECK(!has_key(expanded,
                          [](const TensorKey& k) { return k.IsNewAxis(); }),
                 "index: NewAxis (None) combined with advanced (Tensor) "
                 "indexing is not supported");
        return advanced_index_get(*this, expanded);
    }
    // Basic path: int + slice + NewAxis -> pure view (zero-copy).
    Tensor t = *this;
    int64_t slice_dim = 0;  // axis in the running tensor, counting only
                            // non-Index keys (Index drops an axis; NewAxis
                            // inserts one, both advancing slice_dim past the
                            // new/removed position)
    for (const auto& k : expanded) {
        if (k.IsIndex()) {
            int64_t idx = k.GetIndex();
            t = index_extract(t, slice_dim, idx);
            // IndexExtract removed the axis; slice_dim stays.
        } else if (k.IsSlice()) {
            const auto rs = InstantiateSlice(k.GetSlice(), t.size(slice_dim));
            t = t.slice(slice_dim, rs.start, rs.stop, rs.step);
            ++slice_dim;
        } else if (k.IsNewAxis()) {
            t = t.unsqueeze(slice_dim);
            ++slice_dim;  // the inserted size-1 dim occupies slice_dim; the
                          // next source dim shifted to slice_dim + 1.
        } else {
            ZT_LOG_ERROR("index: unexpected key mode in basic path");
        }
    }
    return t;
}

Tensor& Tensor::index_put_(const std::vector<TensorKey>& keys,
                           const Tensor& value) {
    ZT_CHECK(defined(), "index_put_: tensor is undefined");
    // Normalize: expand Ellipsis into full slices (NewAxis left in place).
    auto expanded = expand_ellipsis(keys, dim());
    const bool has_tensor_key =
        has_key(expanded, [](const TensorKey& k) { return k.IsTensor(); });

    if (!has_tensor_key) {
        // Basic path: build the view (index() handles NewAxis via unsqueeze,
        // still sharing *this's blob), then copy broadcast(value) into it.
        Tensor view = this->index(expanded);
        Tensor v =
            (value.device() == view.device()) ? value : value.to(view.device());
        if (v.scalar_type() != view.scalar_type()) {
            v = v.to(view.scalar_type());
        }
        // 0-d / scalar value broadcasts to the view shape.
        Tensor vb = (v.sizes() == view.sizes()) ? v : v.expand(view.sizes());
        view.copy_(vb);
        return *this;
    }

    ZT_CHECK(
        !has_key(expanded, [](const TensorKey& k) { return k.IsNewAxis(); }),
        "index_put_: NewAxis (None) combined with advanced (Tensor) "
        "indexing is not supported");

    // Advanced path. Rebuild the index_tensors exactly as in advanced_index_get
    // (we duplicate the bookkeeping rather than refactor a shared helper, to
    // keep the SET entry self-contained).
    std::vector<TensorKey> pre_keys;
    pre_keys.reserve(expanded.size());
    for (const auto& k : expanded) {
        if (k.IsSlice())
            pre_keys.push_back(k);
        else
            pre_keys.push_back(TensorKey::Slice(None, None, None));
    }
    Tensor pre = this->index(pre_keys);

    std::vector<Tensor> index_tensors;
    index_tensors.reserve(pre.dim());
    for (std::size_t i = 0; i < expanded.size(); ++i) {
        const TensorKey& k = expanded[i];
        if (k.IsIndex()) {
            Tensor s(Tensor::ShapeVector{1}, ScalarType::Long, device());
            s.data_ptr<int64_t>()[0] = k.GetIndex();
            index_tensors.push_back(s);
        } else if (k.IsSlice()) {
            index_tensors.push_back(slice_sentinel(device()));
        } else {
            Tensor idx = k.GetTensor();
            if (idx.scalar_type() == ScalarType::Bool) {
                // Mask must prefix-match the dims it covers (PyTorch parity);
                // otherwise NonZero coords would index the wrong axes and read
                // or write out of bounds.
                const int64_t dim_offset = bool_mask_dim_offset(expanded, i);
                check_bool_mask_shape(pre, idx, dim_offset);
                auto coords = expand_bool_key(idx, dim_offset);
                for (auto& c : coords) {
                    if (c.device() != device()) c = c.to(device());
                    index_tensors.push_back(std::move(c));
                }
            } else {
                if (idx.scalar_type() != ScalarType::Long) {
                    idx = idx.to(ScalarType::Long);
                }
                if (idx.device() != device()) idx = idx.to(device());
                index_tensors.push_back(idx);
            }
        }
    }
    while (static_cast<int64_t>(index_tensors.size()) < pre.dim()) {
        index_tensors.push_back(slice_sentinel(device()));
    }

    core::AdvancedIndexPreprocessor aip(pre, index_tensors);
    const auto& out_shape = aip.GetOutputShape();
    Tensor::ShapeVector out_sv(out_shape.begin(), out_shape.end());

    // Broadcast `value` to the gather result shape, cast to dst dtype/device.
    Tensor v = (value.device() == device()) ? value : value.to(device());
    if (v.scalar_type() != scalar_type()) v = v.to(scalar_type());
    Tensor vb = (v.sizes() == IntArrayRef(out_sv.data(), out_sv.size()))
                    ? v
                    : v.expand(IntArrayRef(out_sv.data(), out_sv.size()));

    // dst = the restrided view of *this (indexed dims stride 0, shape =
    // output_shape). The Indexer iterates this view at output_shape; the
    // scatter offset (from indexed_shape/strides) advances each write to the
    // real indexed element of *this. aip.GetTensor() shares *this's blob, so
    // writing through it mutates *this.
    kernel::IndexSet(vb,
                     aip.GetTensor(),
                     aip.GetIndexTensors(),
                     aip.GetIndexedShape(),
                     aip.GetIndexedStrides());
    return *this;
}

Tensor& Tensor::index_put_(const std::vector<TensorKey>& keys, Scalar value) {
    // Materialize a 0-d scalar tensor and forward.
    Tensor s(Tensor::ShapeVector{}, value.type(), device());
    s.fill_(value);
    return index_put_(keys, s);
}

Tensor Tensor::operator[](int64_t i) const {
    return index({TensorKey::Index(i)});
}

Tensor Tensor::operator[](const SliceSpec& s) const {
    return index({TensorKey::Slice(s)});
}

Tensor Tensor::operator[](const Tensor& idx) const {
    return index({TensorKey::Tensor(idx)});
}

// ---------------------------------------------------------------------------
// index enhancements (DESIGN §8.6.D)
// ---------------------------------------------------------------------------

namespace {
// True iff `dt` is one of the dtypes whose scatter/index accumulation kernels
// support atomic add (Float/Double/Int32/Int64), matching kernel::IndexAdd /
// kernel::Scatter(accumulate) / Reduction's atomic_supported set.
bool is_accum_dtype(ScalarType dt) {
    return dt == ScalarType::Float || dt == ScalarType::Double ||
           dt == ScalarType::Int || dt == ScalarType::Long;
}
}  // namespace

Tensor& Tensor::index_add_(const std::vector<TensorKey>& keys,
                           const Tensor& value) {
    ZT_CHECK(defined(), "index_add_: tensor is undefined");
    ZT_CHECK(is_accum_dtype(scalar_type()),
             "index_add_: dtype {} not supported for accumulation "
             "(use Float/Double/Int32/Int64)",
             toString(scalar_type()));

    auto expanded = expand_ellipsis(keys, dim());
    const bool has_tensor_key =
        has_key(expanded, [](const TensorKey& k) { return k.IsTensor(); });

    if (!has_tensor_key) {
        // Basic path: basic indexing selects each element at most once, so an
        // in-place add on the view is correct (no duplicate-target race).
        Tensor view = this->index(expanded);
        Tensor v =
            (value.device() == view.device()) ? value : value.to(view.device());
        if (v.scalar_type() != view.scalar_type()) {
            v = v.to(view.scalar_type());
        }
        Tensor vb = (v.sizes() == view.sizes()) ? v : v.expand(view.sizes());
        view.add_(vb);
        return *this;
    }

    ZT_CHECK(
        !has_key(expanded, [](const TensorKey& k) { return k.IsNewAxis(); }),
        "index_add_: NewAxis (None) combined with advanced (Tensor) "
        "indexing is not supported");

    // Advanced path: same SET geometry as index_put_ (the preprocessor's
    // Mode::SET scatters each src[i] to the indexed dst element), but the
    // kernel accumulates instead of overwriting, so duplicate indices sum.
    std::vector<TensorKey> pre_keys;
    pre_keys.reserve(expanded.size());
    for (const auto& k : expanded) {
        pre_keys.push_back(k.IsSlice() ? k
                                       : TensorKey::Slice(None, None, None));
    }
    Tensor pre = this->index(pre_keys);

    std::vector<Tensor> index_tensors;
    index_tensors.reserve(pre.dim());
    for (std::size_t i = 0; i < expanded.size(); ++i) {
        const TensorKey& k = expanded[i];
        if (k.IsIndex()) {
            Tensor s(Tensor::ShapeVector{1}, ScalarType::Long, device());
            s.data_ptr<int64_t>()[0] = k.GetIndex();
            index_tensors.push_back(s);
        } else if (k.IsSlice()) {
            index_tensors.push_back(slice_sentinel(device()));
        } else {
            Tensor idx = k.GetTensor();
            if (idx.scalar_type() == ScalarType::Bool) {
                // Mask must prefix-match the dims it covers (PyTorch parity);
                // otherwise NonZero coords would index the wrong axes and read
                // or write out of bounds.
                const int64_t dim_offset = bool_mask_dim_offset(expanded, i);
                check_bool_mask_shape(pre, idx, dim_offset);
                auto coords = expand_bool_key(idx, dim_offset);
                for (auto& c : coords) {
                    if (c.device() != device()) c = c.to(device());
                    index_tensors.push_back(std::move(c));
                }
            } else {
                if (idx.scalar_type() != ScalarType::Long) {
                    idx = idx.to(ScalarType::Long);
                }
                if (idx.device() != device()) idx = idx.to(device());
                index_tensors.push_back(idx);
            }
        }
    }
    while (static_cast<int64_t>(index_tensors.size()) < pre.dim()) {
        index_tensors.push_back(slice_sentinel(device()));
    }

    core::AdvancedIndexPreprocessor aip(pre, index_tensors);
    const auto& out_shape = aip.GetOutputShape();
    Tensor::ShapeVector out_sv(out_shape.begin(), out_shape.end());

    Tensor v = (value.device() == device()) ? value : value.to(device());
    if (v.scalar_type() != scalar_type()) v = v.to(scalar_type());
    Tensor vb = (v.sizes() == IntArrayRef(out_sv.data(), out_sv.size()))
                    ? v
                    : v.expand(IntArrayRef(out_sv.data(), out_sv.size()));

    kernel::IndexAdd(vb,
                     aip.GetTensor(),
                     aip.GetIndexTensors(),
                     aip.GetIndexedShape(),
                     aip.GetIndexedStrides());
    return *this;
}

Tensor& Tensor::scatter_(int64_t dim, const Tensor& index, const Tensor& src) {
    ZT_CHECK(defined(), "scatter_: tensor is undefined");
    Tensor idx = stage_long_index(index, device());
    Tensor s = (src.device() == device()) ? src : src.to(device());
    if (s.scalar_type() != scalar_type()) {
        s = s.to(scalar_type());
    }
    if (s.sizes() != idx.sizes()) {
        s = s.expand(idx.sizes());
    }
    check_index_bounds(idx, size(core::WrapDim(dim, this->dim())), "scatter_");
    kernel::Scatter(s, idx, *this, dim, /*accumulate=*/false);
    return *this;
}

Tensor& Tensor::scatter_(int64_t dim, const Tensor& index, Scalar value) {
    ZT_CHECK(defined(), "scatter_: tensor is undefined");
    Tensor idx = stage_long_index(index, device());
    Tensor s(idx.sizes(), scalar_type(), device());
    s.fill_(value);
    check_index_bounds(idx, size(core::WrapDim(dim, this->dim())), "scatter_");
    kernel::Scatter(s, idx, *this, dim, /*accumulate=*/false);
    return *this;
}

Tensor& Tensor::scatter_add_(int64_t dim,
                             const Tensor& index,
                             const Tensor& src) {
    ZT_CHECK(defined(), "scatter_add_: tensor is undefined");
    ZT_CHECK(is_accum_dtype(scalar_type()),
             "scatter_add_: dtype {} not supported for accumulation "
             "(use Float/Double/Int32/Int64)",
             toString(scalar_type()));
    Tensor idx = stage_long_index(index, device());
    Tensor s = (src.device() == device()) ? src : src.to(device());
    if (s.scalar_type() != scalar_type()) {
        s = s.to(scalar_type());
    }
    if (s.sizes() != idx.sizes()) {
        s = s.expand(idx.sizes());
    }
    check_index_bounds(
        idx, size(core::WrapDim(dim, this->dim())), "scatter_add_");
    kernel::Scatter(s, idx, *this, dim, /*accumulate=*/true);
    return *this;
}

Tensor Tensor::gather(int64_t dim, const Tensor& index) const {
    ZT_CHECK(defined(), "gather: tensor is undefined");
    Tensor idx = stage_long_index(index, device());
    Tensor dst(idx.sizes(), scalar_type(), device());
    // For gather the index selects out of *this* (the source), so the bounds
    // are this tensor's dim-size, not the destination's.
    check_index_bounds(idx, size(core::WrapDim(dim, this->dim())), "gather");
    kernel::Gather(*this, idx, dst, dim);
    return dst;
}

Tensor& Tensor::index_fill_(const std::vector<TensorKey>& keys, Scalar value) {
    // Semantically index_put_ with a scalar: fill every selected element.
    return index_put_(keys, value);
}

Tensor& Tensor::index_fill_(const std::vector<TensorKey>& keys,
                            const Tensor& value) {
    // value is a (0-d) fill scalar broadcast over the selection.
    return index_put_(keys, value);
}

Tensor& Tensor::index_copy_(const std::vector<TensorKey>& keys,
                            const Tensor& value) {
    // Copy value into the selected positions (shape must match the selection).
    return index_put_(keys, value);
}

Tensor& Tensor::masked_fill_(const Tensor& mask, Scalar value) {
    ZT_CHECK(defined(), "masked_fill_: tensor is undefined");
    ZT_CHECK(mask.scalar_type() == ScalarType::Bool,
             "masked_fill_: mask must be Bool");
    Tensor m = (mask.device() == device()) ? mask : mask.to(device());
    if (m.sizes() != sizes()) m = m.expand(sizes());
    // A single bool mask spanning all dims selects exactly the True elements;
    // index_put_ with a scalar fills them.
    return index_put_({TensorKey::Tensor(m)}, value);
}

Tensor Tensor::masked_select(const Tensor& mask) const {
    ZT_CHECK(defined(), "masked_select: tensor is undefined");
    ZT_CHECK(mask.scalar_type() == ScalarType::Bool,
             "masked_select: mask must be Bool");
    Tensor m = (mask.device() == device()) ? mask : mask.to(device());
    if (m.sizes() != sizes()) m = m.expand(sizes());
    // Bool-mask GET gathers the True elements into a 1-D tensor.
    return this->index({TensorKey::Tensor(m)});
}

Tensor Tensor::nonzero() const {
    ZT_CHECK(defined(), "nonzero: tensor is undefined");
    // NonZero takes a Bool mask; for a numeric tensor, "nonzero" means != 0.
    Tensor mask =
        (scalar_type() == ScalarType::Bool) ? *this : this->ne(Scalar(0));
    if (dim() == 0) {
        // 0-d: a single element with no axes. Result is {1,0} if set, {0,0}
        // else. Read the element on the host: the mask may live in CUDA memory
        // (the ndim>=1 path below routes through kernel::NonZero, which does a
        // D2H, but this 0-d shortcut does not — so materialize a CPU scalar
        // first). item<bool>() is unsafe here for the same reason (it just
        // dereferences data_ptr, device or not).
        const Tensor host_mask = mask.is_cpu() ? mask : mask.to(Device(kCPU));
        const bool is_true = *host_mask.data_ptr<bool>();
        const int64_t n = is_true ? 1 : 0;
        return Tensor(Tensor::ShapeVector{n, 0}, ScalarType::Long, device());
    }
    // NonZero returns one {num_true} column per dim; stack them into {num_true,
    // ndim} (column-major coords -> rows of coordinates).
    auto cols = kernel::NonZero(mask);
    return zt::stack(cols, /*dim=*/1);
}

// ---------------------------------------------------------------------------
// random (DESIGN §8.5.C)
// ---------------------------------------------------------------------------

Tensor& Tensor::uniform_(double from, double to) {
    kernel::Uniform_(*this, from, to);
    return *this;
}

Tensor& Tensor::normal_(double mean, double stddev) {
    kernel::Normal_(*this, mean, stddev);
    return *this;
}

Tensor& Tensor::random_(int64_t from, int64_t to) {
    kernel::Random_(*this, from, to);
    return *this;
}

// ---------------------------------------------------------------------------
// DLPack (DESIGN §8.5.D)
// ---------------------------------------------------------------------------

#include "ztensor/zt/dlpack.h"

namespace {

ScalarType dl_dtype_to_zt(const DLDataType& dt) {
    switch (dt.code) {
        case kDLFloat:
            switch (dt.bits) {
                case 16:
                    return ScalarType::Half;
                case 32:
                    return ScalarType::Float;
                case 64:
                    return ScalarType::Double;
            }
            break;
        case kDLBfloat:
            return dt.bits == 16 ? ScalarType::BFloat16 : ScalarType::Undefined;
        case kDLInt:
            switch (dt.bits) {
                case 8:
                    return ScalarType::Char;
                case 16:
                    return ScalarType::Short;
                case 32:
                    return ScalarType::Int;
                case 64:
                    return ScalarType::Long;
            }
            break;
        case kDLUInt:
            switch (dt.bits) {
                case 8:
                    return ScalarType::Byte;
                // 16/32/64 unsigned not directly representable; map to signed
                case 16:
                    return ScalarType::Short;
                case 32:
                    return ScalarType::Int;
                case 64:
                    return ScalarType::Long;
            }
            break;
        case kDLBool:
            return dt.bits == 8 ? ScalarType::Bool : ScalarType::Undefined;
        default:
            break;
    }
    return ScalarType::Undefined;
}

DLDataType zt_dtype_to_dl(ScalarType st) {
    DLDataType dt;
    dt.lanes = 1;
    switch (st) {
        case ScalarType::Byte:
            dt.code = kDLUInt;
            dt.bits = 8;
            break;
        case ScalarType::Char:
            dt.code = kDLInt;
            dt.bits = 8;
            break;
        case ScalarType::Short:
            dt.code = kDLInt;
            dt.bits = 16;
            break;
        case ScalarType::Int:
            dt.code = kDLInt;
            dt.bits = 32;
            break;
        case ScalarType::Long:
            dt.code = kDLInt;
            dt.bits = 64;
            break;
        case ScalarType::Half:
            dt.code = kDLFloat;
            dt.bits = 16;
            break;
        case ScalarType::BFloat16:
            dt.code = kDLBfloat;  // spec code 4; kDLFloat/16 would be Half
            dt.bits = 16;
            break;
        case ScalarType::Float:
            dt.code = kDLFloat;
            dt.bits = 32;
            break;
        case ScalarType::Double:
            dt.code = kDLFloat;
            dt.bits = 64;
            break;
        case ScalarType::Bool:
            dt.code = kDLBool;
            dt.bits = 8;
            break;
        default:
            dt.code = kDLFloat;
            dt.bits = 32;
            break;
    }
    return dt;
}

DLDevice zt_device_to_dl(const Device& dev) {
    DLDevice dldev;
    if (dev.is_cpu()) {
        dldev.device_type = kDLCPU;
    } else if (dev.is_cuda()) {
        dldev.device_type = kDLCUDA;
    } else {
        dldev.device_type = kDLCPU;  // fallback
    }
    dldev.device_id = dev.is_cpu() ? 0 : dev.index();
    return dldev;
}

Device dl_device_to_zt(const DLDevice& dldev) {
    if (dldev.device_type == kDLCPU) {
        return Device(kCPU);
    }
    if (dldev.device_type == kDLCUDA) {
        return Device(DeviceType::CUDA, static_cast<int8_t>(dldev.device_id));
    }
    return Device(kCPU);  // fallback
}

}  // namespace

DLManagedTensorVersioned* Tensor::ToDLPack() const {
    const int64_t ndim = this->ndimension();

    // Allocate the managed tensor on the heap; the consumer must call deleter.
    auto* dlmt = new DLManagedTensorVersioned;
    std::memset(dlmt, 0, sizeof(*dlmt));

    // DLPack 1.x ABI. The vendored dlpack.h is DLPack 1.2, but we advertise
    // the conservative minimum {1, 0}: the leading fields of the v1.x struct
    // (through `flags`) are ABI-stable across all 1.x minors, and older
    // consumers (NumPy/PyTorch) reject producers whose minor exceeds theirs.
    dlmt->version.major = 1;
    dlmt->version.minor = 0;
    dlmt->flags =
        0;  // reserved; must be 0 (DLPACK_FLAG_BITMASK_READ_ONLY etc.)

    dlmt->dl_tensor.ndim = static_cast<int32_t>(ndim);
    dlmt->dl_tensor.dtype = zt_dtype_to_dl(dtype_);
    dlmt->dl_tensor.device = zt_device_to_dl(device());
    dlmt->dl_tensor.data = static_cast<char*>(data_ptr_);
    dlmt->dl_tensor.byte_offset = 0;

    // Allocate shape array (managed by dlmt, freed by deleter).
    auto* shape_buf = new int64_t[ndim * 2];  // shape + strides together
    auto* stride_buf = shape_buf + ndim;
    for (int64_t i = 0; i < ndim; ++i) {
        shape_buf[i] = shape_[i];
        // DLPack strides are in ELEMENTS (upstream dlpack.h), same unit as
        // ztensor's — copy verbatim. (Earlier this multiplied by esize,
        // wrongly emitting byte strides; consumers then treated them as
        // elements and read garbage on any non-scalar stride.)
        stride_buf[i] = strides_[i];
    }
    dlmt->dl_tensor.shape = shape_buf;
    dlmt->dl_tensor.strides = stride_buf;

    // The deleter keeps the backing Blob alive until the consumer is done.
    // We capture a shared_ptr copy of blob_ to extend its lifetime.
    auto blob_holder = blob_;
    dlmt->manager_ctx = new std::shared_ptr<Blob>(blob_holder);
    dlmt->deleter = [](DLManagedTensorVersioned* self) {
        // Release the Blob reference.
        delete static_cast<std::shared_ptr<Blob>*>(self->manager_ctx);
        // Free shape/strides array.
        delete[] self->dl_tensor.shape;  // shape_buf (strides follows)
        // Free the wrapper.
        delete self;
    };

    return dlmt;
}

Tensor Tensor::FromDLPack(const DLManagedTensorVersioned* dlmt) {
    ZT_CHECK(dlmt != nullptr, "FromDLPack: nullptr argument");
    const DLTensor& dlt = dlmt->dl_tensor;
    const int64_t ndim = static_cast<int64_t>(dlt.ndim);
    ZT_CHECK(ndim >= 0 && ndim <= core::MAX_DIMS,
             "FromDLPack: ndim {} exceeds MAX_DIMS ({})",
             ndim,
             core::MAX_DIMS);

    ScalarType st = dl_dtype_to_zt(dlt.dtype);
    ZT_CHECK(st != ScalarType::Undefined,
             "FromDLPack: unsupported dtype (code={}, bits={})",
             static_cast<int>(dlt.dtype.code),
             static_cast<int>(dlt.dtype.bits));

    Device dev = dl_device_to_zt(dlt.device);

    ShapeVector shape(ndim);
    ShapeVector strides(ndim);

    if (dlt.shape != nullptr) {
        for (int64_t i = 0; i < ndim; ++i) {
            shape[i] = dlt.shape[i];
        }
    }
    if (dlt.strides != nullptr) {
        // DLPack strides are in ELEMENTS (upstream dlpack.h) — same unit as
        // ztensor, so copy verbatim.
        for (int64_t i = 0; i < ndim; ++i) {
            strides[i] = dlt.strides[i];
        }
    } else {
        // No strides → row-major contiguous.
        strides = contiguous_strides(shape);
    }

    void* data = static_cast<char*>(dlt.data) + dlt.byte_offset;

    // Wrap the DLPack memory in a Blob whose deleter invokes the DLPack
    // deleter once. `dlmt` is const, but the original API may cast away const
    // in its deleter call; we capture a copy of dlmt and call its deleter
    // (the DLPack spec says the consumer takes ownership).
    auto* dlmt_copy = dlmt;
    auto blob_deleter = [dlmt_copy](void*) {
        if (dlmt_copy->deleter != nullptr) {
            dlmt_copy->deleter(
                const_cast<DLManagedTensorVersioned*>(dlmt_copy));
        }
    };

    auto blob = std::make_shared<Blob>(dev, data, blob_deleter);
    return Tensor(shape, strides, data, st, blob);
}

Tensor Tensor::FromDLPack(const DLManagedTensor* dlmt) {
    ZT_CHECK(dlmt != nullptr, "FromDLPack(v0): nullptr argument");
    // Wrap the v0 memory in a Blob whose deleter invokes the v0 deleter
    // exactly once on the ORIGINAL struct. v0 strides are in BYTES (unlike
    // v1's elements), so they are divided by the element size below.
    struct V0Context {
        const DLManagedTensor* original;
    };
    auto* ctx = new V0Context{dlmt};

    auto blob_deleter = [ctx](void*) {
        if (ctx->original->deleter != nullptr) {
            ctx->original->deleter(const_cast<DLManagedTensor*>(ctx->original));
        }
        delete ctx;
    };

    // Copy data from v0 struct.
    ScalarType st = dl_dtype_to_zt(dlmt->dl_tensor.dtype);
    ZT_CHECK(st != ScalarType::Undefined, "FromDLPack(v0): unsupported dtype");

    Device dev = dl_device_to_zt(dlmt->dl_tensor.device);
    int64_t ndim = dlmt->dl_tensor.ndim;
    std::size_t esize = elementSize(st);

    ShapeVector shape(ndim);
    ShapeVector strides(ndim);
    if (dlmt->dl_tensor.shape != nullptr) {
        for (int64_t i = 0; i < ndim; ++i) {
            shape[i] = dlmt->dl_tensor.shape[i];
        }
    }
    if (dlmt->dl_tensor.strides != nullptr) {
        for (int64_t i = 0; i < ndim; ++i) {
            strides[i] =
                dlmt->dl_tensor.strides[i] / static_cast<int64_t>(esize);
        }
    } else {
        strides = contiguous_strides(shape);
    }

    void* data =
        static_cast<char*>(dlmt->dl_tensor.data) + dlmt->dl_tensor.byte_offset;
    auto blob = std::make_shared<Blob>(dev, data, blob_deleter);
    return Tensor(shape, strides, data, st, blob);
}

// ---------------------------------------------------------------------------
// string
// ---------------------------------------------------------------------------

std::string Tensor::string() const {
    std::ostringstream oss;
    oss << "zt::Tensor[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i + 1 < shape_.size()) oss << ",";
    }
    oss << "]{" << toString(dtype_) << "," << device().string() << "}";
    return oss.str();
}

}  // namespace zt
