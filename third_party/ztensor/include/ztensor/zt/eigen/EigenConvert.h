// ztensor/zt/eigen/EigenConvert.h
//
// Optional, header-only Eigen <-> ztensor interop. Surfaced through the
// separate CMake target `zt::ztensor_eigen` (INTERFACE), which makes Eigen a
// PUBLIC dependency ONLY for consumers who link it; the core `ztensor` target
// keeps Eigen PRIVATE. Include this header only when you want Eigen types in
// your API — the rest of ztensor is Eigen-free at the public boundary.
//
// Scope:
//   * Batch:  std::vector<E> <-> Tensor, where E is any fixed-size Eigen type.
//             A column-vector element (Vector2f/3f/2d/3d) yields a 2-D
//             {N,Rows} Tensor; a matrix element (Matrix2/3/4) yields a 3-D
//             {N,Rows,Cols} Tensor. One template, rank adapts to the element.
//   * Single: one Eigen matrix <-> 2-D Tensor (from_matrix / to_matrix),
//             supporting fixed (Matrix2/3/4) and dynamic (MatrixX) types and
//             any direct-access expression (Map / Block / Transpose).
//
// Built entirely on zt::from_blob (zero-copy view) and Tensor accessors;
// nothing here touches ztensor internals.
//
// Storage layout (the matrix-specific subtlety):
//   Eigen defaults to COLUMN-major; ztensor is ROW-major. Conversions never
//   transpose data — the Tensor's strides describe the Eigen object's REAL
//   layout instead:
//     * A column-major matrix wraps as a strided (non-row-major-contiguous)
//       2-D Tensor with strides {1, rows} (element units). Element [i,j] still
//       indexes the same value; is_contiguous() is false, and any op needing
//       contiguity copies on demand.
//     * A row-major Eigen matrix wraps as a contiguous 2-D Tensor.
//   from_matrix reads Eigen's own rowStride()/colStride() (which already fold
//   the storage order in), so transpose views and blocks wrap correctly too.
//
// Alignment safety (carries over from the vector case):
//   The per-element stride is derived from sizeof(E), NEVER assumed to equal
//   Rows*Cols*sizeof(Scalar). A std::vector<E> always spaces elements at
//   sizeof(E) byte intervals (any alignas padding folded into sizeof), so the
//   Tensor describes the real layout exactly — no alignment assumption, no UB,
//   no need for Eigen::aligned_allocator (C++17 std::allocator uses aligned
//   operator new for over-aligned types).
//
//   The ONE caveat (a direction not offered here): do NOT build a fixed-size,
//   vectorizable Eigen::Map<Aligned> (e.g. Map<Vector4d>) over memory ztensor
//   ALLOCATED ITSELF (plain std::malloc, 16-byte aligned). Under AVX such a map
//   may demand 32-byte alignment. Use Eigen::Unaligned or a Dynamic map over
//   ztensor-owned buffers; this header never constructs one.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/Tensor.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/TensorOptions.h"
#include "ztensor/zt/utility/Log.h"

namespace zt {
namespace eigen {

// ---- Aliases (point-cloud / small-matrix friendly) -------------------------
// PascalCase to match zt::Tensor / zt::Scalar. Add more as needed.
using Vec2f = ::Eigen::Vector2f;
using Vec3f = ::Eigen::Vector3f;
using Vec2d = ::Eigen::Vector2d;
using Vec3d = ::Eigen::Vector3d;

using Matrix2f = ::Eigen::Matrix2f;
using Matrix3f = ::Eigen::Matrix3f;
using Matrix4f = ::Eigen::Matrix4f;
using Matrix2d = ::Eigen::Matrix2d;
using Matrix3d = ::Eigen::Matrix3d;
using Matrix4d = ::Eigen::Matrix4d;
using MatrixXf = ::Eigen::MatrixXf;  // dynamic (heap)
using MatrixXd = ::Eigen::MatrixXd;  // dynamic (heap)

// ===========================================================================
// std::vector<E>  ->  Tensor   (E = fixed-size Eigen vector or matrix)
// ===========================================================================
// `E` is any fixed-size Eigen type (VectorN* / MatrixNN*). A column-vector
// element yields a 2-D {N,Rows} Tensor; a matrix element yields a 3-D
// {N,Rows,Cols} Tensor. Strides are in ELEMENTS (ztensor invariant) and derived
// from sizeof(E) and E's storage order, so over-alignment / padding and
// column-major matrices are respected automatically.

template<typename E>
Tensor from_vector(const std::vector<E>& v) {
    static_assert(
        E::RowsAtCompileTime > 0 && E::ColsAtCompileTime > 0,
        "from_vector: fixed-size Eigen type (vector or matrix) expected");
    using Scalar = typename E::Scalar;
    constexpr int64_t kRows = E::RowsAtCompileTime;
    constexpr int64_t kCols = E::ColsAtCompileTime;
    constexpr int64_t kBatchStride =
        static_cast<int64_t>(sizeof(E) / sizeof(Scalar));
    constexpr int64_t kRowStride = E::IsRowMajor ? kCols : 1;
    constexpr int64_t kColStride = E::IsRowMajor ? 1 : kRows;
    const int64_t n = static_cast<int64_t>(v.size());
    // Non-owning view: caller must keep `v` alive for the Tensor's lifetime.
    if constexpr (kCols == 1) {
        return from_blob(const_cast<void*>(static_cast<const void*>(v.data())),
                         {n, kRows},
                         {kBatchStride, 1},
                         zt::dtype<Scalar>());
    } else {
        return from_blob(const_cast<void*>(static_cast<const void*>(v.data())),
                         {n, kRows, kCols},
                         {kBatchStride, kRowStride, kColStride},
                         zt::dtype<Scalar>());
    }
}

// Owning overload: moves the vector into a Blob-held holder. The Tensor then
// owns the memory and frees it (deletes the holder) on final Blob release —
// the "moved std::vector" Blob mode (AGENTS.md invariant #3).
template<typename E>
Tensor from_vector(std::vector<E>&& v) {
    static_assert(
        E::RowsAtCompileTime > 0 && E::ColsAtCompileTime > 0,
        "from_vector: fixed-size Eigen type (vector or matrix) expected");
    using Scalar = typename E::Scalar;
    constexpr int64_t kRows = E::RowsAtCompileTime;
    constexpr int64_t kCols = E::ColsAtCompileTime;
    constexpr int64_t kBatchStride =
        static_cast<int64_t>(sizeof(E) / sizeof(Scalar));
    constexpr int64_t kRowStride = E::IsRowMajor ? kCols : 1;
    constexpr int64_t kColStride = E::IsRowMajor ? 1 : kRows;
    // unique_ptr guards the moved vector against a (contractually possible)
    // from_blob throw; ownership transfers to the Blob deleter on success.
    auto held = std::make_unique<std::vector<E>>(std::move(v));
    const int64_t n = static_cast<int64_t>(held->size());
    void* data = static_cast<void*>(held->data());
    std::function<void(void*)> deleter = [p = held.get()](void*) { delete p; };
    Tensor out;
    if constexpr (kCols == 1) {
        out = from_blob(data,
                        {n, kRows},
                        {kBatchStride, 1},
                        std::move(deleter),
                        zt::dtype<Scalar>());
    } else {
        out = from_blob(data,
                        {n, kRows, kCols},
                        {kBatchStride, kRowStride, kColStride},
                        std::move(deleter),
                        zt::dtype<Scalar>());
    }
    (void)held.release();  // the Blob deleter now owns the vector
    return out;
}

// ===========================================================================
// Tensor  ->  std::vector<E>  (deep copy out)
// ===========================================================================
// dtype/shape are validated; the tensor is contiguized first if needed. Vector
// elements use a memcpy fast path; matrix elements copy each row-major block
// into E via an Eigen::Map (Eigen converts storage order on assignment).

template<typename E>
std::vector<E> to_vector(const Tensor& t) {
    static_assert(
        E::RowsAtCompileTime > 0 && E::ColsAtCompileTime > 0,
        "to_vector: fixed-size Eigen type (vector or matrix) expected");
    using Scalar = typename E::Scalar;
    constexpr int64_t kRows = E::RowsAtCompileTime;
    constexpr int64_t kCols = E::ColsAtCompileTime;

    ZT_CHECK(t.scalar_type() == CppTypeToScalarType<Scalar>::value,
             "to_vector: dtype mismatch (tensor {}, expected {})",
             toString(t.scalar_type()),
             toString(CppTypeToScalarType<Scalar>::value));
    if constexpr (kCols == 1) {
        ZT_CHECK(t.dim() == 2 && t.size(1) == kRows,
                 "to_vector: expected shape {{N,{}}}, got {}-D",
                 kRows,
                 t.dim());
    } else {
        ZT_CHECK(t.dim() == 3 && t.size(1) == kRows && t.size(2) == kCols,
                 "to_vector: expected shape {{N,{},{}}}, got {}-D",
                 kRows,
                 kCols,
                 t.dim());
    }

    const Tensor c = t.is_contiguous() ? t : t.contiguous();
    const int64_t n = c.size(0);
    std::vector<E> out(static_cast<std::size_t>(n));
    const Scalar* src = c.data_ptr<Scalar>();

    if constexpr (kCols == 1) {
        if constexpr (sizeof(E) ==
                      static_cast<std::size_t>(kRows) * sizeof(Scalar)) {
            if (n > 0) {
                // Dense Vec: scalar arrays pack contiguously; copy through
                // Scalar* (trivially copyable) — a memcpy of E would trip
                // -Wnontrivial-memcall though it is bitwise-safe in practice.
                std::memcpy(out.front().data(),
                            src,
                            static_cast<std::size_t>(n) *
                                static_cast<std::size_t>(kRows) *
                                sizeof(Scalar));
            }
        } else {
            // Over-aligned Vec with trailing padding (rare, e.g. AVX Vec2d):
            // copy the active scalars per row, leave the padding bytes
            // untouched.
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t r = 0; r < kRows; ++r) {
                    out[static_cast<std::size_t>(i)](r) = src[i * kRows + r];
                }
            }
        }
    } else {
        // Matrix element: contiguous `c` is row-major; map each block as
        // row-major and assign into E (Eigen converts column/row-major order).
        using RowMajorE =
            ::Eigen::Matrix<Scalar, kRows, kCols, ::Eigen::RowMajor>;
        for (int64_t i = 0; i < n; ++i) {
            ::Eigen::Map<const RowMajorE> view(src + i * kRows * kCols);
            out[static_cast<std::size_t>(i)] = view;
        }
    }
    return out;
}

// ===========================================================================
// Single Eigen matrix  <->  2-D Tensor
// ===========================================================================
// from_matrix: zero-copy strided view of ANY direct-access Eigen expression
// (plain Matrix, Map, Block, Transpose). Respects the source's storage order:
// column-major -> non-contiguous {rows,cols} strides {1,rows}; row-major ->
// contiguous.
//
// LIFETIME: the returned Tensor is a NON-OWNING view — the caller must keep
// the source expression's storage alive for the Tensor's lifetime. A computed
// expression must be eval()'d into a NAMED variable, never a temporary:
//   auto e = (a + b).eval();   from_matrix(e);          // correct
//   from_matrix((a + b).eval());                        // WRONG: dangles
// (a temporary's buffer is freed at the end of the full expression, leaving
// the view's data pointer dangling). Requires direct access — expressions
// without it trip the static_assert below.

template<typename Derived>
Tensor from_matrix(const ::Eigen::MatrixBase<Derived>& m) {
    static_assert(Derived::Flags & ::Eigen::DirectAccessBit,
                  "from_matrix: requires direct-access storage (plain "
                  "Matrix/Map/Block/Transpose). Bind a computed expression to "
                  "a named variable first, e.g. auto e = (a+b).eval(); "
                  "from_matrix(e); -- a temporary would dangle the view.");
    using Scalar = typename Derived::Scalar;
    // .data(), rowStride(), colStride() live on the DirectAccess CRTP leaf,
    // not on MatrixBase — resolve every accessor on the concrete Derived type.
    // Eigen's own rowStride()/colStride() already fold the storage order in
    // (including a Transpose's flipped RowMajorBit — its
    // outerStride/innerStride values are UNCHANGED; only the bit flips, and
    // rowStride/colStride are derived from it), so the view matches the
    // expression exactly with no per-bit reasoning here. Strides are in
    // elements, ztensor's unit.
    const Derived& e = m.derived();
    return from_blob(
        const_cast<void*>(static_cast<const void*>(e.data())),
        {static_cast<int64_t>(e.rows()), static_cast<int64_t>(e.cols())},
        {static_cast<int64_t>(e.rowStride()),
         static_cast<int64_t>(e.colStride())},
        zt::dtype<Scalar>());
}

// to_matrix: deep copy a 2-D Tensor into an Eigen matrix (fixed or dynamic).
// The tensor is contiguized first if needed; the row-major Tensor is then
// mapped and assigned, letting Eigen convert to the target's storage order.
template<typename MatrixType>
MatrixType to_matrix(const Tensor& t) {
    using Scalar = typename MatrixType::Scalar;
    constexpr int64_t kRows = MatrixType::RowsAtCompileTime;
    constexpr int64_t kCols = MatrixType::ColsAtCompileTime;

    ZT_CHECK(t.dim() == 2, "to_matrix: expected 2-D, got {}-D", t.dim());
    ZT_CHECK(t.scalar_type() == CppTypeToScalarType<Scalar>::value,
             "to_matrix: dtype mismatch (tensor {}, expected {})",
             toString(t.scalar_type()),
             toString(CppTypeToScalarType<Scalar>::value));
    if constexpr (kRows != ::Eigen::Dynamic) {
        ZT_CHECK(t.size(0) == kRows,
                 "to_matrix: expected {} rows, got {}",
                 kRows,
                 t.size(0));
    }
    if constexpr (kCols != ::Eigen::Dynamic) {
        ZT_CHECK(t.size(1) == kCols,
                 "to_matrix: expected {} cols, got {}",
                 kCols,
                 t.size(1));
    }

    const Tensor c = t.is_contiguous() ? t : t.contiguous();
    using RowMajorView = ::Eigen::Map<const ::Eigen::Matrix<Scalar,
                                                            ::Eigen::Dynamic,
                                                            ::Eigen::Dynamic,
                                                            ::Eigen::RowMajor>>;
    RowMajorView view(c.data_ptr<Scalar>(), c.size(0), c.size(1));
    return MatrixType(view);
}

}  // namespace eigen
}  // namespace zt
