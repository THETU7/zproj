// ztensor/kernel/MatMulCPU.cpp
//
// CPU matrix multiplication (DESIGN §8.5.A), backed by Eigen.
//
// * Inputs are mapped as Eigen::Map<Matrix<T,Dynamic,Dynamic,RowMajor>> over
//   contiguous memory — ztensor is row-major, so RowMajor avoids a transpose.
// * Non-contiguous inputs are contiguized first via Tensor::contiguous().
// * Batched matmul: loop over the (broadcasted, flattened) leading batch dim,
//   one GEMM per batch slice.
// * Half/BFloat16: Eigen has no low-precision GEMM, so Half and BFloat16 inputs
//   are promoted to float for the GEMM and cast back (matches DESIGN's "float
//   accumulate" note; CUDA Half/BFloat16 is native). Integer matmul is rejected
//   with ZT_LOG_ERROR.

#include <cstdint>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "ztensor/zt/BFloat16.h"
#include "ztensor/zt/Half.h"
#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/utility/Log.h"

#include "kernel/MatMul.h"

namespace zt {
namespace kernel {

// Narrow dtype dispatch: Float / Double / Half / BFloat16. Integer matmul is
// out of scope (DESIGN §8.5.A) and hits the trailing ZT_LOG_ERROR.
#define ZT_MATMUL_DISPATCH(DTYPE, ...)               \
    [&] {                                            \
        if (DTYPE == ::zt::ScalarType::Float) {      \
            using scalar_t = float;                  \
            return __VA_ARGS__();                    \
        }                                            \
        if (DTYPE == ::zt::ScalarType::Double) {     \
            using scalar_t = double;                 \
            return __VA_ARGS__();                    \
        }                                            \
        if (DTYPE == ::zt::ScalarType::Half) {       \
            using scalar_t = ::zt::Half;             \
            return __VA_ARGS__();                    \
        }                                            \
        if (DTYPE == ::zt::ScalarType::BFloat16) {   \
            using scalar_t = ::zt::BFloat16;         \
            return __VA_ARGS__();                    \
        }                                            \
        ZT_LOG_ERROR(                                \
            "MatMulCPU: unsupported dtype {} (only " \
            "Float/Double/Half/BFloat16)",           \
            ::zt::toString(DTYPE));                  \
    }()

namespace {

// The compute precision: float for Half inputs (Eigen has no half GEMM),
// the dtype itself otherwise.
template<typename scalar_t>
struct compute_type {
    using type = scalar_t;
};
template<>
struct compute_type<Half> {
    using type = float;
};
template<>
struct compute_type<BFloat16> {
    using type = float;
};

// Get a writable typed pointer out of a `const Tensor&`. Matches the pattern
// used by Indexer (const_cast<void*>(t.data_ptr())); the dst tensor is
// logically mutable (it is the op output) even though the kernel signatures
// take it by const& for consistency with the element-wise quartet.
template<typename T>
T* mut_ptr(const Tensor& t) {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}
template<typename T>
const T* c_ptr(const Tensor& t) {
    return static_cast<const T*>(t.data_ptr());
}

// Contiguous 2D GEMM, working in compute_t.
template<typename compute_t>
void gemm_2d(const compute_t* A,
             const compute_t* B,
             compute_t* C,
             int64_t m,
             int64_t k,
             int64_t n) {
    using Eigen::Dynamic;
    using Eigen::RowMajor;
    using MatrixRM =
        Eigen::Map<const Eigen::Matrix<compute_t, Dynamic, Dynamic, RowMajor>>;
    using OutMatrixRM =
        Eigen::Map<Eigen::Matrix<compute_t, Dynamic, Dynamic, RowMajor>>;
    MatrixRM a(A, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
    MatrixRM b(B, static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
    OutMatrixRM c(
        C, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
    c.noalias() = a * b;
}

// 2D GEMM, dtype-dispatched. A, B, C must already be contiguous 2D.
// Half / BFloat16: read into float, GEMM, write back.
template<typename scalar_t>
void matmul_2d_typed(const Tensor& A, const Tensor& B, const Tensor& C) {
    using compute_t = typename compute_type<scalar_t>::type;
    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(1);

    if constexpr (!std::is_same_v<scalar_t, compute_t>) {
        // Narrow type (Half, BFloat16): promote to compute_t, GEMM, cast back.
        std::vector<compute_t> af(static_cast<std::size_t>(m * k));
        std::vector<compute_t> bf(static_cast<std::size_t>(k * n));
        std::vector<compute_t> cf(static_cast<std::size_t>(m * n));
        const auto* ap = c_ptr<scalar_t>(A);
        const auto* bp = c_ptr<scalar_t>(B);
        for (int64_t i = 0; i < m * k; ++i)
            af[static_cast<std::size_t>(i)] = static_cast<compute_t>(ap[i]);
        for (int64_t i = 0; i < k * n; ++i)
            bf[static_cast<std::size_t>(i)] = static_cast<compute_t>(bp[i]);
        gemm_2d<compute_t>(af.data(), bf.data(), cf.data(), m, k, n);
        auto* cp = mut_ptr<scalar_t>(C);
        for (int64_t i = 0; i < m * n; ++i)
            cp[i] = static_cast<scalar_t>(cf[static_cast<std::size_t>(i)]);
    } else {
        gemm_2d<compute_t>(c_ptr<scalar_t>(A),
                           c_ptr<scalar_t>(B),
                           mut_ptr<scalar_t>(C),
                           m,
                           k,
                           n);
    }
}

void matmul_2d(const Tensor& A, const Tensor& B, const Tensor& C) {
    ZT_MATMUL_DISPATCH(C.scalar_type(),
                       [&] { matmul_2d_typed<scalar_t>(A, B, C); });
}

// Batched GEMM. A, B, C are 3D contiguous {b,m,k}/{b,k,n}/{b,m,n}. Loops over
// the batch dim, one 2D GEMM each.
template<typename scalar_t>
void matmul_3d_typed(const Tensor& A, const Tensor& B, const Tensor& C) {
    using compute_t = typename compute_type<scalar_t>::type;
    using Eigen::Dynamic;
    using Eigen::RowMajor;
    using MatrixRM =
        Eigen::Map<const Eigen::Matrix<compute_t, Dynamic, Dynamic, RowMajor>>;
    using OutMatrixRM =
        Eigen::Map<Eigen::Matrix<compute_t, Dynamic, Dynamic, RowMajor>>;

    const int64_t batch = A.size(0);
    const int64_t m = A.size(1);
    const int64_t k = A.size(2);
    const int64_t n = B.size(2);
    const Eigen::Index em = static_cast<Eigen::Index>(m);
    const Eigen::Index ek = static_cast<Eigen::Index>(k);
    const Eigen::Index en = static_cast<Eigen::Index>(n);

    if constexpr (!std::is_same_v<scalar_t, compute_t>) {
        // Narrow type (Half, BFloat16): promote to compute_t per batch slice.
        std::vector<compute_t> af(static_cast<std::size_t>(m * k));
        std::vector<compute_t> bf(static_cast<std::size_t>(k * n));
        std::vector<compute_t> cf(static_cast<std::size_t>(m * n));
        for (int64_t b = 0; b < batch; ++b) {
            const auto* ap = c_ptr<scalar_t>(A) + b * m * k;
            const auto* bp = c_ptr<scalar_t>(B) + b * k * n;
            for (Eigen::Index i = 0; i < m * k; ++i) af[i] = ap[i];
            for (Eigen::Index i = 0; i < k * n; ++i) bf[i] = bp[i];
            MatrixRM a(af.data(), em, ek);
            MatrixRM b_(bf.data(), ek, en);
            OutMatrixRM c(cf.data(), em, en);
            c.noalias() = a * b_;
            auto* cp = mut_ptr<scalar_t>(C) + b * m * n;
            for (Eigen::Index i = 0; i < m * n; ++i)
                cp[i] = static_cast<scalar_t>(cf[i]);
        }
    } else {
        const auto stride_a = m * k;
        const auto stride_b = k * n;
        const auto stride_c = m * n;
        for (int64_t b = 0; b < batch; ++b) {
            MatrixRM a(c_ptr<scalar_t>(A) + b * stride_a, em, ek);
            MatrixRM b_(c_ptr<scalar_t>(B) + b * stride_b, ek, en);
            OutMatrixRM c(mut_ptr<scalar_t>(C) + b * stride_c, em, en);
            c.noalias() = a * b_;
        }
    }
}

void matmul_3d(const Tensor& A, const Tensor& B, const Tensor& C) {
    ZT_MATMUL_DISPATCH(C.scalar_type(),
                       [&] { matmul_3d_typed<scalar_t>(A, B, C); });
}

}  // namespace

void MatMulCPU(const Tensor& A, const Tensor& B, const Tensor& C) {
    // Contiguize inputs (GEMM needs packed row-major memory).
    const Tensor Ac = A.is_contiguous() ? A : A.contiguous();
    const Tensor Bc = B.is_contiguous() ? B : B.contiguous();
    if (Ac.dim() == 2) {
        matmul_2d(Ac, Bc, C);
    } else if (Ac.dim() == 3) {
        matmul_3d(Ac, Bc, C);
    } else {
        ZT_LOG_ERROR("MatMulCPU: unsupported ndim {} (2 or 3 expected)",
                     Ac.dim());
    }
}

void AddMMCPU(const Tensor& C,
              const Tensor& A,
              const Tensor& B,
              const Tensor& out,
              double beta,
              double alpha) {
    // out = A@B first, then scale + accumulate.
    kernel::MatMulCPU(A, B, out);

    const int64_t mn = out.size(0) * out.size(1);
    ZT_MATMUL_DISPATCH(out.scalar_type(), [&] {
        using compute_t = typename compute_type<scalar_t>::type;
        const auto a_cast = static_cast<compute_t>(alpha);
        const auto b_cast = static_cast<compute_t>(beta);
        auto* out_p = mut_ptr<scalar_t>(out);
        const auto* c_p = c_ptr<scalar_t>(C);
        if constexpr (!std::is_same_v<scalar_t, compute_t>) {
            // Narrow type (Half, BFloat16): promote to compute_t, scale+accum,
            // cast back.
            for (int64_t i = 0; i < mn; ++i)
                out_p[i] = static_cast<scalar_t>(
                    b_cast * static_cast<compute_t>(c_p[i]) +
                    a_cast * static_cast<compute_t>(out_p[i]));
        } else {
            for (int64_t i = 0; i < mn; ++i)
                out_p[i] =
                    static_cast<scalar_t>(b_cast * c_p[i] + a_cast * out_p[i]);
        }
    });
}

#undef ZT_MATMUL_DISPATCH

}  // namespace kernel
}  // namespace zt
