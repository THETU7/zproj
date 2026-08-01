// ztensor/kernel/MatMul.h
//
// Matrix multiplication kernels (DESIGN §8.5.A). Unlike the element-wise
// quartet, MatMul does NOT use ParallelFor/Indexer — it delegates to a BLAS
// library per backend (Eigen on CPU, cuBLAS on CUDA).
//
// Two entry points:
//   * MatMul(A, B, C)        -> C = A @ B          (batched GEMM)
//   * AddMM(C, A, B, out, beta, alpha)
//                            -> out = beta*C + alpha*(A@B)   (2D GEMM)
//
// Shape contract (validated by the caller in Tensor.cpp, NOT here):
//   * MatMul : A {b,m,k}, B {b,k,n} -> C {b,m,n}. The leading `b` dim may be
//     empty (plain 2D GEMM). The three tensors share their leading batch
//     shape. Inputs may be non-contiguous; the kernels contiguize as needed.
//   * AddMM  : C/A/B/out are all 2D {m,n}/{m,k}/{k,n}/{m,n}.
//
// Dtype contract: Float / Double / Half / BFloat16 are supported. Half /
// BFloat16: CPU promotes to float for the GEMM and casts back; CUDA uses
// native cuBLAS low-precision compute (CUBLAS_COMPUTE_32F_FAST_16F /
// CUBLAS_COMPUTE_32F_FAST_16BF). Integer matmul is rejected (cuBLAS int path
// out of scope).

#pragma once

#include "ztensor/zt/Scalar.h"
#include "ztensor/zt/Tensor.h"

namespace zt::kernel {

/// `C = A @ B`. `A` and `B` share a leading batch shape; the trailing two
/// dims are the {m,k}/{k,n}/{m,n} matrix dims. `C` is pre-allocated by the
/// caller with the right batched {b,m,n} shape and dtype.
void MatMul(const Tensor& A, const Tensor& B, const Tensor& C);

/// `out = beta*C + alpha*(A@B)`, 2D only. `out` is pre-allocated by the caller
/// with C's shape and dtype. `alpha`/`beta` are scalars in float64; the kernel
/// casts them to the working precision.
void AddMM(const Tensor& C,
           const Tensor& A,
           const Tensor& B,
           const Tensor& out,
           double beta,
           double alpha);

// ── CPU implementations ────────────────────────────────────────────────────
void MatMulCPU(const Tensor& A, const Tensor& B, const Tensor& C);
void AddMMCPU(const Tensor& C,
              const Tensor& A,
              const Tensor& B,
              const Tensor& out,
              double beta,
              double alpha);

// ── CUDA implementations (only declared under BUILD_CUDA_MODULE) ───────────
#ifdef BUILD_CUDA_MODULE
void MatMulCUDA(const Tensor& A, const Tensor& B, const Tensor& C);
void AddMMCUDA(const Tensor& C,
               const Tensor& A,
               const Tensor& B,
               const Tensor& out,
               double beta,
               double alpha);
#endif

}  // namespace zt::kernel
