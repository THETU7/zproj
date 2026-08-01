// ztensor/kernel/MatMulCUDA.cu
//
// CUDA matrix multiplication (DESIGN §8.5.A), backed by cuBLAS.
//
// Row-major <-> column-major trick: ztensor is row-major; cuBLAS is
// column-major. For C_row = A_row @ B_row, transpose both sides:
//   C_row^T = (A_row @ B_row)^T = B_row^T @ A_row^T
// Reading B_row^T as a column-major matrix gives back B_row's memory layout,
// so we call Gemm with (A=B_row, B=A_row, no-transpose both), writing into
// C_row's memory interpreted column-major = C_row^T. Result: the row-major
// C holds A_row @ B_row with no extra copies.
//
// * Per-device singleton cublasHandle_t, stream-bound to cuda::GetStream() on
//   every call (cheap no-op when already bound).
// * Inputs contiguized first via Tensor::contiguous().
// * Half: native CUDA_R_16F inputs with CUBLAS_COMPUTE_32F_FAST_16F (float
//   accumulate).
// * BFloat16: native CUDA_R_16BF inputs with CUBLAS_COMPUTE_32F_FAST_16BF
//   (float accumulate, CUDA 11.8+).
// * Integer matmul rejected with ZT_LOG_ERROR (DESIGN out of scope).
//
// Two batch counts are exercised by the dispatcher:
//   * MatMulCUDA: 2D GEMM (batch=1 via cublasGemmEx_64) or true 3D batched
//     (via cublasGemmStridedBatchedEx_64).
//   * AddMMCUDA:  2D GEMM only, with beta applied to C and alpha to A@B.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <unordered_map>

#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/utility/Log.h"

#include "ztensor/zt/cuda/Guard.h"
#include "ztensor/zt/cuda/Stream.h"
#include "kernel/MatMul.h"

namespace zt {
namespace kernel {
namespace {

// Check a cublasStatus_t and throw on failure with the context.
void check_cublas(cublasStatus_t st,
                  const char* call,
                  const char* file,
                  int line) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        ZT_LOG_ERROR("cuBLAS error {} from {} at {}:{}",
                     static_cast<int>(st),
                     call,
                     file,
                     line);
    }
}
#define ZT_CUBLAS_CHECK(call) check_cublas((call), #call, __FILE__, __LINE__)

// Check the dtype is supported by the CUDA matmul (Float/Double/Half/BFloat16).
void check_supported_dtype(ScalarType dt) {
    if (dt == ScalarType::Float || dt == ScalarType::Double ||
        dt == ScalarType::Half || dt == ScalarType::BFloat16) {
        return;
    }
    ZT_LOG_ERROR("MatMulCUDA: unsupported dtype {} (only Float/Double/Half/"
                 "BFloat16)",
                 toString(dt));
}

// Translate a ztensor dtype to the cuBLAS data-type and compute-type pair.
cudaDataType_t cuda_dtype(ScalarType dt) {
    switch (dt) {
        case ScalarType::Float:
            return CUDA_R_32F;
        case ScalarType::Double:
            return CUDA_R_64F;
        case ScalarType::Half:
            return CUDA_R_16F;
        case ScalarType::BFloat16:
            return CUDA_R_16BF;
        default:
            ZT_LOG_ERROR("MatMulCUDA: no cudaDataType for {}", toString(dt));
    }
}
cublasComputeType_t compute_type(ScalarType dt) {
    switch (dt) {
        case ScalarType::Float:
            return CUBLAS_COMPUTE_32F;
        case ScalarType::Double:
            return CUBLAS_COMPUTE_64F;
        // Half inputs, float-accumulate (DESIGN's "Half default float
        // accumulate"). cuBLAS still takes the alpha/beta scalars in float
        // for this compute type, matching make_scalar() below.
        case ScalarType::Half:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        // BFloat16 inputs, float-accumulate (CUBLAS_COMPUTE_32F_FAST_16BF,
        // CUDA 11.8+). Alpha/beta stay float, same as the Half path.
        case ScalarType::BFloat16:
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        default:
            ZT_LOG_ERROR("MatMulCUDA: no computeType for {}", toString(dt));
    }
}

// Per-device cuBLAS handle cache. Handles are bound to the active stream on
// every call; stream 0 is currently the only stream (cuda::GetStream).
struct HandleEntry {
    cublasHandle_t handle = nullptr;
};
std::unordered_map<int, HandleEntry>& handle_cache() {
    // Meyers singleton: thread-unsafe init is fine because cuBLAS calls are
    // serialized on the (default) stream by ztensor's single-stream model.
    static std::unordered_map<int, HandleEntry> cache;
    return cache;
}

// Get (or lazily create) the cuBLAS handle for the device backing `t`, and
// bind it to the current stream. The handle lives for the program lifetime.
cublasHandle_t get_cublas_handle(const Tensor& t) {
    CUDAScopedDevice scoped(t.device());
    const int dev = cuda::GetDevice();
    auto& entry = handle_cache()[dev];
    if (entry.handle == nullptr) {
        ZT_CUBLAS_CHECK(cublasCreate_v2(&entry.handle));
    }
    ZT_CUBLAS_CHECK(cublasSetStream_v2(entry.handle, cuda::GetStream()));
    return entry.handle;
}

// Typed alpha/beta scalars. cuBLAS takes them as `const void*` interpreted in
// the compute type. Use a small host buffer + a tag for float vs double.
struct ScalarBuf {
    float f = 0;
    double d = 0;
    bool is_double = false;
    const void* ptr() const {
        return is_double ? static_cast<const void*>(&d)
                         : static_cast<const void*>(&f);
    }
};
ScalarBuf make_scalar(double v, ScalarType dt) {
    ScalarBuf s;
    if (dt == ScalarType::Double) {
        s.is_double = true;
        s.d = v;
    } else {
        s.f = static_cast<float>(v);
    }
    return s;
}

// 2D GEMM. A {m,k}, B {k,n}, C {m,n}, all contiguous. Implements the
// row-major trick described at the top of the file.
void gemm_2d(const Tensor& A,
             const Tensor& B,
             const Tensor& C,
             double alpha,
             double beta,
             const Tensor* C_in = nullptr) {
    // When beta != 0 and a separate C_in is provided (AddMM), C already holds
    // a copy of beta*C_in (caller's responsibility) — so for AddMM we pass
    // beta through directly with C_in == C. For plain MatMul beta == 0 and
    // C_in is unused.
    (void)C_in;
    cublasHandle_t handle = get_cublas_handle(C);
    const ScalarType dt = C.scalar_type();
    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(1);
    const ScalarBuf a_buf = make_scalar(alpha, dt);
    const ScalarBuf b_buf = make_scalar(beta, dt);
    // Row-major trick: pass (B, A) reversed, no-transpose both. Leading dims
    // are the *other* matrix's contraction dim (n for B-as-first, k for A).
    void* C_ptr = const_cast<void*>(C.data_ptr());
    const void* A_ptr = A.data_ptr();
    const void* B_ptr = B.data_ptr();
    ZT_CUBLAS_CHECK(cublasGemmEx_64(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,  // (n, m, k) — swapped order
                                    a_buf.ptr(),
                                    B_ptr,
                                    cuda_dtype(dt),
                                    n,  // first matrix: B, ldb = n
                                    A_ptr,
                                    cuda_dtype(dt),
                                    k,  // second matrix: A, lda = k
                                    b_buf.ptr(),
                                    C_ptr,
                                    cuda_dtype(dt),
                                    n,  // C, ldc = n
                                    compute_type(dt),
                                    CUBLAS_GEMM_DEFAULT));
}

// Batched GEMM. A {b,m,k}, B {b,k,n}, C {b,m,n}, all contiguous. Strides are
// in *elements* (cuBLAS wants element counts, matching ztensor's element-unit
// strides).
void gemm_batched(const Tensor& A,
                  const Tensor& B,
                  const Tensor& C,
                  double alpha,
                  double beta) {
    cublasHandle_t handle = get_cublas_handle(C);
    const ScalarType dt = C.scalar_type();
    const int64_t batch = A.size(0);
    const int64_t m = A.size(1);
    const int64_t k = A.size(2);
    const int64_t n = B.size(2);
    const ScalarBuf a_buf = make_scalar(alpha, dt);
    const ScalarBuf b_buf = make_scalar(beta, dt);
    const long long stride_a = static_cast<long long>(m * k);
    const long long stride_b = static_cast<long long>(k * n);
    const long long stride_c = static_cast<long long>(m * n);
    void* C_ptr = const_cast<void*>(C.data_ptr());
    const void* A_ptr = A.data_ptr();
    const void* B_ptr = B.data_ptr();
    ZT_CUBLAS_CHECK(
        cublasGemmStridedBatchedEx_64(handle,
                                      CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      n,
                                      m,
                                      k,  // (n, m, k) — swapped order
                                      a_buf.ptr(),
                                      B_ptr,
                                      cuda_dtype(dt),
                                      n,
                                      stride_b,  // first matrix: B
                                      A_ptr,
                                      cuda_dtype(dt),
                                      k,
                                      stride_a,  // second matrix: A
                                      b_buf.ptr(),
                                      C_ptr,
                                      cuda_dtype(dt),
                                      n,
                                      stride_c,
                                      static_cast<int64_t>(batch),
                                      compute_type(dt),
                                      CUBLAS_GEMM_DEFAULT));
}

}  // namespace

void MatMulCUDA(const Tensor& A, const Tensor& B, const Tensor& C) {
    check_supported_dtype(C.scalar_type());
    CUDAScopedDevice scoped(C.device());
    // Contiguize inputs (cuBLAS needs packed row-major memory).
    const Tensor Ac = A.is_contiguous() ? A : A.contiguous();
    const Tensor Bc = B.is_contiguous() ? B : B.contiguous();
    if (Ac.dim() == 2) {
        gemm_2d(Ac, Bc, C, /*alpha=*/1.0, /*beta=*/0.0);
    } else if (Ac.dim() == 3) {
        gemm_batched(Ac, Bc, C, /*alpha=*/1.0, /*beta=*/0.0);
    } else {
        ZT_LOG_ERROR("MatMulCUDA: unsupported ndim {} (2 or 3 expected)",
                     Ac.dim());
    }
}

void AddMMCUDA(const Tensor& C,
               const Tensor& A,
               const Tensor& B,
               const Tensor& out,
               double beta,
               double alpha) {
    check_supported_dtype(out.scalar_type());
    CUDAScopedDevice scoped(out.device());
    const Tensor Ac = A.is_contiguous() ? A : A.contiguous();
    const Tensor Bc = B.is_contiguous() ? B : B.contiguous();
    const Tensor Cc = C.is_contiguous() ? C : C.contiguous();
    // cuBLAS GEMM with beta != 0 reads C into the output: copy C into out
    // first (cast if needed), then GEMM accumulates in place. `out` is the
    // logical output of this op (mutable) even though the kernel signature
    // takes it by const& for quartet consistency — cast away const, matching
    // how the Indexer path writes through a const Tensor& dst.
    Tensor& out_mut = const_cast<Tensor&>(out);
    out_mut.copy_(Cc);
    gemm_2d(Ac, Bc, out_mut, alpha, beta);
}

#undef ZT_CUBLAS_CHECK

}  // namespace kernel
}  // namespace zt
