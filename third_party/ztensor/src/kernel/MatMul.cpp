// ztensor/kernel/MatMul.cpp
//
// Device dispatch shim for MatMul / AddMM (DESIGN §8.5.A). Mirrors the
// BinaryEW.cpp dispatch shape: CPU branch unconditionally, CUDA branch under
// BUILD_CUDA_MODULE, else ZT_LOG_ERROR (which is [[noreturn]]).

#include "kernel/MatMul.h"

#include "ztensor/zt/utility/Log.h"

namespace zt::kernel {

void MatMul(const Tensor& A, const Tensor& B, const Tensor& C) {
    if (A.is_cpu() && B.is_cpu() && C.is_cpu()) {
        MatMulCPU(A, B, C);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (A.is_cuda() && B.is_cuda() && C.is_cuda()) {
        MatMulCUDA(A, B, C);
        return;
    }
#endif
    ZT_LOG_ERROR("MatMul: unsupported device (A {}, B {}, C {})",
                 A.device().string(),
                 B.device().string(),
                 C.device().string());
}

void AddMM(const Tensor& C,
           const Tensor& A,
           const Tensor& B,
           const Tensor& out,
           double beta,
           double alpha) {
    if (C.is_cpu() && A.is_cpu() && B.is_cpu() && out.is_cpu()) {
        AddMMCPU(C, A, B, out, beta, alpha);
        return;
    }
#ifdef BUILD_CUDA_MODULE
    if (C.is_cuda() && A.is_cuda() && B.is_cuda() && out.is_cuda()) {
        AddMMCUDA(C, A, B, out, beta, alpha);
        return;
    }
#endif
    ZT_LOG_ERROR("AddMM: unsupported device (C {}, A {}, B {}, out {})",
                 C.device().string(),
                 A.device().string(),
                 B.device().string(),
                 out.device().string());
}

}  // namespace zt::kernel
