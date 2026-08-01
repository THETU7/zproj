// ztensor/kernel/TernaryEWCUDA.cu
//
// CUDA TernaryEW kernel.  Mirrors TernaryEWCPU.cpp: cond is Bool,
// a/b share dtype (promoted by caller), |cond| via a separate cast.

#include <cuda_runtime.h>

#include "ztensor/zt/utility/Log.h"

#include "ztensor/zt/cuda/Guard.h"
#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/TernaryEW.h"

namespace zt {
namespace kernel {

void TernaryEWCUDA(const Tensor& cond,
                   const Tensor& a,
                   const Tensor& b,
                   const Tensor& dst,
                   TernaryEWOpCode op) {
    (void)op;  // only one op code (Where) currently
    if (cond.scalar_type() != zt::ScalarType::Bool) {
        ZT_LOG_ERROR("TernaryEWCUDA: cond must be Bool, got {}",
                     zt::toString(cond.scalar_type()));
    }

    CUDAScopedDevice scoped(dst.device());
    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(a.scalar_type(), [&] {
        core::Indexer indexer({cond, a, b}, dst, core::DtypePolicy::NONE);
        core::ParallelFor(
            dst.device(), indexer.NumWorkloads(), [=] __device__(int64_t i) {
                const bool c = *indexer.GetInputPtr<bool>(0, i);
                const scalar_t va = *indexer.GetInputPtr<scalar_t>(1, i);
                const scalar_t vb = *indexer.GetInputPtr<scalar_t>(2, i);
                *indexer.GetOutputPtr<scalar_t>(i) = c ? va : vb;
            });
    });
}

}  // namespace kernel
}  // namespace zt
