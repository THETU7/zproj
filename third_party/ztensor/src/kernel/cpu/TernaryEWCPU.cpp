// ztensor/kernel/TernaryEWCPU.cpp
//
// CPU TernaryEW kernel.  cond is Bool, a and b share dtype (promoted by the
// caller), output dtype = dtype of a/b.
//
// The Indexer gets 3 inputs; the caller ensures cond/a/b/dst are broadcast
// to the same shape.  DtypePolicy::ALL_SAME is used because the caller
// already promoted a/b to a common dtype; cond is separately dispatched.

#include "ztensor/zt/utility/Log.h"

#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "kernel/TernaryEW.h"

namespace zt::kernel {

void TernaryEWCPU(const Tensor& cond,
                  const Tensor& a,
                  const Tensor& b,
                  const Tensor& dst,
                  TernaryEWOpCode op) {
    // cond must be Bool.
    if (cond.scalar_type() != zt::ScalarType::Bool) {
        ZT_LOG_ERROR("TernaryEWCPU: cond must be Bool, got {}",
                     zt::toString(cond.scalar_type()));
    }

    switch (op) {
        case TernaryEWOpCode::Where:
            break;
    }

    ZT_DISPATCH_SCALARTYPE_TO_TEMPLATE(a.scalar_type(), [&] {
        core::Indexer indexer({cond, a, b}, dst, core::DtypePolicy::NONE);
        core::ParallelFor(
            dst.device(), indexer.NumWorkloads(), [&](int64_t i) {
                const bool c = *indexer.GetInputPtr<bool>(0, i);
                const scalar_t va = *indexer.GetInputPtr<scalar_t>(1, i);
                const scalar_t vb = *indexer.GetInputPtr<scalar_t>(2, i);
                *indexer.GetOutputPtr<scalar_t>(i) = c ? va : vb;
            });
    });
}

}  // namespace zt::kernel
