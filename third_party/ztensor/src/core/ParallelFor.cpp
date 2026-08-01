// ztensor/core/ParallelFor.cpp
//
// CPU helpers for ParallelFor. The OpenMP pragmas themselves live in the
// ParallelFor template (ParallelFor.h); this translation unit only provides the
// thread-count / nested-region queries that the pragmas and the reduction
// engine consult.

#include "core/ParallelFor.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace zt::core {

int EstimateMaxThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

bool InParallel() {
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

}  // namespace zt::core
