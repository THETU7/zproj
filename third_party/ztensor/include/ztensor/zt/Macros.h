// ztensor/zt/Macros.h
//
// Common compiler annotations shared across public and internal headers
// (formerly defined in the internal CUDAUtils.h; now public so no header
// needs to pull in private implementation details).

#pragma once

// Mark a function/lambda callable from both host and device. Under NVCC
// this expands to __host__ __device__; otherwise it is a no-op.
#ifdef __CUDACC__
#define ZT_HOST_DEVICE __host__ __device__
#else
#define ZT_HOST_DEVICE
#endif
