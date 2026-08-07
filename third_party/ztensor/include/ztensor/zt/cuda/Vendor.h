// ztensor/zt/cuda/Vendor.h
//
// Single GPU-vendor selection header. Every translation unit that today
// `#include`s <cuda_runtime.h> (and the one that includes <cublas_v2.h>)
// instead includes this header. It then resolves to the right vendor's
// runtime + BLAS headers and, for the HIP/ROCm backend, applies the
// `cuda*`/`cublas*` -> `hip*`/`hipblas*` symbol remap that lets the existing
// CUDA sources compile unchanged as HIP.
//
// This mirrors ggml's vendors/{cuda.h,hip.h} + common.cuh selection (no
// hipify tool: the same .cu files are compiled as HIP via enable_language(HIP),
// and a macro shim translates the NVIDIA-spelled symbols). ztensor's GPU
// surface is far smaller than ggml's (one <<<>>> launch site, no warp
// intrinsics, no shared memory, no driver/VMM/graph API, no PTX), so the shim
// below only remaps the symbols ztensor actually uses.
//
// Build modes (mutually exclusive, set by CMake):
//   * BUILD_CUDA_MODULE — NVIDIA CUDA. Includes the CUDA headers verbatim;
//     no remapping. Identical to the pre-HIP behavior, just centralized.
//   * BUILD_HIP_MODULE  — AMD HIP/ROCm. CMake also defines BUILD_CUDA_MODULE
//     so the existing #ifdef BUILD_CUDA_MODULE dispatch code compiles; this
//     header's HIP branch is what selects the hip runtime + the remap.
//
// Extension point: a future MUSA backend adds an `#elif
// defined(BUILD_MUSA_MODULE)` branch here (cf. ggml vendors/musa.h). Not
// implemented in this phase.

#pragma once

#if defined(BUILD_HIP_MODULE)
// ============================================================================
// AMD HIP / ROCm
// ============================================================================
// Suppress the deprecated masked warp-sync builtins; HIP's shuffles take no
// mask. (ztensor uses no warp builtins today, but define it for parity with
// ggml and to silence any transitive header use.)
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1

// Runtime + BLAS only. The fp16/bf16 *device-type* headers (<hip/hip_fp16.h>,
// <hip/hip_bf16.h>) are deliberately NOT included here: hip_bf16.h requires the
// __bf16 compiler builtin, which only hip-clang (defining __HIPCC__) provides —
// plain host g++ lacks it, and these macros are expanded in host .cpp TUs. The
// device half/bf16 interop is pulled in by Half.h / BFloat16.h, gated on
// __HIPCC__, so it only reaches the kernel .cu TUs (compiled by hip-clang).
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#if !defined(__HIP_PLATFORM_AMD__)
#error "ztensor HIP backend supports only AMD targets (__HIP_PLATFORM_AMD__)"
#endif

// CUDART_VERSION is CUDA's runtime-version macro, used in CudaMemoryPool.cpp
// (gates the cudaMallocAsync pool path: #if CUDART_VERSION >= 11020) and logged
// by the pool. HIP has no such macro, so synthesize one. ROCm 7.x supports
// hipMallocAsync, so the async path must always be enabled; pick a CUDA-12-era
// value (>= 11020 AND >= 12000) so every existing guard lights up correctly.
// (MatMulCUDA.cu's #elif CUDART_VERSION >= 12000 is already short-circuited by
// its #if defined(__HIPCC__) branch, but this keeps the value consistent.)
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

// ----------------------------------------------------------------------------
// Runtime API: cuda* -> hip*
// ----------------------------------------------------------------------------
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemPool_t hipMemPool_t
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReservedMemCurrent hipMemPoolAttrReservedMemCurrent
#define cudaMemPoolAttrUsedMemCurrent hipMemPoolAttrUsedMemCurrent
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolTrimTo hipMemPoolTrimTo
#define cudaMemset hipMemset
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

// ----------------------------------------------------------------------------
// cuBLAS -> hipBLAS
// ----------------------------------------------------------------------------
#define cublasCreate_v2 hipblasCreate
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetStream_v2 hipblasSetStream
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUDA_R_16BF HIPBLAS_R_16B
#define CUDA_R_16F HIPBLAS_R_16F
#define CUDA_R_32F HIPBLAS_R_32F
#define CUDA_R_64F HIPBLAS_R_64F

// Compute-type enums. hipBLAS gained the HIPBLAS_COMPUTE_* /
// hipblasComputeType_t surface in HIP 6.5. ROCm 7.x has it; the guard keeps
// older toolkits on the R_* fallback (cf. ggml vendors/hip.h).
#if defined(HIP_VERSION) && (HIP_VERSION >= 60500000)
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
// cuBLAS's _FAST_16F / _FAST_16BF compute types request the fast (tensor-core)
// float-accumulate path for fp16/bf16 inputs. hipBLAS *defines* the matching
// HIPBLAS_COMPUTE_32F_FAST_16F (=4) / _FAST_16BF (=5) enums (in
// hipblas-common.h), but they are absent from hipBLAS Gemm's supported
// compute-type table (hipblas.h lists only COMPUTE_16F/32F/64F/32I), so the
// call returns HIPBLAS_STATUS_NOT_SUPPORTED for fp16/bf16 inputs (verified:
// MMHalf / MMBFloat16 fail on gfx1103). Map both to plain HIPBLAS_COMPUTE_32F
// — fp16/bf16 inputs with float32 accumulate, numerically identical to the
// intent; HIPBLAS_GEMM_DEFAULT still selects the fastest kernel on the device.
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16BF HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_64F HIPBLAS_COMPUTE_64F
#define cublasComputeType_t hipblasComputeType_t
#define cudaDataType_t hipDataType
#else
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16BF HIPBLAS_R_32F
#define CUBLAS_COMPUTE_64F HIPBLAS_R_64F
#define cublasComputeType_t hipblasDatatype_t
#define cudaDataType_t hipblasDatatype_t
#endif

#elif defined(BUILD_CUDA_MODULE)
// ============================================================================
// NVIDIA CUDA (centralized includes; no symbol remapping)
// ============================================================================
#include <cublas_v2.h>
#include <cuda_runtime.h>

#else
#error \
    "ztensor/zt/cuda/Vendor.h included without BUILD_CUDA_MODULE or BUILD_HIP_MODULE"
#endif
