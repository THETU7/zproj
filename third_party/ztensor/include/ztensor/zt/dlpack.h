// ztensor/zt/dlpack.h
//
// Vendored DLPack v1.2 (dmlc/dlpack). DLPack is the de-facto C ABI for
// zero-copy tensor exchange between frameworks (PyTorch, NumPy, JAX,
// TensorFlow, CuPy, etc.).
//
// This file is a self-contained, unmodified copy of the upstream dlpack.h
// with the minimum definitions needed for ztensor's ToDLPack / FromDLPack.
// Only the v1.x (DLManagedTensorVersioned) API is included; the deprecated
// v0.x (DLManagedTensor) is provided for compatibility.
//
// Original: https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
// License: Apache 2.0

#pragma once

#include <cstdint>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

// ── Version ───────────────────────────────────────────────────────────────

#define DLPACK_VERSION 120

// ── Data type codes ───────────────────────────────────────────────────────

typedef enum {
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
    kDLOpaqueHandle = 3U,
    kDLBfloat = 4U,
    kDLComplex = 5U,
    kDLBool = 6U,
} DLDataTypeCode;

typedef struct {
    uint8_t code;  // DLDataTypeCode
    uint8_t bits;
    uint16_t lanes;  // always 1 for scalar types
} DLDataType;

// ── Device type codes ─────────────────────────────────────────────────────

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
} DLDeviceType;

typedef struct {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

// ── Tensor structure ──────────────────────────────────────────────────────

typedef struct {
    void* data;            // pointer to the backing buffer
    DLDevice device;       // device where data lives
    int32_t ndim;          // number of dimensions
    DLDataType dtype;      // element type
    int64_t* shape;        // shape array (ndim entries), may be nullptr
    int64_t* strides;      // stride array (ndim entries, in ELEMENTS — NOT
                           // bytes; matches upstream dlpack.h). May be
                           // nullptr only for ndim==0 (pre-v1.2 it could be
                           // nullptr to mean row-major contiguous, but
                           // v1.2+ requires it set when ndim != 0).
    uint64_t byte_offset;  // offset from data to the first element
} DLTensor;

// ── Managed tensor (v0.x, deprecated but still common) ───────────────────

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;  // opaque context passed to deleter
    void (*deleter)(struct DLManagedTensor* self);  // called on destruction
} DLManagedTensor;

// ── Managed tensor (v1.x, versioned) ─────────────────────────────────────

// DLPack ABI version, matching upstream dlpack.h (DLPackVersion.major/minor).
// The v1.x struct layout below is ABI-stable through the `flags` field —
// consumers (NumPy/PyTorch/JAX) read the leading fields by offset, so the
// field order and types MUST match the upstream dlpack.h exactly.
typedef struct {
    uint32_t major;
    uint32_t minor;
} DLPackVersion;

typedef struct DLManagedTensorVersioned {
    // ABI version of this struct. Set by the producer; consumers check before
    // casting (the capsule name "dltensor_versioned" implies this struct).
    DLPackVersion version;

    // Opaque producer context passed to `deleter`. May be NULL.
    void* manager_ctx;

    // Destructor: called by the consumer when it is done with the tensor.
    // Deletes `self` as well. May be NULL if the producer cannot provide one.
    void (*deleter)(struct DLManagedTensorVersioned* self);

    // Bitmask flags (reserved, must be 0; see DLPACK_FLAG_BITMASK_* upstream).
    // NOTE: this is uint64_t (not uint32_t) per the upstream spec — a size
    // mismatch here shifts every following field and corrupts `dl_tensor`.
    uint64_t flags;

    // The underlying tensor descriptor. Field order is load-bearing: upstream
    // places dl_tensor LAST, after version/manager_ctx/deleter/flags.
    DLTensor dl_tensor;
} DLManagedTensorVersioned;

#ifdef __cplusplus
}  // extern "C"
#endif
