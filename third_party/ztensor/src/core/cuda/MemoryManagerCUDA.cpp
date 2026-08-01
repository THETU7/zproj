// ztensor/core/MemoryManagerCUDA.cu
//
// CUDA backend for zt::MemoryManager. Provides the CUDA_Malloc / CUDA_Free /
// CUDA_Memcpy{H2D,D2H,D2D} functions declared (forward) in MemoryManager.cpp
// and dispatched from the MemoryManager facade when BUILD_CUDA_MODULE is on.
//
// CUDA_Malloc / CUDA_Free are thin shims over CudaMemoryPool (§8.5.E E-3c) —
// the four-tier caching allocator (slab → bucketed → cudaMallocAsync →
// cudaMalloc).  The Memcpy functions are unchanged (they are not pooled).
//
// Compiled only under BUILD_CUDA_MODULE.

#include <cstddef>

#include "ztensor/zt/cuda/Exception.h"  // ZT_CUDA_CHECK
#include "ztensor/zt/cuda/Guard.h"      // CUDAScopedDevice
#include "ztensor/zt/cuda/Stream.h"     // cuda::GetStream

#include "core/cuda/CudaMemoryPool.h"  // CudaMemoryPool::instance()

namespace zt {
namespace {

// Guard that pins the current device to `device_index` for the duration of a
// call. Asserts the device exists first.
struct ScopedDevice {
    CUDAScopedDevice guard;
    explicit ScopedDevice(int device_index) : guard(device_index) {}
};

}  // namespace

// ── Allocation / free ────────────────────────────────────────────────────────

void* CUDA_Malloc(std::size_t byte_size, int device_index) {
    ScopedDevice scoped(device_index);
    if (byte_size == 0) {
        return nullptr;
    }
    return CudaMemoryPool::instance().allocate(byte_size,
                                               cuda::GetCurrentStream());
}

void CUDA_Free(void* ptr, int device_index) {
    if (ptr == nullptr) {
        return;
    }
    ScopedDevice scoped(device_index);
    CudaMemoryPool::instance().deallocate(ptr, cuda::GetCurrentStream());
}

// ── Copies ───────────────────────────────────────────────────────────────────
// Synchronization contract (matches PyTorch, aten/src/ATen/native/cuda/Copy.cu
// + c10/cuda/CUDAFunctions.h:95-96 `memcpy_and_sync`; LichtFeld-Studio's
// `Tensor::to(Device)` follows the same split):
//
//   • H2D / D2H (host-visible): the copy is followed by a stream synchronize,
//     i.e. it completes before the call returns. This is load-bearing for host
//     buffer lifetime: a caller may pass a short-lived host buffer (notably the
//     factory path `t.to(dst_device)` in TensorFactories.cpp, where the CPU
//     source `t` dies at the return statement). It is also what makes the
//     `.cpu()` pattern safe — `.cpu()` returns a NEW CPU tensor (a D2H temp);
//     synchronizing here means the temp is fully populated at construction, so
//     `x.cpu().data_ptr()` is safe regardless of when the temp dies. PyTorch's
//     `Tensor::cpu()` (TensorBody.h:333) relies on the same guarantee via the
//     default `non_blocking=false` -> `memcpy_and_sync` path.
//
//   • D2D (device-to-device): left asynchronous on the current stream, with NO
//     host sync. Same-device D2D correctness comes from CUDA stream ordering
//     (operations on a stream execute in order); cross-stream hazards are
//     handled by the caching allocator's bridgeStreams / record_stream (event
//     barriers, see CudaMemoryPool), not by a host sync here. Mirrors PyTorch's
//     `copy_device_to_device` (Copy.cu:213), which only inserts event barriers
//     for cross-device copies and otherwise trusts the stream.
//
// All copies run on the current thread-local stream (cuda::GetStream()). On
// the legacy default stream (ztensor's default) the ordering is additionally
// serialized, but the H2D/D2H sync above is what makes the API contract
// correct on a non-default stream too.

void CUDA_MemcpyH2D(void* dst,
                    const void* src,
                    std::size_t n,
                    int device_index) {
    ScopedDevice scoped(device_index);
    const cudaStream_t stream = cuda::GetStream();
    // Host source may be short-lived (factory .to(device) path) -> complete
    // before returning, matching PyTorch's memcpy_and_sync.
    ZT_CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream));
    ZT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CUDA_MemcpyD2H(void* dst,
                    const void* src,
                    std::size_t n,
                    int device_index) {
    ScopedDevice scoped(device_index);
    const cudaStream_t stream = cuda::GetStream();
    // Returned tensor (e.g. Tensor::cpu()) must be fully populated before the
    // caller reads it -> complete before returning.
    ZT_CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream));
    ZT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CUDA_MemcpyD2D(void* dst,
                    const void* src,
                    std::size_t n,
                    int device_index) {
    ScopedDevice scoped(device_index);
    // D2D is stream-ordered: no host sync. Cross-stream reuse is serialized by
    // the allocator (bridgeStreams / record_stream), not here.
    ZT_CUDA_CHECK(cudaMemcpyAsync(
        dst, src, n, cudaMemcpyDeviceToDevice, cuda::GetStream()));
}

}  // namespace zt
