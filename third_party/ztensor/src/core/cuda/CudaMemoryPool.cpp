// ztensor/core/CudaMemoryPool.cu
//
// Out-of-line definitions for CudaMemoryPool — the four-tier CUDA memory pool
// dispatcher.  Compiled only under BUILD_CUDA_MODULE (NVCC).
//
// Design (§8.5.E E-3c): Meyers singleton; allocates via slab → bucketed →
// cudaMallocAsync → cudaMalloc; deallocates by routing through allocation_map_
// back to the correct backend.  Transparent to MemoryManager / Blob.
//
// Modeled on LichtFeld-Studio's CudaMemoryPool (memory_pool.hpp).

#include "core/cuda/CudaMemoryPool.h"

#include <algorithm>  // std::remove
#include <cstdlib>    // std::getenv
#include <sstream>

#include "ztensor/zt/cuda/Exception.h"  // ZT_CUDA_CHECK
#include "ztensor/zt/cuda/Guard.h"      // CUDAScopedDevice
#include "ztensor/zt/cuda/Stream.h"     // GetCurrentStream
#include "ztensor/zt/utility/Log.h"

#include "core/cuda/CUDAEventPool.h"  // bridgeStreams
#include "core/tensor/cuda/internal/DeferredFreeQueue.hpp"
#include "core/tensor/cuda/internal/GPUSlabAllocator.hpp"
#include "core/tensor/cuda/internal/SizeBucketedPool.hpp"

namespace zt {

// ── Environment check
// ──────────────────────────────────────────────────────────

/*static*/ bool CudaMemoryPool::cache_disabled_by_env() {
    const char* val = std::getenv("ZT_DISABLE_CUDA_CACHE");
    return val != nullptr && (val[0] == '1' || val[0] == 'y' || val[0] == 'Y');
}

// ── Singleton
// ──────────────────────────────────────────────────────────────────

CudaMemoryPool& CudaMemoryPool::instance() {
    static CudaMemoryPool pool;
    return pool;
}

// ── Construction / destruction
// ─────────────────────────────────────────────────

CudaMemoryPool::CudaMemoryPool() {
    // Call configure() lazily on first allocate() to avoid touching the CUDA
    // driver during static-init.  We still initialise here in case configure()
    // is called early.
}

CudaMemoryPool::~CudaMemoryPool() { shutdown(); }

// ── configure
// ──────────────────────────────────────────────────────────────────

void CudaMemoryPool::configure() {
    std::call_once(configure_once_, [this]() {
#if CUDART_VERSION >= 11020
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            ZT_LOG_WARNING(
                "CudaMemoryPool: cudaGetDevice failed ({}) — "
                "skipping driver pool config",
                cudaGetErrorString(err));
            slab_enabled_ = true;
            configured_ = true;
            return;
        }

        cudaMemPool_t pool;
        err = cudaDeviceGetDefaultMemPool(&pool, device);
        if (err != cudaSuccess) {
            ZT_LOG_WARNING(
                "CudaMemoryPool: cudaDeviceGetDefaultMemPool failed "
                "({}) — skipping driver pool config",
                cudaGetErrorString(err));
            slab_enabled_ = true;
            configured_ = true;
            return;
        }

        // 64 MiB headroom: keeps typical per-iteration scratch buffers
        // pool-resident while letting the driver reclaim beyond peak spikes.
        // UINT64_MAX hoards indefinitely and causes pool overhead inflation.
        uint64_t threshold = static_cast<uint64_t>(64) << 20;
        ZT_CUDA_CHECK_SOFT(cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &threshold));

        ZT_LOG_INFO(
            "CudaMemoryPool: driver pool configured (device {}, "
            "ReleaseThreshold=64 MiB, CUDA {})",
            device,
            CUDART_VERSION);
#else
        ZT_LOG_INFO("CudaMemoryPool: CUDA {} — driver pool API not available, "
                    "async tier disabled",
                    CUDART_VERSION);
#endif

        slab_enabled_ = true;
        configured_ = true;
    });
}

// ── shutdown
// ───────────────────────────────────────────────────────────────────

void CudaMemoryPool::shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) return;

    if (suspend_deallocations_.load(std::memory_order_acquire)) {
        ZT_LOG_INFO("CudaMemoryPool: shutdown skipped (process exit path)");
        return;
    }

    ZT_LOG_INFO("CudaMemoryPool: shutting down...");
    DeferredFreeQueue::instance().shutdown();
    SizeBucketedPool::instance().shutdown();
    GPUSlabAllocator::instance().shutdown();
    ::zt::cuda::CudaEventPool::instance().shutdown();
}

// ── allocate
// ───────────────────────────────────────────────────────────────────

void* CudaMemoryPool::allocate(size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return nullptr;

    // Lazy initialisation — first allocation triggers driver pool config.
    configure();

    if (shutdown_.load(std::memory_order_acquire)) {
        ZT_LOG_ERROR("CudaMemoryPool: allocate() after shutdown");
        return nullptr;
    }

    // Env-driven bypass: skip all caching and go straight to the driver.
    if (cache_disabled_by_env()) {
        return allocate_direct(bytes);
    }

    void* ptr = nullptr;

    // ── Tier 1: Slab (≤ 256 KiB) ──────────────────────────────────────────
    if (bytes <= SLAB_ALLOC_THRESHOLD && slab_enabled_) {
        ptr = GPUSlabAllocator::instance().allocate(bytes, stream);
        if (ptr) {
            stats_.slab_allocs.fetch_add(1, std::memory_order_relaxed);
            stats_.slab_bytes.fetch_add(bytes, std::memory_order_relaxed);
            track_allocation(ptr, bytes, AllocMethod::Slab, stream);
            return ptr;
        }
    }

    // ── Tier 2: Bucketed (≤ 16 GiB) ───────────────────────────────────────
    if (bytes <= BUCKET_ALLOC_THRESHOLD) {
        ptr = SizeBucketedPool::instance().try_allocate_cached(bytes, stream);
        if (ptr) {
            stats_.bucket_cache_hits.fetch_add(1, std::memory_order_relaxed);
            stats_.bucket_bytes.fetch_add(bytes, std::memory_order_relaxed);
            track_allocation(ptr, bytes, AllocMethod::Bucketed, stream);
            return ptr;
        }

        const size_t bucket_size = SizeBucketedPool::get_bucket_size(bytes);

#if CUDART_VERSION >= 11020
        cudaError_t err = cudaMallocAsync(&ptr, bucket_size, stream);
        if (err == cudaSuccess) {
            stats_.bucket_allocs.fetch_add(1, std::memory_order_relaxed);
            stats_.bucket_bytes.fetch_add(bytes, std::memory_order_relaxed);
            stats_.bucket_waste.fetch_add(bucket_size - bytes,
                                          std::memory_order_relaxed);
            track_allocation(ptr, bytes, AllocMethod::Bucketed, stream);

            // Periodic deferred-free processing (every ~100 bucket + async
            // allocs).
            const uint64_t combined =
                stats_.bucket_allocs.load(std::memory_order_relaxed) +
                stats_.async_allocs.load(std::memory_order_relaxed);
            if (combined % 100 == 0) {
                DeferredFreeQueue::instance().process();
            }

            return ptr;
        }
#endif
    }

    // ── Tier 3: cudaMallocAsync ────────────────────────────────────────────
#if CUDART_VERSION >= 11020
    {
        cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
        if (err == cudaSuccess) {
            stats_.async_allocs.fetch_add(1, std::memory_order_relaxed);
            stats_.async_bytes.fetch_add(bytes, std::memory_order_relaxed);
            track_allocation(ptr, bytes, AllocMethod::Async, stream);
            return ptr;
        }
    }
#endif

    // ── Tier 4: cudaMalloc (direct, with trim-and-retry) ──────────────────
    return allocate_direct(bytes);
}

// ── deallocate
// ─────────────────────────────────────────────────────────────────

void CudaMemoryPool::deallocate(void* ptr, size_t bytes, cudaStream_t stream) {
    (void)bytes;  // size is retrieved from allocation_map_
    if (!ptr) return;
    if (shutdown_.load(std::memory_order_acquire)) return;

    if (suspend_deallocations_.load(std::memory_order_acquire)) {
        // Process is exiting — drop the tracking entry and skip the actual
        // free.  The OS / CUDA context will reclaim everything.
        AllocationInfo info;
        take_allocation(ptr, info);
        return;
    }

    AllocationInfo info;
    if (take_allocation(ptr, info)) {
        free_routed(ptr, info);
        return;
    }

    // Pointer was not tracked — free it asynchronously as a safety measure.
#if CUDART_VERSION >= 11020
    ZT_CUDA_CHECK_SOFT(cudaFreeAsync(ptr, stream));
#else
    (void)bytes;
    (void)stream;
    ZT_CUDA_CHECK_SOFT(cudaFree(ptr));
#endif
}

void CudaMemoryPool::deallocate(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    if (shutdown_.load(std::memory_order_acquire)) return;

    if (suspend_deallocations_.load(std::memory_order_acquire)) {
        AllocationInfo info;
        take_allocation(ptr, info);
        return;
    }

    AllocationInfo info;
    if (take_allocation(ptr, info)) {
        free_routed(ptr, info);
        return;
    }

    // Untracked pointer — best-effort async free.
#if CUDART_VERSION >= 11020
    ZT_CUDA_CHECK_SOFT(cudaFreeAsync(ptr, stream));
#else
    (void)stream;
    ZT_CUDA_CHECK_SOFT(cudaFree(ptr));
#endif
}

// ── Stream helpers
// ─────────────────────────────────────────────────────────────

void CudaMemoryPool::record_stream(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = allocation_map_.find(ptr);
    if (it == allocation_map_.end()) return;
    AllocationInfo& info = it->second;
    if (stream == info.home_stream) return;
    if (std::find(info.extra_streams.begin(),
                  info.extra_streams.end(),
                  stream) == info.extra_streams.end()) {
        info.extra_streams.push_back(stream);
    }
}

void CudaMemoryPool::rehome_stream(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = allocation_map_.find(ptr);
    if (it == allocation_map_.end()) return;
    AllocationInfo& info = it->second;
    if (stream == info.home_stream) return;
    // The old home becomes a recorded extra use.
    if (std::find(info.extra_streams.begin(),
                  info.extra_streams.end(),
                  info.home_stream) == info.extra_streams.end()) {
        info.extra_streams.push_back(info.home_stream);
    }
    info.extra_streams.erase(
        std::remove(
            info.extra_streams.begin(), info.extra_streams.end(), stream),
        info.extra_streams.end());
    info.home_stream = stream;
}

void CudaMemoryPool::release_stream(cudaStream_t stream) {
    if (!stream) return;

    ZT_CUDA_CHECK_SOFT(cudaStreamSynchronize(stream));

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        for (auto& [ptr, info] : allocation_map_) {
            info.extra_streams.erase(std::remove(info.extra_streams.begin(),
                                                 info.extra_streams.end(),
                                                 stream),
                                     info.extra_streams.end());
            if (info.home_stream == stream) {
                info.home_stream = nullptr;  // migrate to legacy stream
            }
        }
    }

    GPUSlabAllocator::instance().merge_stream_into_virgin(stream);
    SizeBucketedPool::instance().retag_stream(stream, nullptr);
}

// ── Maintenance
// ────────────────────────────────────────────────────────────────

void CudaMemoryPool::trim() {
    SizeBucketedPool::instance().trim_cache();
#if CUDART_VERSION >= 11020
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) return;
    (void)cudaMemPoolTrimTo(pool, 0);
#endif
}

void CudaMemoryPool::trim_cached_memory() {
    if (suspend_deallocations_.load(std::memory_order_acquire)) return;

    ZT_CUDA_CHECK_SOFT(cudaDeviceSynchronize());
    DeferredFreeQueue::instance().flush();

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        for (auto& [ptr, info] : allocation_map_) {
            info.extra_streams.clear();
        }
    }

    GPUSlabAllocator::instance().merge_all_streams_into_virgin();
    SizeBucketedPool::instance().retag_all_streams(nullptr);
    SizeBucketedPool::instance().trim_cache();

#if CUDART_VERSION >= 11020
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) return;
    (void)cudaMemPoolTrimTo(pool, 0);
#endif
}

// ── Statistics
// ─────────────────────────────────────────────────────────────────

std::string CudaMemoryPool::get_stats_string() const {
    std::ostringstream oss;
    oss << "CudaMemoryPool Stats:\n";
    oss << "  Slab:     " << stats_.slab_allocs.load() << " allocs ("
        << (stats_.slab_bytes.load() / 1024.0 / 1024.0) << " MiB)\n";
    oss << "  Bucketed: " << stats_.bucket_allocs.load() << " allocs, "
        << stats_.bucket_cache_hits.load() << " cache hits ("
        << (stats_.bucket_bytes.load() / 1024.0 / 1024.0) << " MiB, "
        << (stats_.bucket_waste.load() / 1024.0 / 1024.0) << " MiB wasted)\n";
    oss << "  Async:    " << stats_.async_allocs.load() << " allocs ("
        << (stats_.async_bytes.load() / 1024.0 / 1024.0) << " MiB)\n";
    oss << "  Direct:   " << stats_.direct_allocs.load() << " allocs ("
        << (stats_.direct_bytes.load() / 1024.0 / 1024.0) << " MiB)\n";

#if CUDART_VERSION >= 11020
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
            uint64_t used = 0, reserved = 0;
            (void)cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrUsedMemCurrent, &used);
            (void)cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
            oss << "  CUDA pool: " << (used / 1024.0 / 1024.0) << " / "
                << (reserved / 1024.0 / 1024.0) << " MiB used/reserved\n";
        }
    }
#endif
    return oss.str();
}

void CudaMemoryPool::print_stats() const {
    ZT_LOG_INFO("{}", get_stats_string());
    GPUSlabAllocator::instance().print_stats();
    SizeBucketedPool::instance().print_stats();
}

// ── Private helpers
// ────────────────────────────────────────────────────────────

void CudaMemoryPool::track_allocation(void* ptr,
                                      size_t size,
                                      AllocMethod method,
                                      cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    allocation_map_[ptr] = {size, method, stream, {}};
}

bool CudaMemoryPool::take_allocation(void* ptr, AllocationInfo& info) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = allocation_map_.find(ptr);
    if (it == allocation_map_.end()) return false;
    info = std::move(it->second);
    allocation_map_.erase(it);
    return true;
}

void CudaMemoryPool::free_routed(void* ptr, const AllocationInfo& info) {
    // Bridge every extra cross-stream use back into the home stream so the
    // block is safe to reuse (or free) stream-ordered on home.
    for (cudaStream_t extra : info.extra_streams) {
        ::zt::cuda::bridgeStreams(extra, info.home_stream);
    }

    switch (info.method) {
        case AllocMethod::Slab:
            GPUSlabAllocator::instance().deallocate(
                ptr, info.size, info.home_stream);
            return;
        case AllocMethod::Bucketed:
            SizeBucketedPool::instance().deallocate(
                ptr, info.size, info.home_stream);
            return;
        case AllocMethod::Direct:
            ZT_CUDA_CHECK_SOFT(cudaFree(ptr));
            direct_alloc_count_.fetch_sub(1, std::memory_order_release);
            return;
        case AllocMethod::Async:
            break;  // fall through to cudaFreeAsync below
    }

#if CUDART_VERSION >= 11020
    ZT_CUDA_CHECK_SOFT(cudaFreeAsync(ptr, info.home_stream));
#else
    ZT_CUDA_CHECK_SOFT(cudaFree(ptr));
#endif
}

void* CudaMemoryPool::allocate_direct(size_t bytes) {
    void* ptr = nullptr;

    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        ZT_LOG_WARNING(
            "CudaMemoryPool: cudaMalloc({} B) failed ({}), "
            "trimming and retrying...",
            bytes,
            cudaGetErrorString(err));
        ZT_CUDA_CHECK_SOFT(cudaDeviceSynchronize());
        SizeBucketedPool::instance().trim_cache();
#if CUDART_VERSION >= 11020
        int device = 0;
        if (cudaGetDevice(&device) == cudaSuccess) {
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                (void)cudaMemPoolTrimTo(pool, 0);
            }
        }
#endif
        err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            ZT_LOG_ERROR(
                "CudaMemoryPool: cudaMalloc({} B) retry also "
                "failed ({})",
                bytes,
                cudaGetErrorString(err));
            (void)cudaGetLastError();  // clear sticky error
            return nullptr;
        }
    }

    stats_.direct_allocs.fetch_add(1, std::memory_order_relaxed);
    stats_.direct_bytes.fetch_add(bytes, std::memory_order_relaxed);
    direct_alloc_count_.fetch_add(1, std::memory_order_release);

    track_allocation(ptr, bytes, AllocMethod::Direct);
    return ptr;
}

}  // namespace zt
