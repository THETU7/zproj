// Minimal CUDA error checking for the library internals (private).
#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace zproj::cuda {

inline void check(cudaError_t err, const char* file, int line) noexcept
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace zproj::cuda

#define ZPROJ_CUDA_CHECK(err) ::zproj::cuda::check((err), __FILE__, __LINE__)
