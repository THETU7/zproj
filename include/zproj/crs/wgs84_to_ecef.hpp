// Public host API for the CUDA-accelerated WGS84 geodetic -> ECEF transform.
//
// The signatures intentionally avoid CUDA types, so plain C++ consumers
// (examples, tests, downstream apps) can call these without pulling in the
// CUDA headers or being compiled by nvcc.
#pragma once

#include <cstddef>
#include <vector>

#include "zproj/crs/wgs84.hpp"

namespace zproj::crs {

// Transform an array of geodetic points to ECEF on the GPU.
//
// Convenience wrapper: it allocates device memory, copies the input to the
// device, launches the kernel, synchronizes, and copies the result back.
// Intended for ease of use / learning; for sustained throughput prefer
// wgs84_to_ecef_device() with your own device buffers and streams.
std::vector<Ecef> wgs84_to_ecef(const std::vector<Geodetic>& geodetic);

// Launch the kernel on device-resident buffers (the default stream).
// d_in and d_out must each hold at least `n` elements.
void wgs84_to_ecef_device(const Geodetic* d_in, Ecef* d_out, std::size_t n);

}  // namespace zproj::crs
