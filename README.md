# zproj

A learning project for **CUDA-accelerated coordinate transformations**. The core
idea is to rewrite common [PROJ](https://proj.org/)/[GDAL](https://gdal.org/)
coordinate operations (e.g. WGS84 geodetic → ECEF) on the GPU and compare them
against the GDAL/PROJ reference.

Toolchain: **C++20 · CMake · GDAL (wraps PROJ) · CUDA · Eigen (vendored)**.

## Build

GDAL, PROJ, CUDA (nvcc), and a recent GCC are expected on the system.
The CMake preset pins **g++-15** as both the host C++ compiler and the nvcc
host compiler (CUDA 13.2 does not yet accept GCC 16), and lets CMake
auto-detect the local GPU architecture (`native`).

```bash
# one-time: fetch the vendored Eigen submodule (latest, for GPU/Tensor modules)
git submodule update --init --recursive

cmake --preset default
cmake --build --preset default
ctest --preset default                 # unit tests
./build/default/bin/wgs84_to_ecef      # GDAL-vs-CUDA comparison + timing
./build/default/bin/eigen_gpu          # Eigen GPU/Tensor modules sanity check
```

Debug build: `cmake --preset debug && cmake --build --preset debug`.

## Layout

```
zproj/
├── CMakeLists.txt          # C++20 + CUDA, GDAL, vendored Eigen
├── CMakePresets.json        # g++-15 / nvcc, native GPU arch
├── third_party/eigen/       # git submodule (latest Eigen, GPU/Tensor modules)
├── include/zproj/           # public headers
│   └── crs/                 #   wgs84 constants + geodetic/ECEF types + math
├── src/                     # library sources (.cu kernels + host wrappers)
│   ├── crs/                 #   CUDA implementations
│   └── zproj/cuda/          #   private CUDA helpers (error checking)
├── examples/                # runnable demos
│   ├── wgs84_to_ecef/       #   GDAL reference vs CUDA kernel + timing
│   └── eigen_gpu/           #   validates Eigen's GPU Core + Tensor modules (.cu)
└── tests/                   # analytic smoke tests (CTest)
```

## Design notes

- **Reference path** is GDAL/PROJ on the CPU (e.g. EPSG:4326 → EPSG:4978). It is
  the ground truth the GPU kernel is validated against.
- **Fast path** is a CUDA kernel. The geodetic → ECEF math is identical on host
  and device (see `ZPROJ_HD` in `wgs84.hpp`), so the CPU reference and the GPU
  kernel cannot drift apart.
- **Eigen** is vendored as a submodule (never the system copy) so the latest
  GPU/Tensor modules are available for more advanced acceleration work.

## Status

Skeleton + first transform (`wgs84 → ecef`), with the GDAL reference path and a
CUDA kernel that agree to ~1e-8 m. Eigen's GPU Core and Tensor modules are
verified to compile and run in this toolchain (`examples/eigen_gpu`).

The convenience `wgs84_to_ecef()` wrapper currently shows ~1× vs the CPU
reference because the single launch is dominated by `cudaMalloc` + H2D/D2H
copies + synchronization, not the kernel itself — the natural next step is to
keep device buffers resident across batches (or port the kernel to
`Eigen::Tensor` + `GpuDevice`) to amortize that overhead.
