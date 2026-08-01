# zproj

A learning project for **CUDA-accelerated coordinate transformations**. The core
idea is to rewrite common [PROJ](https://proj.org/)/[GDAL](https://gdal.org/)
coordinate operations (e.g. WGS84 geodetic → ECEF) on the GPU and compare them
against the GDAL/PROJ reference.

Toolchain: **C++20 · CMake · GDAL (wraps PROJ) · CUDA · Eigen (vendored) · ztensor (vendored)**.

## Build

GDAL, PROJ, CUDA (nvcc), **spdlog**, and a recent GCC are expected on the
system (spdlog is required by the vendored ztensor).
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
./build/default/bin/ztensor_basic      # vendored ztensor (CPU + CUDA) smoke
```

Debug build: `cmake --preset debug && cmake --build --preset debug`.

## Layout

```
zproj/
├── CMakeLists.txt          # C++20 + CUDA, GDAL, vendored Eigen
├── CMakePresets.json        # g++-15 / nvcc, native GPU arch
├── third_party/eigen/       # git submodule (latest Eigen, GPU/Tensor modules)
├── third_party/ztensor/      # vendored ztensor source (editable in-tree)
├── scripts/                   # ztensor <-> zproj sync helpers
│   ├── push-ztensor.sh        #   vendored changes -> upstream ztensor checkout
│   └── sync-ztensor.sh        #   upstream ztensor checkout -> vendored copy
├── include/zproj/           # public headers
│   └── crs/                 #   wgs84 constants + geodetic/ECEF types + math
├── src/                     # library sources (.cu kernels + host wrappers)
│   ├── crs/                 #   CUDA implementations
│   └── zproj/cuda/          #   private CUDA helpers (error checking)
├── examples/                # runnable demos
│   ├── wgs84_to_ecef/       #   GDAL reference vs CUDA kernel + timing
│   ├── eigen_gpu/           #   validates Eigen's GPU Core + Tensor modules (.cu)
│   └── ztensor_basic/        #   vendored ztensor CPU + CUDA smoke
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

## Vendored ztensor (llama.cpp-style ggml integration)

[ztensor](https://github.com/THETU7/ztensor) is integrated the same way
[llama.cpp integrates ggml](https://github.com/ggml-org/llama.cpp/tree/master/ggml):
the upstream source is **tracked directly in this repository** (not a git
submodule) at `third_party/ztensor/`, so it can be edited in place and rebuilt
as part of the normal `cmake --build` flow. Changes are then synced back to the
ztensor repository with the helper scripts.

Only the build-essential file set is vendored (the same kind of file set
llama.cpp keeps for ggml): `CMakeLists.txt`, `VERSION`, `cmake/`, `include/`,
`src/`. The upstream examples/tests/python bindings stay in the ztensor repo.

### CMake integration

`CMakeLists.txt` mirrors llama.cpp's ggml block: before `add_subdirectory(
third_party/ztensor)` it overrides ztensor's option defaults —

| ztensor option            | default in zproj      | why |
|---------------------------|-----------------------|-----|
| `BUILD_CUDA_MODULE`       | `ON`                  | zproj is CUDA-first |
| `BUILD_TESTS`             | `OFF`                 | ztensor tests live upstream |
| `BUILD_EXAMPLES`          | `OFF`                 | ditto |
| `BUILD_PYTHON_MODULE`     | `OFF`                 | ditto |
| `ZT_WARNINGS_AS_ERRORS`   | follows `ZPROJ_WARNINGS_AS_ERRORS` | one switch for the tree |

The vendored copy also carries a few small, upstream-friendly CMake patches
that make ztensor behave correctly when embedded as a subdirectory
(`PROJECT_SOURCE_DIR` instead of `CMAKE_SOURCE_DIR`, reuse an existing
`Eigen3::Eigen` target, overridable option defaults). They are synced back to
ztensor via `push-ztensor.sh`, so upstream and the vendored copy converge.

### Editing ztensor code here

```bash
# edit third_party/ztensor/... freely, then rebuild
cmake --build --preset default
ctest --preset default

# stage the same changes in the sibling upstream checkout (~/code/ztensor),
# then review / commit / push there as usual
./scripts/push-ztensor.sh

# later: refresh the vendored copy from upstream (overwrites uncommitted
# local edits -- push them first!)
./scripts/sync-ztensor.sh
```

`scripts/sync-ztensor.last` records the upstream commit the vendored copy was
last synced from.

## Status

Skeleton + first transform (`wgs84 → ecef`), with the GDAL reference path and a
CUDA kernel that agree to ~1e-8 m. Eigen's GPU Core and Tensor modules are
verified to compile and run in this toolchain (`examples/eigen_gpu`).

The convenience `wgs84_to_ecef()` wrapper currently shows ~1× vs the CPU
reference because the single launch is dominated by `cudaMalloc` + H2D/D2H
copies + synchronization, not the kernel itself — the natural next step is to
keep device buffers resident across batches (or port the kernel to
`Eigen::Tensor` + `GpuDevice`) to amortize that overhead.
