# zproj

A learning project for **GPU-accelerated coordinate transformations** (NVIDIA CUDA
and AMD ROCm). The core
idea is to rewrite common [PROJ](https://proj.org/)/[GDAL](https://gdal.org/)
coordinate operations (e.g. WGS84 geodetic ↔ ECEF) on the GPU and compare them
against the GDAL/PROJ reference.

Toolchain: **C++20 · CMake · GDAL (wraps PROJ) · CUDA or HIP/ROCm ·
Eigen (system) · ztensor (vendored)**.

## Build

GDAL, PROJ, a GPU toolchain (nvcc for CUDA, or ROCm's hipcc/hipBLAS/rocBLAS
for HIP), **spdlog**, a recent GCC, and system Eigen are expected on the
system (spdlog is required by the vendored ztensor). The committed presets
are machine-independent and auto-detect the host compiler and GPU
architecture. Machine-specific pins (g++-15 as host/nvcc-host compiler,
`native` GPU arch) live in the gitignored `CMakeUserPresets.json`
(`--preset local`).

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default                 # unit tests
./build/default/bin/wgs84_to_ecef      # GDAL-vs-CUDA comparison + timing
./build/default/bin/rpc               # RPC sensor model (lon/lat/alt <-> col/row) vs GDAL
./build/default/bin/ztensor_basic      # vendored ztensor (CPU + CUDA) smoke
```

GPU-dependent tests/examples skip gracefully when no GPU device is present
(e.g. `zproj.ztensor`). Debug build:
`cmake --preset debug && cmake --build --preset debug`.

## ROCm / HIP build

The GPU backend is selectable through two mutually exclusive options:
`ZPROJ_ZTENSOR_CUDA` (default ON) or `ZPROJ_ZTENSOR_HIP` (OFF by default).
With HIP enabled, zproj's kernels and the vendored ztensor are compiled by
hip-clang, and the `cuda*` -> `hip*` symbol remap is provided by the vendored
ztensor's `cuda/Vendor.h` shim. On a ROCm machine:

```bash
cmake -B build/rocm -G Ninja \
    -DZPROJ_ZTENSOR_CUDA=OFF -DZPROJ_ZTENSOR_HIP=ON \
    -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15 \
    -DCMAKE_HIP_ARCHITECTURES=native
cmake --build build/rocm
ctest --test-dir build/rocm
```

(`local-rocm` is provided as a ready-made user preset in the gitignored
`CMakeUserPresets.json`.)

## Layout

```
zproj/
├── CMakeLists.txt          # C++20 + CUDA/HIP, GDAL, system Eigen, vendored ztensor
├── CMakePresets.json        # machine-independent presets (compiler/GPU auto-detected)
├── AGENTS.md                # code style & constraints (adopted from ztensor)
├── third_party/ztensor/      # vendored ztensor source (editable in-tree)
├── scripts/                   # ztensor <-> zproj sync helpers
│   ├── push-ztensor.sh        #   vendored changes -> upstream ztensor checkout
│   └── sync-ztensor.sh        #   upstream ztensor checkout -> vendored copy
├── include/zproj/           # public headers
│   └── crs/                 #   wgs84 + RPC: constants, types, math, transforms, triangulation
├── src/                     # library sources (.cu kernels + host wrappers)
│   └── crs/                 #   wgs84 <-> ecef + RPC forward/inverse/triangulation (cpu/ + cuda/)
├── examples/                # runnable demos
│   ├── wgs84_to_ecef/       #   GDAL reference vs CUDA kernels (both ways) + timing
│   ├── rpc/                 #   RPC forward/inverse vs GDAL on a real satellite RPC
│   └── ztensor_basic/        #   vendored ztensor CPU + CUDA smoke
└── tests/                   # analytic smoke tests (CTest)
```

## Design notes

- **Reference path** is GDAL/PROJ on the CPU (e.g. EPSG:4326 → EPSG:4978). It is
  the ground truth the GPU kernel is validated against.
- **Fast path** is a CUDA kernel. The geodetic ↔ ECEF math (forward +
  Bowring-method inverse) is identical on host and device (see `ZT_HOST_DEVICE`
  in `wgs84.hpp`), so the CPU reference and the GPU kernel cannot drift apart.
- **Eigen** (system install) is used only for plain host-side math (error
  norms, etc.). GPU/tensor acceleration is provided by the vendored ztensor
  instead — see the ztensor section below.

## Code style & constraints (from ztensor)

zproj adopts ztensor's conventions: same `.clang-format` (Google, 4-space,
80-col) and `.clang-tidy`, `-Wall -Wextra -Werror` by default, grouped
includes, conventional commits, and a performance-first rule. See
[`AGENTS.md`](AGENTS.md).

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
`Eigen3::Eigen` target — the system one here, overridable option defaults).
They are synced back to ztensor via `push-ztensor.sh`, so upstream and the
vendored copy converge.

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

Skeleton + the `wgs84 ↔ ecef` transforms (forward + inverse), with the GDAL
reference path and CUDA kernels that agree to ~1e-8 m. The vendored ztensor
builds with its CUDA backend and is exercised by `examples/ztensor_basic` +
`tests/test_ztensor`.

The RPC sensor model (`zproj::crs`) implements the forward
`lon/lat/alt -> col/row` and the iterative `col/row/alt -> lon/lat` transforms
(GDAL's GDALRPCTransformer, no DEM -- heights are always supplied by the
caller), with host/device-shared math, CPU + CUDA paths, an RPC text-file
parser (`rpc_io`), and a GDAL cross-check on a real GeoEye RPC
(`examples/rpc`, `tests/test_rpc.cpp`).

RPC stereo triangulation (`zproj::crs::RpcStereo`) back-projects a matched
pixel pair through the analytic inverse at two heights to build viewing rays
(`rpc_ray`), intersects them (`triangulate_pair`, plus a pointwise
`triangulate_nview` for N rays), and reports the geodetic point plus an
error metric -- the same ray/intersection math as ASP / VisionWorkbench,
shared host/device with CPU + CUDA paths (`tests/test_triangulation.cpp`).

The convenience `wgs84_to_ecef()` wrapper currently shows ~1× vs the CPU
reference because the single launch is dominated by `cudaMalloc` + H2D/D2H
copies + synchronization, not the kernel itself — the natural next step is to
keep device buffers resident across batches (or port the kernel to ztensor
CUDA tensors) to amortize that overhead.
