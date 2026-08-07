# AGENTS.md

> Onboarding guide for AI agents (and human contributors) working on zproj.

## What zproj is

A learning project for **CUDA-accelerated coordinate transformations**
(PROJ-style): rewrite common GDAL/PROJ operations (e.g. WGS84 geodetic →
ECEF) on the GPU and compare them against the GDAL/PROJ reference.

- Namespace: `zproj` (`zproj::crs::Geodetic`, `zproj::crs::wgs84_to_ecef`,
  `zproj::crs::RpcModel`, ...).
- C++20 + CUDA; GDAL/PROJ is the CPU reference/ground truth, CUDA kernels are
  the fast path. The transform math is shared host/device via `ZT_HOST_DEVICE`
  (`include/zproj/crs/wgs84.hpp`).
- Tensor work (GPU buffers, matmul, reductions, future acceleration) is done
  on the **vendored ztensor**, never directly on Eigen's GPU/Tensor modules.

## Code style & constraints (adopted from ztensor)

- **clang-format** is Google style, 4-space indent, 80-col limit
  (`.clang-format`, same as ztensor). Run `clang-format -i` on changed files;
  `clang-format --dry-run -Werror` must pass. Use `/usr/bin/clang-format`
  (the bare `clang-format` on PATH may be a broken AppImage shim).
- **clang-tidy** config in `.clang-tidy` (bugprone/performance/modernize/
  readability, same as ztensor).
- **Warnings are errors**: `-Wall -Wextra -Werror` is the default
  (`ZPROJ_WARNINGS_AS_ERRORS=ON`). Do not weaken this — fix the root cause.
- **Includes** are grouped via `.clang-format IncludeCategories`:
  C system (incl. CUDA/GDAL) → C++ stdlib → Eigen → `"zproj/..."` /
  `"ztensor/..."` public → `"..."` private. `clang-format -i` sorts them.
- **Performance first.** Never add per-call work that scales with tensor size
  (O(numel) scans, full D2H copies, host loops over device data) for
  validation or diagnostics. Value-level checks are debug-only
  (`#ifndef NDEBUG`) or run on the device.
- **Conventional Commits** are mandatory (`feat:`, `fix:`, `refactor:`,
  `docs:`, `test:`, `chore:`), optionally with a scope.

## Vendored ztensor

`third_party/ztensor/` is the upstream ztensor source, tracked directly in
this repo (llama.cpp-style ggml integration) so it can be edited in place.

- Edit `third_party/ztensor/...`, rebuild, then sync back with
  `./scripts/push-ztensor.sh` (stages in `../ztensor`; commit/push there).
- Refresh from upstream with `./scripts/sync-ztensor.sh` (refuses to clobber
  uncommitted local edits).
- ztensor code follows its own AGENTS.md (`~/code/ztensor/AGENTS.md`) and
  DESIGN.md — read those before touching vendored code.
- The vendored ztensor builds with the same strict warnings
  (`ZT_WARNINGS_AS_ERRORS` follows `ZPROJ_WARNINGS_AS_ERRORS`).

## Build & test

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default

./build/default/bin/wgs84_to_ecef   # GDAL-vs-CUDA comparison + timing
./build/default/bin/ztensor_basic   # vendored ztensor smoke (CPU + CUDA)
```

- The GPU backend is CUDA by default (`ZPROJ_ZTENSOR_CUDA=ON`) with AMD
  ROCm/HIP as the alternative (`-DZPROJ_ZTENSOR_HIP=ON`, mutually exclusive);
  with both off the build is CPU-only. The committed presets are
  machine-independent and auto-detect host compiler and GPU arch.
  Machine-specific pins (gcc-15/g++-15 as host compilers, g++-15 as the nvcc
  host compiler, `CMAKE_CUDA_ARCHITECTURES=native`) live in the gitignored
  `CMakeUserPresets.json` — use `cmake --preset local` (plus
  `--build --preset local` / `ctest --preset local`) for the GPU-tuned build,
  and `local-rocm` for the ROCm/HIP build.
- GPU-dependent tests/examples skip gracefully when no GPU-capable device
  is present (e.g. `zproj.ztensor`, `ztensor_basic`).
- Before finishing work: clang-format clean, `-Werror` clean, `ctest` green,
  conventional-commit message.
