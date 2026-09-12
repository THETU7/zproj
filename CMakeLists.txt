cmake_minimum_required(VERSION 3.28)

project(zproj
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated coordinate transformations (PROJ-style), for learning"
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ZPROJ_BUILD_EXAMPLES "Build zproj examples" ON)
option(ZPROJ_BUILD_TESTS "Build zproj tests" ON)
option(ZPROJ_WARNINGS_AS_ERRORS "Treat host compiler warnings as errors" ON)
option(ZPROJ_ENABLE_SANITIZER  "Enable Address/UndefinedBehaviorSanitizer (GCC/Clang)" OFF)

option(ZPROJ_ZTENSOR      "Build the vendored ztensor (third_party/ztensor)" ON)
option(ZPROJ_ZTENSOR_CUDA "Build the vendored ztensor's CUDA backend"        ON)
option(ZPROJ_ZTENSOR_HIP  "Build the vendored ztensor's HIP/ROCm backend"    OFF)

if(ZPROJ_ZTENSOR_CUDA AND ZPROJ_ZTENSOR_HIP)
    message(FATAL_ERROR
        "ZPROJ_ZTENSOR_CUDA and ZPROJ_ZTENSOR_HIP are mutually exclusive "
        "(NVIDIA vs AMD GPU backend). Enable at most one.")
endif()

# ---------------------------------------------------------------------------
# GPU backend language (CUDA or HIP; both off -> CPU-only build). The GPU
# kernels live in src/crs/cuda/*.cu and are compiled by nvcc (CUDA) or by
# hip-clang as the HIP language (ROCm), mirroring the vendored ztensor's
# backend selection.
# ---------------------------------------------------------------------------
if(ZPROJ_ZTENSOR_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
elseif(ZPROJ_ZTENSOR_HIP)
    # ROCm path discovery: honor ROCM_PATH / ROCM_ROOT_DIR, else /opt/rocm.
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH})
        elseif(EXISTS "/opt/rocm")
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()
    if(ROCM_PATH)
        list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
        list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
    endif()

    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)

    # Forward the conventional ROCm target variables to
    # CMAKE_HIP_ARCHITECTURES; default to 'native' for dev builds (same as
    # the vendored ztensor).
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES native)
    endif()

    # hip-clang adds -flto for the offload link; LTO's internalize pass
    # strips entry points nothing in the link references (the vendored
    # ztensor hit this with its Python module init). Disable it build-wide
    # for HIP, same as upstream.
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

    # ROCm shared libs (libamdhip64 / libhipblas / librocblas) live under
    # ${ROCM_PATH}/lib, which base ROCm installs do not add to the loader
    # path. Bake the dir into the build-tree RPATH so freshly-built
    # executables (and the gtest_discover_tests build-time probes) load them
    # without a manual LD_LIBRARY_PATH.
    if(ROCM_PATH)
        file(TO_CMAKE_PATH "${ROCM_PATH}/lib" _zproj_rocm_libdir)
        set(CMAKE_BUILD_RPATH "${_zproj_rocm_libdir}")
        list(APPEND CMAKE_INSTALL_RPATH "${_zproj_rocm_libdir}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Standards. Host C++ is always C++20.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# zproj uses no C++20 modules; disable CMake's module dependency scanning
# (CMake >= 4 enables it by default for Ninja + GCC 15). The scanning injects
# GCC-only flags (-fmodules-ts, -fmodule-mapper=..., -fdeps-format=p1689r5)
# into every C++ compile command, which clangd cannot parse and which are pure
# overhead here.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------------------
# Eigen (system install), used for plain host-side math only. GPU/tensor
# acceleration goes through the vendored ztensor (CUDA tensors), not Eigen's
# GPU/Tensor modules.
# ---------------------------------------------------------------------------
find_package(Eigen3 CONFIG REQUIRED)


# ---------------------------------------------------------------------------
# GDAL (wraps PROJ; used for the CPU reference / ground-truth transforms).
# ---------------------------------------------------------------------------
find_package(GDAL REQUIRED)

# GPU runtime, so the static library can propagate the runtime to C++-only
# consumers (examples, tests) that never touch nvcc/hipcc themselves:
# CUDA::cudart for NVIDIA, hip::host + roc::hipblas for AMD ROCm. hipblas is
# PUBLIC on zproj because the vendored ztensor's cuda/Vendor.h shim includes
# <hipblas/hipblas.h> in every HIP translation unit (ztensor itself links it
# PRIVATE).
if(ZPROJ_ZTENSOR_CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(ZPROJ_ZTENSOR_HIP)
    find_package(hip CONFIG REQUIRED)
    find_package(hipblas CONFIG REQUIRED)
endif()

# OpenMP backs the CPU wgs84_to_ecef loop. Linked only when available; the
# loop degrades to serial otherwise (the source guards on _OPENMP).
find_package(OpenMP QUIET)

# ---------------------------------------------------------------------------
# ztensor (vendored upstream -- llama.cpp-style ggml integration).
#
# The upstream ztensor source is tracked directly in third_party/ztensor so it
# can be edited in place and rebuilt here. Push local edits back to the
# upstream checkout with scripts/push-ztensor.sh, and refresh the vendored
# copy from upstream with scripts/sync-ztensor.sh.
# ---------------------------------------------------------------------------
set(ZPROJ_ZTENSOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ztensor")
if(ZPROJ_ZTENSOR)
  if(NOT EXISTS "${ZPROJ_ZTENSOR_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
            "Vendored ztensor not found at '${ZPROJ_ZTENSOR_DIR}'.\n"
            "Restore it with:\n"
            "    ./scripts/sync-ztensor.sh")
  endif()

  # Override ztensor's option defaults the same way llama.cpp drives ggml:
  # the vendored tree keeps upstream-compatible options, and zproj picks the
  # defaults that make sense when embedding it (GPU backend on, no standalone
  # examples/tests/python, warnings-as-errors following ZPROJ_*).
  set(BUILD_CUDA_MODULE_DEFAULT    ${ZPROJ_ZTENSOR_CUDA})
  set(BUILD_HIP_MODULE_DEFAULT     ${ZPROJ_ZTENSOR_HIP})
  set(BUILD_TESTS_DEFAULT          OFF)
  set(BUILD_EXAMPLES_DEFAULT       OFF)
  set(BUILD_PYTHON_MODULE_DEFAULT  OFF)
  set(ZT_WARNINGS_AS_ERRORS_DEFAULT ${ZPROJ_WARNINGS_AS_ERRORS})

  add_subdirectory(${ZPROJ_ZTENSOR_DIR})
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(_zproj_gpu_sources
    src/crs/cuda/wgs84_to_ecef_cuda.cu
    src/crs/cuda/ecef_to_wgs84_cuda.cu
    src/crs/cuda/rpc_cuda.cu
    src/crs/cuda/triangulation_cuda.cu
)

set(_zproj_sources
    src/crs/wgs84_to_ecef.cpp
    src/crs/cpu/wgs84_to_ecef_cpu.cpp
    src/crs/ecef_to_wgs84.cpp
    src/crs/cpu/ecef_to_wgs84_cpu.cpp
    src/crs/rpc.cpp
    src/crs/cpu/rpc_cpu.cpp
    src/crs/rpc_io.cpp
    src/crs/triangulation.cpp
    src/crs/cpu/triangulation_cpu.cpp
)

# Compile the kernel .cu files as HIP (hip-clang) instead of CUDA (NVCC)
# under the ROCm backend. Triple-chevron launches and __global__ are native
# to hip-clang; the cuda* -> hip* symbol remap is handled by the vendored
# ztensor's cuda/Vendor.h shim (selected by BUILD_HIP_MODULE).
if(ZPROJ_ZTENSOR_HIP)
    set_source_files_properties(${_zproj_gpu_sources} PROPERTIES LANGUAGE HIP)
endif()

if(ZPROJ_ZTENSOR_CUDA OR ZPROJ_ZTENSOR_HIP)
    list(APPEND _zproj_sources ${_zproj_gpu_sources})
endif()

add_library(zproj STATIC ${_zproj_sources})
add_library(zproj::zproj ALIAS zproj)

target_include_directories(zproj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(ZPROJ_ZTENSOR_CUDA)
    set(_zproj_gpu_runtime CUDA::cudart)
elseif(ZPROJ_ZTENSOR_HIP)
    set(_zproj_gpu_runtime hip::host roc::hipblas)
endif()

target_link_libraries(zproj
    PUBLIC
        Eigen3::Eigen
        zt::ztensor
        $<$<BOOL:${ZPROJ_ZTENSOR_CUDA}>:CUDA::cudart>
        $<$<BOOL:${ZPROJ_ZTENSOR_HIP}>:hip::host roc::hipblas>
    PRIVATE
        GDAL::GDAL
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

include(cmake/warnings.cmake)
zproj_enable_warnings(zproj)

# ---------------------------------------------------------------------------
# RPC affine refinement (zproj_refine): the Ceres-based solver for image-space
# RPC bias correction (rpc_affine.hpp). A separate static library so the core
# zproj target stays Ceres-free; its public headers carry no Ceres includes.
# ---------------------------------------------------------------------------
option(ZPROJ_BUILD_RPC_REFINE
       "Build the Ceres-based RPC affine refinement library (zproj_refine)" ON)
if(ZPROJ_BUILD_RPC_REFINE)
    find_package(Ceres 2.2 CONFIG REQUIRED)
    add_library(zproj_refine STATIC
        src/crs/rpc_affine.cpp
        src/crs/rpc_bundle_adjust.cpp
        src/crs/rpc_bundle_adjust_gridded.cpp
    )
    add_library(zproj::refine ALIAS zproj_refine)
    target_link_libraries(zproj_refine
        PUBLIC
            zproj
        PRIVATE
            Ceres::ceres
            # OpenMP parallelizes the match-triangulation initialization loop
            # (the pragma is inert without it).
            $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
    )
    zproj_enable_warnings(zproj_refine)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(ZPROJ_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(ZPROJ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
