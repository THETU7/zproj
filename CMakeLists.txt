cmake_minimum_required(VERSION 3.28)

project(zproj
    VERSION 0.1.0
    DESCRIPTION "CUDA-accelerated coordinate transformations (PROJ-style), for learning"
    LANGUAGES C CXX CUDA
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ZPROJ_BUILD_EXAMPLES "Build zproj examples" ON)
option(ZPROJ_BUILD_TESTS "Build zproj tests" ON)
option(ZPROJ_WARNINGS_AS_ERRORS "Treat host compiler warnings as errors" ON)
option(ZPROJ_ENABLE_SANITIZER  "Enable Address/UndefinedBehaviorSanitizer (GCC/Clang)" OFF)

# ---------------------------------------------------------------------------
# Standards. CUDA host compiler / architecture are resolved through the
# CMake preset (CMAKE_CUDA_HOST_COMPILER, CMAKE_CUDA_ARCHITECTURES).
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

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

# CUDA runtime, so the static library can propagate cudart to C++-only
# consumers (examples, tests) that never touch nvcc themselves.
find_package(CUDAToolkit REQUIRED)

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
option(ZPROJ_ZTENSOR      "Build the vendored ztensor (third_party/ztensor)" ON)
option(ZPROJ_ZTENSOR_CUDA "Build the vendored ztensor's CUDA backend"        ON)

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
  # defaults that make sense when embedding it (CUDA on, no standalone
  # examples/tests/python, warnings-as-errors following ZPROJ_*).
  set(BUILD_CUDA_MODULE_DEFAULT    ${ZPROJ_ZTENSOR_CUDA})
  set(BUILD_TESTS_DEFAULT          OFF)
  set(BUILD_EXAMPLES_DEFAULT       OFF)
  set(BUILD_PYTHON_MODULE_DEFAULT  OFF)
  set(ZT_WARNINGS_AS_ERRORS_DEFAULT ${ZPROJ_WARNINGS_AS_ERRORS})

  add_subdirectory(${ZPROJ_ZTENSOR_DIR})
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(zproj STATIC
    src/crs/wgs84_to_ecef.cpp
    src/crs/cpu/wgs84_to_ecef_cpu.cpp
    src/crs/cuda/wgs84_to_ecef_cuda.cu
    src/crs/ecef_to_wgs84.cpp
    src/crs/cpu/ecef_to_wgs84_cpu.cpp
    src/crs/cuda/ecef_to_wgs84_cuda.cu
    src/crs/rpc.cpp
    src/crs/cpu/rpc_cpu.cpp
    src/crs/cuda/rpc_cuda.cu
    src/crs/rpc_io.cpp
)
add_library(zproj::zproj ALIAS zproj)

target_include_directories(zproj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(zproj
    PUBLIC
        Eigen3::Eigen
        CUDA::cudart
        zt::ztensor
    PRIVATE
        GDAL::GDAL
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

include(cmake/warnings.cmake)
zproj_enable_warnings(zproj)

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
