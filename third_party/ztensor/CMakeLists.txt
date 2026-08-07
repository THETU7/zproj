# ----------------------------------------------------------------------------
# ztensor: a minimal C++17 CPU + CUDA tensor library (no autograd).
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.21)

# Read version from a single source of truth. VERSION may carry a PEP 440
# local-version tag (e.g. "0.1.0+cu118" for the CUDA wheel) — the wheel build
# reads the FULL string via scikit-build-core's regex metadata provider
# (pyproject.toml). CMake's project(VERSION ...) only accepts
# major.minor.patch[.tweak], so split off any "+local" suffix here; the dev
# build's VERSION has no suffix and is unaffected.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _zt_version_raw LIMIT_COUNT 1)
string(STRIP "${_zt_version_raw}" _zt_version_raw)
string(REGEX REPLACE "\\+.*$" "" _zt_version "${_zt_version_raw}")
if(NOT _zt_version)
    set(_zt_version "0.1.0")
endif()

project(ztensor
    VERSION ${_zt_version}
    LANGUAGES CXX
    DESCRIPTION "A minimal C++17 CPU + CUDA tensor library (no autograd)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release if the user did not specify a build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Export compile_commands.json for clangd / clang-tidy.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
# Overridable defaults. When ztensor is embedded as a CMake subdirectory (e.g.
# the zproj integration), the parent project sets e.g.
# BUILD_CUDA_MODULE_DEFAULT=ON / BUILD_TESTS_DEFAULT=OFF before the
# add_subdirectory() call; standalone, the defaults below apply.
if(NOT DEFINED BUILD_CUDA_MODULE_DEFAULT)
    set(BUILD_CUDA_MODULE_DEFAULT OFF)
endif()
if(NOT DEFINED BUILD_HIP_MODULE_DEFAULT)
    set(BUILD_HIP_MODULE_DEFAULT OFF)
endif()
if(NOT DEFINED BUILD_TESTS_DEFAULT)
    set(BUILD_TESTS_DEFAULT ON)
endif()
if(NOT DEFINED BUILD_EXAMPLES_DEFAULT)
    set(BUILD_EXAMPLES_DEFAULT ON)
endif()
if(NOT DEFINED BUILD_PYTHON_MODULE_DEFAULT)
    set(BUILD_PYTHON_MODULE_DEFAULT OFF)
endif()
if(NOT DEFINED ZT_WARNINGS_AS_ERRORS_DEFAULT)
    set(ZT_WARNINGS_AS_ERRORS_DEFAULT ON)
endif()

option(BUILD_CUDA_MODULE    "Build the CUDA backend (requires NVIDIA toolkit)"     ${BUILD_CUDA_MODULE_DEFAULT})
option(BUILD_HIP_MODULE     "Build the AMD HIP/ROCm backend (mutually exclusive with CUDA)" ${BUILD_HIP_MODULE_DEFAULT})
option(BUILD_TESTS          "Build unit tests"                                     ${BUILD_TESTS_DEFAULT})
option(BUILD_EXAMPLES       "Build examples"                                        ${BUILD_EXAMPLES_DEFAULT})
option(BUILD_PYTHON_MODULE  "Build pybind11 Python module (requires pybind11)"     ${BUILD_PYTHON_MODULE_DEFAULT})
option(ZT_ENABLE_SANITIZER  "Enable Address/UndefinedBehaviorSanitizer (Clang/GCC)" OFF)
option(ZT_ENABLE_COVERAGE   "Enable clang source-based coverage report (Clang only)" OFF)
option(ZT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"                    ${ZT_WARNINGS_AS_ERRORS_DEFAULT})
option(ZT_INSTALL_CXX       "Install libztensor + headers + CMake config (CPack/release path)" OFF)

# ----------------------------------------------------------------------------
# GPU backend (opt-in, mutually exclusive): CUDA (NVIDIA) or HIP (AMD ROCm)
# ----------------------------------------------------------------------------
# The HIP backend reuses the CUDA source tree: the same .cu files are compiled
# as HIP (set_source_files_properties(... LANGUAGE HIP)) and a vendor shim
# (include/ztensor/zt/cuda/Vendor.h) remaps cuda*/cublas* -> hip*/hipblas* at
# compile time. To keep every existing #ifdef BUILD_CUDA_MODULE dispatch shim
# and header compiling unchanged, the HIP build ALSO defines BUILD_CUDA_MODULE
# (set in src/CMakeLists.txt). Modeled on ggml's ggml-hip backend.
if(BUILD_CUDA_MODULE AND BUILD_HIP_MODULE)
    message(FATAL_ERROR
        "BUILD_CUDA_MODULE and BUILD_HIP_MODULE are mutually exclusive "
        "(NVIDIA vs AMD GPU backend). Enable at most one.")
endif()

if(BUILD_CUDA_MODULE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Default to native architecture if the user did not specify one.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

if(BUILD_HIP_MODULE)
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
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    # Forward the standard ROCm target variables to CMAKE_HIP_ARCHITECTURES.
    # AMDGPU_TARGETS / GPU_TARGETS are the conventional ROCm spellings
    # (semicolon-separated gfxNNNN list). Default to 'native' for dev builds.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES native)
    endif()

    # Disable link-time optimization build-wide for HIP. hip-clang adds -flto for
    # the offload link by default; LTO's internalize pass then strips entry points
    # that nothing inside the link references — most importantly the pybind11
    # module init PyInit_ztensor (only the Python dynamic loader calls it). With
    # LTO on, the wheel's extension .so loses that symbol and `import ztensor`
    # fails with "does not define module export function". g++/nvcc add no -flto,
    # so the CPU/CUDA builds are unaffected. LTO is an optimization, not a
    # correctness requirement; turning it off here is the clean, global lever.
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

    # ROCm's shared libs (libamdhip64 / libhipblas / librocblas) live under
    # ${ROCM_PATH}/lib, which base ROCm containers do NOT add to the default
    # loader path (no /etc/ld.so.conf.d/rocm.conf, nothing in ldconfig). Bake
    # the dir into the build-tree RPATH so freshly-built executables — including
    # the gtest_discover_tests probe that runs the test binary at build time,
    # and ctest — load the libs without a manual LD_LIBRARY_PATH. auditwheel
    # rewrites RPATH for the wheel regardless.
    if(ROCM_PATH)
        file(TO_CMAKE_PATH "${ROCM_PATH}/lib" _zt_rocm_libdir)
        set(CMAKE_BUILD_RPATH "${_zt_rocm_libdir}")
        list(APPEND CMAKE_INSTALL_RPATH "${_zt_rocm_libdir}")
    endif()
endif()

# ----------------------------------------------------------------------------
# Module path & helper modules
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ZtWarnings)
include(ZtCoverage)

# ----------------------------------------------------------------------------
# spdlog (system-installed, compiled static library)
# ----------------------------------------------------------------------------
# Some CI bases (e.g. AlmaLinux 8) only ship spdlog 1.5 / fmt 6.2, which is
# too old for Log.h's fmt::format_string (needs fmt >= 7). When the system
# copy is insufficient, ZT_USE_VENDORED_SPDLOG FetchContents a recent static
# spdlog with its bundled fmt instead. Default OFF so the validated dev build
# (system spdlog) is unchanged; the wheel build turns it on.
option(ZT_USE_VENDORED_SPDLOG
       "FetchContent a recent spdlog (bundled fmt) instead of find_package" OFF)
if(ZT_USE_VENDORED_SPDLOG)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz
        URL_HASH SHA256=25c843860f039a1600f232c6eb9e01e6627f7d030a2ae5e232bdd3c9205d26cc
    )
    # Static + bundled fmt: no external fmt dependency, and the static lib is
    # linked into ztensor so the wheel ships no libspdlog/libfmt.
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spdlog)
else()
    find_package(spdlog CONFIG REQUIRED)
endif()

# ----------------------------------------------------------------------------
# Eigen (header-only; backs the CPU matmul GEMM).
# Always required — the CPU MatMul kernel maps Eigen over contiguous memory.
# When embedded in a parent project that already provides Eigen3::Eigen (e.g.
# zproj's vendored Eigen), reuse that target instead of finding a system copy.
# ----------------------------------------------------------------------------
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 CONFIG REQUIRED)
endif()

# ----------------------------------------------------------------------------
# GPU backend libraries.
#   * CUDA: CUDAToolkit ships CUDA::cublas (cuBLAS link for the CUDA MatMul).
#   * HIP : roc::hipblas / roc::rocblas back the HIP MatMul (hipBLAS GEMM).
# Pulled in here so both src/ and tests/ see the imported targets.
# ----------------------------------------------------------------------------
if(BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)
endif()
if(BUILD_HIP_MODULE)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(rocblas REQUIRED)
endif()

# ----------------------------------------------------------------------------
# Library & subdirectories
# ----------------------------------------------------------------------------
add_subdirectory(src)

if(ZT_INSTALL_CXX)
    # C++ install/export rules + CPack packaging. Gated OFF by default so the
    # wheel build (scikit-build-core runs `cmake --install`) never ships the
    # C++ artifacts (libztensor.a + headers + cmake config) inside a wheel.
    include(Packaging)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Python bindings (opt-in, DESIGN §8.6.F)
# ----------------------------------------------------------------------------
if(BUILD_PYTHON_MODULE)
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(python)
endif()

# ----------------------------------------------------------------------------
# Status summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "ztensor ${PROJECT_VERSION}")
message(STATUS "  C++ standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA backend          : ${BUILD_CUDA_MODULE}")
if(BUILD_CUDA_MODULE)
    message(STATUS "  CUDA architectures    : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  HIP backend           : ${BUILD_HIP_MODULE}")
if(BUILD_HIP_MODULE)
    message(STATUS "  HIP architectures     : ${CMAKE_HIP_ARCHITECTURES}")
endif()
message(STATUS "  Warnings as errors    : ${ZT_WARNINGS_AS_ERRORS}")
message(STATUS "  Sanitizers            : ${ZT_ENABLE_SANITIZER}")
message(STATUS "  Coverage              : ${ZT_ENABLE_COVERAGE}")
message(STATUS "  Python module         : ${BUILD_PYTHON_MODULE}")
message(STATUS "  Install C++ package   : ${ZT_INSTALL_CXX}")
message(STATUS "")
