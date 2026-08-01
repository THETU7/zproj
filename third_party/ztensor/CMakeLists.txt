# ----------------------------------------------------------------------------
# ztensor: a minimal C++17 CPU + CUDA tensor library (no autograd).
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)

# Read version from a single source of truth.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _zt_version LIMIT_COUNT 1)
string(STRIP "${_zt_version}" _zt_version)
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

option(BUILD_CUDA_MODULE    "Build the CUDA backend (requires NVIDIA toolkit)" ${BUILD_CUDA_MODULE_DEFAULT})
option(BUILD_TESTS          "Build unit tests"                                     ${BUILD_TESTS_DEFAULT})
option(BUILD_EXAMPLES       "Build examples"                                        ${BUILD_EXAMPLES_DEFAULT})
option(BUILD_PYTHON_MODULE  "Build pybind11 Python module (requires pybind11)"     ${BUILD_PYTHON_MODULE_DEFAULT})
option(ZT_ENABLE_SANITIZER  "Enable Address/UndefinedBehaviorSanitizer (Clang/GCC)" OFF)
option(ZT_ENABLE_COVERAGE   "Enable clang source-based coverage report (Clang only)" OFF)
option(ZT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"                    ${ZT_WARNINGS_AS_ERRORS_DEFAULT})

# ----------------------------------------------------------------------------
# CUDA backend (opt-in)
# ----------------------------------------------------------------------------
if(BUILD_CUDA_MODULE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Default to native architecture if the user did not specify one.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
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
find_package(spdlog CONFIG REQUIRED)

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
# CUDA toolkit (only needed for the cuBLAS link path).
# The CUDAToolkit package ships the CUDA::cublas imported target used by the
# CUDA MatMul kernel. Pulled in here so both src/ and tests/ see it.
# ----------------------------------------------------------------------------
if(BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)
endif()

# ----------------------------------------------------------------------------
# Library & subdirectories
# ----------------------------------------------------------------------------
add_subdirectory(src)

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
message(STATUS "  Warnings as errors    : ${ZT_WARNINGS_AS_ERRORS}")
message(STATUS "  Sanitizers            : ${ZT_ENABLE_SANITIZER}")
message(STATUS "  Coverage              : ${ZT_ENABLE_COVERAGE}")
message(STATUS "  Python module         : ${BUILD_PYTHON_MODULE}")
message(STATUS "")
