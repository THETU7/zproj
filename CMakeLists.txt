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
option(ZPROJ_WARNINGS_AS_ERRORS "Treat host compiler warnings as errors" OFF)

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

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------------------
# Eigen (vendored submodule — never the system copy). The submodule gives us
# the latest Eigen, whose GPU/Tensor modules we rely on for acceleration.
# We expose it as a header-only INTERFACE target aliased to Eigen3::Eigen so
# downstream code can use it the same way as a system Eigen would be found.
# ---------------------------------------------------------------------------
set(ZPROJ_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
if(NOT EXISTS "${ZPROJ_EIGEN_DIR}/Eigen/Core")
    message(FATAL_ERROR
        "Eigen submodule not found at '${ZPROJ_EIGEN_DIR}'.\n"
        "Fetch it with:\n"
        "    git submodule update --init --recursive")
endif()

add_library(zproj_eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS zproj_eigen)
target_include_directories(zproj_eigen SYSTEM INTERFACE "${ZPROJ_EIGEN_DIR}")
target_compile_definitions(zproj_eigen INTERFACE EIGEN_MPL2_ONLY)

# ---------------------------------------------------------------------------
# GDAL (wraps PROJ; used for the CPU reference / ground-truth transforms).
# ---------------------------------------------------------------------------
find_package(GDAL REQUIRED)

# CUDA runtime, so the static library can propagate cudart to C++-only
# consumers (examples, tests) that never touch nvcc themselves.
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(zproj STATIC
    src/crs/wgs84_to_ecef.cu
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
    PRIVATE
        GDAL::GDAL
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
