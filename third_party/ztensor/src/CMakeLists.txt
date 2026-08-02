# ----------------------------------------------------------------------------
# ztensor library
# ----------------------------------------------------------------------------

# Public headers.
set(_zt_public_headers
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Macros.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Half.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/BFloat16.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/ScalarType.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Scalar.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Layout.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/MemoryFormat.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Device.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/TensorOptions.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/ArrayRef.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/SmallVector.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Blob.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Tensor.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/TensorKey.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/TensorFactories.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/Generator.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/dlpack.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/utility/Log.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/cuda/Exception.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/cuda/Stream.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/cuda/Guard.h
)

# Core sources (compiled in every build; CUDA sources added below).
set(_zt_sources
    core/ScalarType.cpp
    core/Device.cpp
    core/TensorOptions.cpp
    core/Blob.cpp
    core/MemoryManager.cpp
    core/Tensor.cpp
    core/TensorKey.cpp
    core/Generator.cpp
    core/Indexer.cpp
    core/ParallelFor.cpp
    core/AdvancedIndexing.cpp
    kernel/Fill.cpp
    kernel/cpu/FillCPU.cpp
    kernel/BinaryEW.cpp
    kernel/cpu/BinaryEWCPU.cpp
    kernel/UnaryEW.cpp
    kernel/cpu/UnaryEWCPU.cpp
    kernel/TernaryEW.cpp
    kernel/cpu/TernaryEWCPU.cpp
    kernel/Copy.cpp
    kernel/cpu/CopyCPU.cpp
    kernel/Reduction.cpp
    kernel/cpu/ReductionCPU.cpp
    kernel/ArgReduce.cpp
    kernel/cpu/ArgReduceCPU.cpp
    kernel/Scan.cpp
    kernel/cpu/ScanCPU.cpp
    kernel/Index.cpp
    kernel/cpu/IndexCPU.cpp
    kernel/NonZero.cpp
    kernel/cpu/NonZeroCPU.cpp
    kernel/CheckIndexBounds.cpp
    kernel/cpu/CheckIndexBoundsCPU.cpp
    kernel/Scatter.cpp
    kernel/cpu/ScatterCPU.cpp
    kernel/MatMul.cpp
    kernel/cpu/MatMulCPU.cpp
    kernel/Rand.cpp
    kernel/cpu/RandCPU.cpp
    factory/TensorFactories.cpp
    utility/Log.cpp
)

add_library(ztensor ${_zt_sources})
add_library(zt::ztensor ALIAS ztensor)

# CUDA backend sources (only compiled when BUILD_CUDA_MODULE is on).
# Runtime-only .cpp files are compiled by the host compiler and include
# <cuda_runtime.h> through BUILD_CUDA_MODULE-gated headers. Kernel .cu
# files are compiled by NVCC and contain __global__ / <<<>>> syntax.
if(BUILD_CUDA_MODULE)
    set(_zt_cuda_sources
        core/cuda/CUDAUtils.cpp
        core/cuda/CUDAStreamContext.cpp
        core/cuda/CUDAEventPool.cpp
        core/cuda/CudaMemoryPool.cpp
        core/cuda/MemoryManagerCUDA.cpp
        core/cuda/TensorCUDA.cpp
        kernel/cuda/FillCUDA.cu
        kernel/cuda/BinaryEWCUDA.cu
        kernel/cuda/UnaryEWCUDA.cu
        kernel/cuda/TernaryEWCUDA.cu
        kernel/cuda/CopyCUDA.cu
        kernel/cuda/ReductionCUDA.cu
        kernel/cuda/ArgReduceCUDA.cu
        kernel/cuda/ScanCUDA.cu
        kernel/cuda/IndexCUDA.cu
        kernel/cuda/NonZeroCUDA.cu
        kernel/cuda/CheckIndexBoundsCUDA.cu
        kernel/cuda/ScatterCUDA.cu
        kernel/cuda/MatMulCUDA.cu
        kernel/cuda/RandCUDA.cu
    )
    target_sources(ztensor PRIVATE ${_zt_cuda_sources})
endif()

target_include_directories(ztensor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Eigen: header-only, used by the CPU matmul kernel. SYSTEM include silences
# Eigen's own warnings under -Werror without weakening the project's flags.
if(EIGEN3_INCLUDE_DIR)
    target_include_directories(ztensor SYSTEM PRIVATE
        ${EIGEN3_INCLUDE_DIR})
endif()

# spdlog: consumers of public Log.h also need it -> PUBLIC.
target_link_libraries(ztensor PUBLIC spdlog::spdlog)

# Eigen (header-only CPU matmul) and cuBLAS (CUDA matmul). Eigen links in
# every build; cuBLAS only when the CUDA backend is on.
target_link_libraries(ztensor PRIVATE Eigen3::Eigen)
if(BUILD_CUDA_MODULE)
    target_link_libraries(ztensor PRIVATE CUDA::cublas)
endif()

# OpenMP backs the CPU ParallelFor. ParallelFor degrades to serial when it is
# unavailable, so the find is QUIET and the link conditional.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ztensor PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(ztensor PUBLIC cxx_std_17)

# Propagate BUILD_CUDA_MODULE so headers can #ifdef on it.
target_compile_definitions(ztensor PUBLIC
    $<$<BOOL:${BUILD_CUDA_MODULE}>:BUILD_CUDA_MODULE>)

# Version info as compile definitions (usable from Log / build info).
target_compile_definitions(ztensor PUBLIC
    ZT_VERSION="${PROJECT_VERSION}")

# Strict warnings.
zt_target_set_warnings(ztensor)

# Group the public headers in IDEs.
source_group(TREE ${PROJECT_SOURCE_DIR}/include FILES ${_zt_public_headers})

# ----------------------------------------------------------------------------
# Optional Eigen interop module (zt::eigen), header-only.
# ----------------------------------------------------------------------------
# Eigen is already a hard dependency (CPU matmul GEMM). This INTERFACE target
# makes it PUBLIC so consumers who link zt::ztensor_eigen see Eigen's headers;
# the core ztensor target above keeps Eigen PRIVATE, so embedding ztensor as a
# subdirectory never forces Eigen on the parent. The header lives under
# include/ztensor/zt/eigen/ and ships with the include tree.
add_library(ztensor_eigen INTERFACE)
add_library(zt::ztensor_eigen ALIAS ztensor_eigen)
target_link_libraries(ztensor_eigen INTERFACE zt::ztensor Eigen3::Eigen)
target_compile_features(ztensor_eigen INTERFACE cxx_std_17)
