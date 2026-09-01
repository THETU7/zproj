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
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/cuda/Vendor.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/cuda/Warp.h
    ${PROJECT_SOURCE_DIR}/include/ztensor/zt/cuda/VectorIntrinsics.h
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

# GPU backend sources (compiled when either GPU backend is on).
#
# The HIP backend reuses the CUDA source tree verbatim (see the top-level
# CMakeLists.txt): the same runtime .cpp and kernel .cu files are compiled
# unchanged. Under CUDA, NVCC compiles the .cu files; under HIP, hip-clang
# compiles them as the HIP language, and the cuda*/cublas* -> hip*/hipblas*
# API remap lives entirely in include/ztensor/zt/cuda/Vendor.h.
#
# Runtime-only .cpp files are always compiled by the host compiler (they
# include the vendor header through BUILD_CUDA_MODULE-gated headers). The
# kernel .cu files are tracked separately so that, under HIP, only they get
# set_source_files_properties(... LANGUAGE HIP) — the .cpp host files must
# stay CXX.
if(BUILD_CUDA_MODULE OR BUILD_HIP_MODULE)
    set(_zt_cuda_sources
        core/cuda/CUDAUtils.cpp
        core/cuda/CUDAStreamContext.cpp
        core/cuda/CUDAEventPool.cpp
        core/cuda/CudaMemoryPool.cpp
        core/cuda/MemoryManagerCUDA.cpp
        core/cuda/TensorCUDA.cpp
    )
    set(_zt_cuda_kernel_sources
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
    target_sources(ztensor PRIVATE ${_zt_cuda_sources} ${_zt_cuda_kernel_sources})

    # Compile the kernel .cu files as HIP (hip-clang) instead of CUDA (NVCC).
    # Triple-chevron <<<>>> launches and __global__ are natively supported by
    # hip-clang, so no source changes are needed for the launch syntax.
    if(BUILD_HIP_MODULE)
        set_source_files_properties(${_zt_cuda_kernel_sources}
            PROPERTIES LANGUAGE HIP)
    endif()
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

# Eigen (header-only CPU matmul) and the GPU BLAS (CUDA/HIP matmul). Eigen
# links in every build; the BLAS only when a GPU backend is on. cuBLAS backs
# the CUDA MatMul; hipBLAS + rocBLAS back the HIP MatMul (hipBLAS dispatches
# to rocBLAS for the GEMM). hip::host provides the host-side HIP runtime
# symbols (hipMalloc etc.) used by the runtime .cpp files via the vendor shim.
target_link_libraries(ztensor PRIVATE Eigen3::Eigen)
if(BUILD_CUDA_MODULE)
    target_link_libraries(ztensor PRIVATE CUDA::cublas)
elseif(BUILD_HIP_MODULE)
    target_link_libraries(ztensor PRIVATE hip::host roc::rocblas roc::hipblas)
endif()

# OpenMP backs the CPU ParallelFor. ParallelFor degrades to serial when it is
# unavailable, so the find is QUIET and the link conditional.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ztensor PRIVATE OpenMP::OpenMP_CXX)
    # The .a leaves OpenMP symbols unresolved -> consumers must link it too.
    # For STATIC libs CMake auto-propagates PRIVATE deps as $<LINK_ONLY:...> into
    # the exported interface; spelling it out explicitly keeps the requirement
    # correct even if the library ever becomes SHARED (where PRIVATE deps are
    # not propagated at all). $<LINK_ONLY:> avoids dragging OpenMP's includes
    # onto consumers (no public header includes OpenMP). CMake >= 3.13; project
    # min is 3.18. Keeping this makes enabling OpenMP in the release tarball a
    # one-flag flip (Packaging.cmake records ZT_CONFIG_HAS_OPENMP for the
    # installed config).
    target_link_libraries(ztensor INTERFACE $<LINK_ONLY:OpenMP::OpenMP_CXX>)
endif()

target_compile_features(ztensor PUBLIC cxx_std_17)

# Propagate the active GPU backend so headers/dispatch can #ifdef on it.
#
# Both backends define BUILD_CUDA_MODULE: the existing #ifdef BUILD_CUDA_MODULE
# dispatch shims, MemoryManager, and public headers then compile unchanged for
# HIP (the cuda*/cublas* -> hip*/hipblas* remap is selected by BUILD_HIP_MODULE
# inside Vendor.h). BUILD_HIP_MODULE additionally selects the HIP vendor path.
# This mirrors ggml re-exporting GGML_USE_CUDA for its HIP backend.
target_compile_definitions(ztensor PUBLIC
    $<$<BOOL:${BUILD_CUDA_MODULE}>:BUILD_CUDA_MODULE>
    $<$<BOOL:${BUILD_HIP_MODULE}>:BUILD_CUDA_MODULE>
    $<$<BOOL:${BUILD_HIP_MODULE}>:BUILD_HIP_MODULE>)

# Version info as compile definitions (usable from Log / build info).
target_compile_definitions(ztensor PUBLIC
    ZT_VERSION="${PROJECT_VERSION}")

# Strict warnings.
zt_target_set_warnings(ztensor)

# HIP fast-math. hip-clang (the HIP language compiler) is a clang front-end,
# so it takes the same -funsafe-math-optimizations as host clang. We use the
# targeted flag rather than -ffast-math: -ffast-math also enables
# -ffinite-math-only, which assumes no NaN/Inf and breaks ReductionCUDA's
# is_nan checks (and the fp16/bf16 NaN round-trips in Half/BFloat16).
# Scoped to the HIP language so it never touches host CXX compilation.
if(BUILD_HIP_MODULE)
    target_compile_options(ztensor PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-funsafe-math-optimizations>)
endif()

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
