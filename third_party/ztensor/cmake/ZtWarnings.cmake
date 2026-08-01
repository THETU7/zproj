# ZtWarnings.cmake
#
# Applies strict warning flags to every ztensor target across CXX and CUDA.
# Mirrors Open3D's "abort on warning" philosophy:
#   - GCC/Clang CXX: -Wall -Wextra (-Wno-unused-parameter) [-Werror]
#   - NVCC: host flags travel via -Xcompiler; treat cross-execution-space-call
#           as an error; enable relaxed-constexpr and extended-lambda so that
#           the ParallelFor CUDA path can capture host-device lambdas.
#
# The flags are exposed through an INTERFACE helper target `zt::warnings` so
# that each ztensor target (and downstream consumers) inherits them via a
# single `target_link_libraries(... PUBLIC zt::warnings)`.
#
# Options (set on the command line / cache):
#   ZT_WARNINGS_AS_ERRORS  (default ON)
#   ZT_ENABLE_SANITIZER    (default OFF; ASan + UBSan, GCC/Clang only)
#   ZT_ENABLE_COVERAGE     (default OFF; clang source-based coverage only)

include_guard(GLOBAL)

if(NOT TARGET zt::warnings)
    add_library(zt_warnings INTERFACE)
    add_library(zt::warnings ALIAS zt_warnings)

    # Treat warnings as errors when the option is on.
    set(_zt_cxx_werror
        "$<$<BOOL:${ZT_WARNINGS_AS_ERRORS}>:-Werror>")

    target_compile_options(zt_warnings INTERFACE
        # ---- GCC / Clang (CXX) ----
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:
            -Wall
            -Wextra
            ${_zt_cxx_werror}
            -Wno-unused-parameter
        >
    )

    if(BUILD_CUDA_MODULE)
        # Build the comma-separated -Xcompiler host-flags list once. NVCC
        # expects a single -Xcompiler=a,b,c argument; we cannot rely on
        # generator-expression expansion inside that string, so we resolve
        # Werror at configure time (it is a cache option, not per-config).
        #
        # Each host flag is spelled in its dashed GCC form (-Wall, not Wall):
        # CUDA 13.x's nvcc mis-parses a bare "Wall" (no dash) passed through
        # -Xcompiler when combined with -o + -c, surfacing as the host
        # compiler error "cannot specify -o with -c and multiple files". The
        # dashed form parses cleanly.
        if(ZT_WARNINGS_AS_ERRORS)
            set(_zt_nvcc_xcompiler "-Xcompiler=-Wall,-Wextra,-Werror")
        else()
            set(_zt_nvcc_xcompiler "-Xcompiler=-Wall,-Wextra")
        endif()

        target_compile_options(zt_warnings INTERFACE
            # ---- NVCC (CUDA) ----
            $<$<COMPILE_LANGUAGE:CUDA>:
                ${_zt_nvcc_xcompiler}
                --expt-relaxed-constexpr
                --expt-extended-lambda
                --Werror=cross-execution-space-call
            >
        )
    endif()

    # Sanitizers (opt-in, GCC/Clang only).
    if(ZT_ENABLE_SANITIZER)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            message(WARNING
                "ZT_ENABLE_SANITIZER is only supported on GCC/Clang; ignored.")
        else()
            target_compile_options(zt_warnings INTERFACE
                -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(zt_warnings INTERFACE
                -fsanitize=address,undefined)
        endif()
    endif()

    # Coverage (opt-in, clang only — source-based LLVM instrumentation).
    # GCC has no equivalent of -fcoverage-mapping; -DZT_ENABLE_COVERAGE=ON
    # under GCC is warned and ignored, matching the sanitizer option's UX.
    # Note: instrumentation requires -O0 to be meaningful, so it is forced
    # here regardless of CMAKE_BUILD_TYPE.
    if(ZT_ENABLE_COVERAGE)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            message(WARNING
                "ZT_ENABLE_COVERAGE is only supported on Clang; ignored.")
        else()
            target_compile_options(zt_warnings INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:
                    -fprofile-instr-generate
                    -fcoverage-mapping
                    -O0
                    -g
                >)
            target_link_options(zt_warnings INTERFACE
                -fprofile-instr-generate)
        endif()
    endif()
endif()

# Convenience function: attach the warning interface to a target.
function(zt_target_set_warnings target)
    target_link_libraries(${target} PUBLIC zt::warnings)
endfunction()
