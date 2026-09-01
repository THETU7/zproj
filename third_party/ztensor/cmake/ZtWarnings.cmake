# ZtWarnings.cmake
#
# Applies strict warning flags to every ztensor target across CXX and CUDA.
# Mirrors Open3D's "abort on warning" philosophy:
#   - GCC/Clang CXX: -Wall -Wextra (-Wno-unused-parameter) [-Werror]
#   - NVCC: host flags travel via -Xcompiler; treat cross-execution-space-call
#           as an error; enable relaxed-constexpr and extended-lambda so that
#           the ParallelFor CUDA path can capture host-device lambdas.
#
# The flags are exposed through an INTERFACE helper target `zt::warnings`,
# attached per-target via zt_target_set_warnings with a $<BUILD_INTERFACE:...>
# link. They are a build-time implementation concern and must NOT leak into
# the exported interface: install(EXPORT) would fail (zt::warnings is not in
# the export set) and shipped consumers would inherit -Wall -Wextra -Werror.
# In-tree consumers (tests/examples/python) call the helper on their own
# targets.
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
        #
        # -Wno-unused-local-typedefs: nvcc's host pass cannot see uses of a
        # local typedef that occur only inside a __device__ lambda, so -Wall's
        # -Wunused-local-typedefs fires false positives — in our
        # ZT_DISPATCH_SCALARTYPE_* macros (scalar_t) and inside the vendored
        # fmt/spdlog headers (char_type, mapped_type, ...). Suppress it for
        # CUDA TUs only; CXX builds keep the warning (and pass cleanly).
        if(ZT_WARNINGS_AS_ERRORS)
            set(_zt_nvcc_xcompiler
                "-Xcompiler=-Wall,-Wextra,-Werror,-Wno-unused-local-typedefs")
        else()
            set(_zt_nvcc_xcompiler
                "-Xcompiler=-Wall,-Wextra,-Wno-unused-local-typedefs")
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
    # PUBLIC $<BUILD_INTERFACE:...>: applies the flags in the build tree only;
    # install(EXPORT) strips the BUILD_INTERFACE part, so the packaged library
    # never references (or exports) zt::warnings. Do NOT use plain PUBLIC
    # (would export -Wall -Wextra -Werror to consumers) or plain PRIVATE (CMake
    # auto-wraps a STATIC library's PRIVATE deps as $<LINK_ONLY:...>, which
    # install(EXPORT) refuses because the helper is not in the export set).
    target_link_libraries(${target} PUBLIC $<BUILD_INTERFACE:zt::warnings>)
endfunction()
