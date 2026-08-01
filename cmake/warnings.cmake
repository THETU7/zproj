# warnings.cmake
#
# Strict warning flags for every zproj target across CXX and CUDA, adopted
# from ztensor's cmake/ZtWarnings.cmake ("abort on warning" philosophy):
#   - GCC/Clang CXX: -Wall -Wextra (-Wno-unused-parameter) [-Werror]
#   - NVCC: host flags travel via -Xcompiler; treat cross-execution-space-call
#           as an error; enable relaxed-constexpr and extended-lambda.
#
# The flags are exposed through an INTERFACE helper target `zproj::warnings`
# so that each target (and downstream consumers) inherits them via a single
# `zproj_enable_warnings(...)` call.
#
# Options (set on the command line / cache):
#   ZPROJ_WARNINGS_AS_ERRORS  (default ON)
#   ZPROJ_ENABLE_SANITIZER    (default OFF; ASan + UBSan, GCC/Clang only)

include_guard(GLOBAL)

if(NOT TARGET zproj::warnings)
    add_library(zproj_warnings INTERFACE)
    add_library(zproj::warnings ALIAS zproj_warnings)

    # Treat warnings as errors when the option is on.
    set(_zproj_cxx_werror
        "$<$<BOOL:${ZPROJ_WARNINGS_AS_ERRORS}>:-Werror>")

    target_compile_options(zproj_warnings INTERFACE
        # ---- GCC / Clang (CXX) ----
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:
            -Wall
            -Wextra
            ${_zproj_cxx_werror}
            -Wno-unused-parameter
        >
    )

    # zproj always builds CUDA (project LANGUAGES includes CUDA); gate on
    # the compiler being present rather than a sub-project option.
    if(CMAKE_CUDA_COMPILER)
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
        if(ZPROJ_WARNINGS_AS_ERRORS)
            set(_zproj_nvcc_xcompiler "-Xcompiler=-Wall,-Wextra,-Werror")
        else()
            set(_zproj_nvcc_xcompiler "-Xcompiler=-Wall,-Wextra")
        endif()

        target_compile_options(zproj_warnings INTERFACE
            # ---- NVCC (CUDA) ----
            $<$<COMPILE_LANGUAGE:CUDA>:
                ${_zproj_nvcc_xcompiler}
                --expt-relaxed-constexpr
                --expt-extended-lambda
                --Werror=cross-execution-space-call
            >
        )
    endif()

    # Sanitizers (opt-in, GCC/Clang only).
    if(ZPROJ_ENABLE_SANITIZER)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            message(WARNING
                "ZPROJ_ENABLE_SANITIZER is only supported on GCC/Clang; ignored.")
        else()
            target_compile_options(zproj_warnings INTERFACE
                -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(zproj_warnings INTERFACE
                -fsanitize=address,undefined)
        endif()
    endif()
endif()

# Convenience function: attach the warning interface to a target.
function(zproj_enable_warnings target)
    target_link_libraries(${target} PUBLIC zproj::warnings)
endfunction()
