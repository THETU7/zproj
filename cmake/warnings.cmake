# Host-compiler warning flags. Kept deliberately light and never applied to
# nvcc (which gets its own, much chattier warnings). We do not enable -Werror
# by default because the bleeding-edge toolchain (g++-15 / CUDA 13.2 / Eigen
# master) will produce occasional benign warnings that would otherwise block
# the build.
function(zproj_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Restrict to CXX only: -Wpedantic on nvcc fires on the #line directives
        # nvcc embeds in its generated host stubs, which is pure noise.
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wpedantic>
        )
        if(ZPROJ_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Werror>
            )
        endif()
    endif()
endfunction()
