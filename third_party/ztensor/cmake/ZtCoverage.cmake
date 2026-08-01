# ZtCoverage.cmake
#
# Generates a clang source-based coverage report for a test executable. The
# instrumentation itself is added in ZtWarnings.cmake (gated on
# ZT_ENABLE_COVERAGE); this module only builds the *report-generation*
# targets, so it is a no-op unless that option is ON.
#
# Workflow (invoked by `cmake --build build --target coverage`):
#   1. Run the test(s) through ctest with LLVM_PROFILE_FILE pointed at
#      ${CMAKE_BINARY_DIR}/coverage/%p_%m.profraw (one file per process).
#   2. llvm-profdata merge  -> zt.profdata
#   3. llvm-cov export      -> coverage.json   (machine-readable)
#   4. llvm-cov show        -> html/           (annotated source tree)
#   5. llvm-cov report      -> stdout summary  (per-file region/line table)
#
# Coverage is clang-only by design (see ZtWarnings.cmake); the helper refuses
# to register a target under any other compiler.
#
# Usage:
#   zt_add_coverage_target(<test-target> [OUTPUT_DIR <dir>])

include_guard(GLOBAL)

# Resolve the LLVM coverage tools once, globally. Fail fast at configure time
# if they are missing while coverage is requested.
function(_zt_find_coverage_tools)
    find_program(LLVM_PROFDATA_EXE NAMES llvm-profdata)
    find_program(LLVM_COV_EXE NAMES llvm-cov)
    set(LLVM_PROFDATA_EXE ${LLVM_PROFDATA_EXE} PARENT_SCOPE)
    set(LLVM_COV_EXE ${LLVM_COV_EXE} PARENT_SCOPE)
endfunction()

function(zt_add_coverage_target test_target)
    cmake_parse_arguments(ZT_COV "" "OUTPUT_DIR" "" ${ARGN})

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Mirrors ZtWarnings.cmake: a non-clang compiler under
        # ZT_ENABLE_COVERAGE=ON is warned and ignored (configure succeeds, no
        # coverage target registered) rather than a hard error, so the option
        # is a no-op on GCC just like ZT_ENABLE_SANITIZER-on-GCC-unsupported.
        return()
    endif()

    _zt_find_coverage_tools()
    if(NOT LLVM_PROFDATA_EXE OR NOT LLVM_COV_EXE)
        message(FATAL_ERROR
            "zt_add_coverage_target(${test_target}): could not find "
            "llvm-profdata / llvm-cov on PATH. Install llvm-tools.")
    endif()

    if(NOT TARGET ${test_target})
        message(FATAL_ERROR
            "zt_add_coverage_target(${test_target}): no such target.")
    endif()

    set(_out_dir ${CMAKE_BINARY_DIR}/coverage)
    if(ZT_COV_OUTPUT_DIR)
        set(_out_dir ${ZT_COV_OUTPUT_DIR})
    endif()
    file(MAKE_DIRECTORY ${_out_dir})
    file(MAKE_DIRECTORY ${_out_dir}/html)

    set(_profraw_glob "${_out_dir}/*.profraw")
    set(_profdata "${_out_dir}/zt.profdata")
    set(_json "${_out_dir}/coverage.json")
    # llvm-cov wants the binary that owns the coverage mapping.
    set(_bin "$<TARGET_FILE:${test_target}>")
    # Restrict the report to ztensor sources (drop system/gtest headers).
    # Anchored to the repo so absolute build paths still match.
    set(_source_filter "^${CMAKE_SOURCE_DIR}/(src|include)/")

    # Stale profraw from a previous run would poison the merge; clean first.
    # `llvm-profdata merge` does NOT glob — it takes literal file paths — and
    # `llvm-cov export` has no -o flag (writes to stdout), so those two steps
    # run under `sh -c` to expand the profraw glob and redirect the JSON.
    # All commands inherit WORKING_DIRECTORY=${_out_dir}, so globs are relative
    # there; we assume the build path has no spaces (CMake convention).
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E echo "-- Running tests for coverage"
        COMMAND sh -c "rm -f *.profraw *.profdata ${_json}"
        COMMAND ${CMAKE_COMMAND} -E env
                LLVM_PROFILE_FILE=${_out_dir}/%p_%m.profraw
                ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR}
                --output-on-failure
        COMMAND ${CMAKE_COMMAND} -E echo "-- Merging profile data -> ${_profdata}"
        COMMAND sh -c
                "${LLVM_PROFDATA_EXE} merge --failure-mode=any -o ${_profdata} ${_profraw_glob}"
        COMMAND ${CMAKE_COMMAND} -E echo "-- Exporting JSON -> ${_json}"
        COMMAND sh -c
                "${LLVM_COV_EXE} export ${_bin} -instr-profile=${_profdata} --ignore-filename-regex='${_source_filter}' > ${_json}"
        COMMAND ${CMAKE_COMMAND} -E echo
                "-- Rendering HTML -> ${_out_dir}/html/index.html"
        COMMAND ${LLVM_COV_EXE} show ${_bin}
                -instr-profile=${_profdata}
                --format=html
                -show-branches=count
                -show-line-counts-or-regions
                --ignore-filename-regex="${_source_filter}"
                -output-dir=${_out_dir}/html
        COMMAND ${CMAKE_COMMAND} -E echo "-- Per-file summary"
        COMMAND ${LLVM_COV_EXE} report ${_bin}
                -instr-profile=${_profdata}
                --ignore-filename-regex="${_source_filter}"
        DEPENDS ${test_target}
        WORKING_DIRECTORY ${_out_dir}
        VERBATIM
        COMMENT "ztensor: generating clang coverage report"
    )
endfunction()
