# GoogleTest-driven, like the vendored ztensor's own test suite: the framework
# is discovered through the system install via find_package (CONFIG) -- never
# fetched or vendored. gtest_discover_tests registers each TEST() as its own
# ctest entry.
find_package(GTest CONFIG REQUIRED)

add_executable(test_wgs84_to_ecef test_wgs84_to_ecef.cpp)
target_link_libraries(test_wgs84_to_ecef PRIVATE zproj GTest::gtest)
include(GoogleTest)
gtest_discover_tests(test_wgs84_to_ecef)

add_executable(test_ecef_to_wgs84 test_ecef_to_wgs84.cpp)
target_link_libraries(test_ecef_to_wgs84 PRIVATE zproj GTest::gtest)
include(GoogleTest)
gtest_discover_tests(test_ecef_to_wgs84)

add_executable(test_rpc test_rpc.cpp)
target_link_libraries(test_rpc PRIVATE zproj GDAL::GDAL GTest::gtest)
# Path to the vendored real-world RPC sample used by the GDAL cross-check.
target_compile_definitions(test_rpc PRIVATE
    ZPROJ_TEST_RPC_DATA="${CMAKE_CURRENT_SOURCE_DIR}/../examples/rpc/data/md_ge_rgb_0010000_rpc.txt")
include(GoogleTest)
gtest_discover_tests(test_rpc)

add_executable(test_ztensor test_ztensor.cpp)
target_link_libraries(test_ztensor PRIVATE zt::ztensor ${_zproj_gpu_runtime})
add_test(NAME zproj.ztensor COMMAND test_ztensor)

add_executable(test_triangulation test_triangulation.cpp)
target_link_libraries(test_triangulation PRIVATE zproj GTest::gtest)
include(GoogleTest)
gtest_discover_tests(test_triangulation)

# RPC affine bias-correction solver (Ceres): synthetic corruption-recovery
# tests. Skipped entirely when ZPROJ_BUILD_RPC_REFINE=OFF.
if(TARGET zproj::refine)
    add_executable(test_rpc_affine test_rpc_affine.cpp)
    target_link_libraries(test_rpc_affine PRIVATE zproj zproj::refine GTest::gtest)
    include(GoogleTest)
    gtest_discover_tests(test_rpc_affine)

    # N-view RPC bundle adjustment (Ceres): control-network corruption
    # recovery, partial visibility, gauge constraints, GCP anchoring.
    add_executable(test_rpc_bundle_adjust test_rpc_bundle_adjust.cpp)
    target_link_libraries(test_rpc_bundle_adjust PRIVATE zproj zproj::refine GTest::gtest)
    include(GoogleTest)
    gtest_discover_tests(test_rpc_bundle_adjust)
endif()
