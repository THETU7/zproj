# GoogleTest-driven, like the vendored ztensor's own test suite: the framework
# is discovered through the system install via find_package (CONFIG) -- never
# fetched or vendored. gtest_discover_tests registers each TEST() as its own
# ctest entry.
find_package(GTest CONFIG REQUIRED)

add_executable(test_wgs84_to_ecef test_wgs84_to_ecef.cpp)
target_link_libraries(test_wgs84_to_ecef PRIVATE zproj GTest::gtest)
include(GoogleTest)
gtest_discover_tests(test_wgs84_to_ecef)

add_executable(test_ztensor test_ztensor.cpp)
target_link_libraries(test_ztensor PRIVATE zt::ztensor CUDA::cudart)
add_test(NAME zproj.ztensor COMMAND test_ztensor)
