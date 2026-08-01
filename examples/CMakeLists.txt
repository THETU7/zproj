# add_executable(wgs84_to_ecef wgs84_to_ecef/main.cpp)
# target_link_libraries(wgs84_to_ecef PRIVATE zproj GDAL::GDAL Eigen3::Eigen)

# DISABLED: eigen_gpu validated Eigen's GPU Core + Tensor modules, which
# needed the vendored master Eigen (now removed; the system Eigen has no
# GPU/Tensor backend). To be reimplemented on top of the vendored ztensor.
# add_executable(eigen_gpu eigen_gpu/eigen_gpu_demo.cu)
# target_link_libraries(eigen_gpu PRIVATE Eigen3::Eigen CUDA::cudart)
# # A CUDA-only executable links with the C compiler by default and would miss
# # libstdc++; link it with the C++ host compiler instead.
# set_property(TARGET eigen_gpu PROPERTY LINKER_LANGUAGE CXX)

# Vendored ztensor (third_party/ztensor): CPU + CUDA tensor-ops smoke.
add_executable(ztensor_basic ztensor_basic/main.cpp)
target_link_libraries(ztensor_basic PRIVATE zt::ztensor CUDA::cudart)
