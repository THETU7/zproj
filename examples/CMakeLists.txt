# WGS84 geodetic -> ECEF speed test: zproj CPU/CUDA vs the GDAL/PROJ reference.
# OpenMP is linked so the benchmark can toggle the CPU loop's thread count
# (single-threaded vs parallel) and report it in the header.
add_executable(wgs84_to_ecef wgs84_to_ecef/main.cpp)
target_link_libraries(wgs84_to_ecef
    PRIVATE
        zproj
        GDAL::GDAL
        Eigen3::Eigen
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# DISABLED: eigen_gpu validated Eigen's GPU Core + Tensor modules, which
# needed the vendored master Eigen (now removed; the system Eigen has no
# GPU/Tensor backend). To be reimplemented on top of the vendored ztensor.
# add_executable(eigen_gpu eigen_gpu/eigen_gpu_demo.cu)
# target_link_libraries(eigen_gpu PRIVATE Eigen3::Eigen CUDA::cudart)
# # A CUDA-only executable links with the C compiler by default and would miss
# # libstdc++; link it with the C++ host compiler instead.
# set_property(TARGET eigen_gpu PROPERTY LINKER_LANGUAGE CXX)

# RPC sensor model: zproj CPU/CUDA vs GDAL reference (forward + inverse) on a
# real satellite RPC file (examples/rpc/data/md_ge_rgb_0010000_rpc.txt). The
# data path is baked in so the binary works from any cwd.
add_executable(rpc rpc/main.cpp)
target_link_libraries(rpc PRIVATE zproj GDAL::GDAL)
target_compile_definitions(rpc PRIVATE
    ZPROJ_RPC_DATA="${CMAKE_CURRENT_SOURCE_DIR}/rpc/data/md_ge_rgb_0010000_rpc.txt")

# Vendored ztensor (third_party/ztensor): CPU + CUDA tensor-ops smoke.
add_executable(ztensor_basic ztensor_basic/main.cpp)
target_link_libraries(ztensor_basic PRIVATE zt::ztensor CUDA::cudart)
