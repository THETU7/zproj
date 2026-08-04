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

# RPC stereo triangulation: back-project a matched pixel pair to viewing rays
# and intersect them. Synthetic nadir + oblique pair (no data file needed):
# reports CPU/CUDA throughput, closed-loop accuracy vs known ground truth, and
# a ray-height-span accuracy sweep.
add_executable(triangulation triangulation/main.cpp)
target_link_libraries(triangulation PRIVATE zproj)
