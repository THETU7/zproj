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

# Vendored ztensor (third_party/ztensor): CPU + GPU tensor-ops smoke.
add_executable(ztensor_basic ztensor_basic/main.cpp)
target_link_libraries(ztensor_basic PRIVATE zt::ztensor ${_zproj_gpu_runtime})

# RPC stereo triangulation: back-project a matched pixel pair to viewing rays
# and intersect them. Synthetic nadir + oblique pair (no data file needed):
# reports CPU/CUDA throughput, closed-loop accuracy vs known ground truth, and
# a ray-height-span accuracy sweep.
add_executable(triangulation triangulation/main.cpp)
target_link_libraries(triangulation PRIVATE zproj)

# Multi-view (N >= 2) RPC triangulation: least-squares intersection of N
# viewing rays (RpcMultiStereo, the Slabaugh normal equations). Synthetic
# nadir + rotating-azimuth oblique views (no data file needed): reports
# CPU/CUDA throughput for both precision paths, closed-loop accuracy vs known
# ground truth, and CUDA-vs-CPU agreement.
add_executable(nview_triangulation nview_triangulation/main.cpp)
target_link_libraries(nview_triangulation PRIVATE zproj)

# RPC affine bias-correction demo (Ceres): recover a known affine corruption
# of a synthetic stereo pair from noisy matches with outliers; reports the
# recovered vs true affines, before/after reprojection RMS, and the ground
# accuracy of corrected vs raw triangulation. Built only with
# ZPROJ_BUILD_RPC_REFINE=ON.
if(TARGET zproj::refine)
    add_executable(rpc_affine rpc_affine/main.cpp)
    target_link_libraries(rpc_affine PRIVATE zproj zproj::refine)

    # Multi-view (N >= 2) RPC bundle adjustment demo (Ceres): recover known
    # per-view affine corruptions of a synthetic multi-view network from
    # noisy tie points with outliers and partial visibility, then show the
    # ground accuracy of N-view triangulation before/after the correction.
    add_executable(rpc_bundle_adjust rpc_bundle_adjust/main.cpp)
    target_link_libraries(rpc_bundle_adjust PRIVATE zproj zproj::refine)

    # Banded bundle adjustment demo (Ceres): the wave-absorbing camera model
    # (one global affine per scene + tent-basis band shifts) recovering a
    # known affine + single-period row-wave corruption; reports the
    # residual-vs-row profile (the W shape) flattening and the ground
    # accuracy of corrected vs raw triangulation.
    add_executable(rpc_bundle_adjust_banded rpc_bundle_adjust_banded/main.cpp)
    target_link_libraries(rpc_bundle_adjust_banded PRIVATE zproj zproj::refine)
endif()
