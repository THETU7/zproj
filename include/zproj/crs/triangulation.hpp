// Batch RPC stereo triangulation: the most common two-view entry point.
//
// A point observed at (col,row) in two overlapping RPC images is back-
// projected at two heights to build a viewing ray per image (rpc_ray), the
// two rays are intersected (triangulate_pair), and the ECEF result is
// converted back to geodetic (lon deg, lat deg, h m).
//
// The analytic inverse is forced internally (threshold 1e-9 px, 20
// iterations): the affine inverse's ~0.1 px error would bias the ray
// directions and accumulate in the intersection point.
#pragma once

#include <vector>

#include "zproj/crs/rpc.hpp"
#include "zproj/crs/rpc_ray_float.hpp"
#include "ztensor/zt/Tensor.h"

namespace zproj::crs {

// Precision of the per-point triangulation math. Inputs and outputs are
// double tensors either way; this selects the internal ray pipeline.
enum class StereoPrecision {
    // All-double reference path (rpc_ray.hpp): machine-precision rays,
    // FP64-throughput-bound on consumer GeForce parts (FP64 = 1/64 FP32).
    Double,
    // Float Newton inverse + scene-local ENU float rays/intersection
    // (rpc_ray_float.hpp): ~4x faster on consumer GeForce, millimetre-level
    // on scene-sized footprints (scales with footprint/GSD; continental
    // scenes degrade toward metres). Geodetic<->cartesian stays double.
    // Dateline-crossing scenes are supported: the float inverse seeds and
    // wraps the longitude exactly like the double path (+/-270 deg rule).
    // No speedup on CPU -- it exists for the GPU (see below).
    FloatEnu,
};

// Two-view RPC stereo triangulation. Internally forces the analytic inverse.
class RpcStereo {
public:
    RpcStereo(RpcInfo left,
              RpcInfo right,
              double h_low,
              double h_high,
              StereoPrecision precision = StereoPrecision::Double);

    // left_colrow / right_colrow : [N, 2] double (col, row)
    // lonlath  : [N, 3] (lon deg, lat deg, h m)   -- may be empty, allocated
    //                                                here on the input device
    // rms      : [N]   (metres)                   -- may be empty, allocated
    //                                                here on the input device
    // Failed points (non-convergent inverse, parallel rays) follow the GDAL
    // convention: HUGE_VAL in every output element.
    void triangulate(const zt::Tensor& left_colrow,
                     const zt::Tensor& right_colrow,
                     zt::Tensor& lonlath,
                     zt::Tensor& rms) const;

    StereoPrecision precision() const noexcept { return precision_; }

private:
    RpcModel left_;  // each holds its own RpcInverseInit
    RpcModel right_;
    double h_low_;
    double h_high_;
    StereoPrecision precision_;

    // Float-path precomputations (built alongside; used only for FloatEnu).
    RpcInfoFloat left_float_;
    RpcInfoFloat right_float_;
    RpcInverseInitFloat left_init_float_;
    RpcInverseInitFloat right_init_float_;
    EnuFrame enu_frame_;
};

// N-view RPC multi-stereo triangulation (V >= 2): the multi-image
// generalization of RpcStereo above, mirroring ASP's RPCStereoModel /
// VisionWorkbench's StereoModel::triangulate_point. Each view contributes a
// viewing ray (same two-height analytic back-projection as the two-view
// path); the rays are intersected in the least-squares sense by the Slabaugh
// normal equations (triangulate_nview), with the two-valid-view case
// delegating to the closed form exactly like the pair path.
//
// A view that does not observe a point marks it with a non-finite pixel
// (NaN/Inf in either component of colrow[v, k, :]) and is skipped for that
// point, following VW StereoModel's NaN-pixel filtering; at least two valid
// views per point are required. Failed points (fewer than two valid rays,
// non-convergent inverses leaving < 2 rays, parallel/singular ray bundle)
// follow the GDAL convention: HUGE_VAL in every output element.
//
// The CUDA path stages the per-view model parameters in device memory; that
// buffer holds at most 32 views, a limit the CPU path does not have.
class RpcMultiStereo {
public:
    // infos: V >= 2 RPC models, all observing the same points.
    RpcMultiStereo(std::vector<RpcInfo> infos,
                   double h_low,
                   double h_high,
                   StereoPrecision precision = StereoPrecision::Double);

    // colrow  : [V, N, 2] double -- (col, row) of view v for point k, the
    //           rows matching the constructor's model order
    // lonlath : [N, 3] (lon deg, lat deg, h m)   -- may be empty, allocated
    //                                                here on the input device
    // rms     : [N]   (metres)                   -- may be empty, allocated
    //                                                here on the input device
    void triangulate(const zt::Tensor& colrow,
                     zt::Tensor& lonlath,
                     zt::Tensor& rms) const;

    int num_views() const noexcept { return num_views_; }
    StereoPrecision precision() const noexcept { return precision_; }

private:
    std::vector<RpcInfo> infos_;  // view order == input tensor rows
    std::vector<RpcInverseInit> inits_;
    int num_views_;
    double h_low_;
    double h_high_;
    StereoPrecision precision_;

    // Float-path precomputations (built alongside; used only for FloatEnu).
    std::vector<RpcInfoFloat> infos_float_;
    std::vector<RpcInverseInitFloat> inits_float_;
    EnuFrame enu_frame_;

    // Lazily-uploaded device blobs of infos_/inits_ (kByte tensors; cached
    // for the object's lifetime so an in-flight kernel never races the
    // allocator's free, and rebuilt when inputs arrive on another device).
    // Not guarded for concurrent first calls from multiple threads.
    mutable zt::Tensor infos_dev_;
    mutable zt::Tensor inits_dev_;
    mutable zt::Tensor infos_float_dev_;
    mutable zt::Tensor inits_float_dev_;
};

}  // namespace zproj::crs
