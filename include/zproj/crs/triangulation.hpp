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

}  // namespace zproj::crs
