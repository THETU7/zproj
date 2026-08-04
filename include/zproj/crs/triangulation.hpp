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
#include "ztensor/zt/Tensor.h"

namespace zproj::crs {

// Two-view RPC stereo triangulation. Internally forces the analytic inverse.
class RpcStereo {
public:
    RpcStereo(RpcInfo left,
              RpcInfo right,
              double h_low,
              double h_high,
              RpcOptions options = {});

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

private:
    RpcModel left_;  // each holds its own RpcInverseInit
    RpcModel right_;
    double h_low_;
    double h_high_;
};

}  // namespace zproj::crs
