#include "rpc.h"

#include <cmath>
#include <stdexcept>

#include "ztensor/zt/ScalarType.h"
#include "ztensor/zt/TensorFactories.h"
#include "ztensor/zt/utility/Log.h"

namespace zproj::rpc {

namespace {

// Mirror GDAL's reference-point selection and affine construction from
// GDALCreateRPCTransformerV2, minus the DEM parts (height is always 0 here).
// The result maps (col, row) -> (lon, lat) and seeds the iterative inverse.
RpcInverseInit ComputeInverseInit(const RpcInfo& info) {
    double ref_lon = 0.0;
    double ref_lat = 0.0;
    double ref_col = -1.0;
    double ref_row = -1.0;

    // Prefer the centre of the validity bounds when they are known (GDAL
    // treats the default -180..180 longitude range as "unknown").
    if (info.min_lon != -180.0 || info.max_lon != 180.0) {
        ref_lon = (info.min_lon + info.max_lon) * 0.5;
        ref_lat = (info.min_lat + info.max_lat) * 0.5;
        rpc_forward_point(info, ref_lon, ref_lat, 0.0, ref_col, ref_row);
    }

    // Fall back to the offsets when the bounds are absent or produce a daft
    // reference point (GDAL's "results seem daft" check).
    if (ref_col < 0.0 || ref_row < 0.0 || ref_col > 100000.0 ||
        ref_row > 100000.0) {
        ref_lon = info.long_off;
        ref_lat = info.lat_off;
        rpc_forward_point(info, ref_lon, ref_lat, 0.0, ref_col, ref_row);
    }

    // LL -> PL affine by finite differences (GDAL's dfLLDelta = 0.0001 deg).
    constexpr double kLLDelta = 0.0001;
    double col_dx = 0.0;
    double row_dx = 0.0;
    rpc_forward_point(info, ref_lon + kLLDelta, ref_lat, 0.0, col_dx, row_dx);
    const double gt1 = (col_dx - ref_col) / kLLDelta;
    const double gt4 = (row_dx - ref_row) / kLLDelta;

    double col_dy = 0.0;
    double row_dy = 0.0;
    rpc_forward_point(info, ref_lon, ref_lat + kLLDelta, 0.0, col_dy, row_dy);
    const double gt2 = (col_dy - ref_col) / kLLDelta;
    const double gt5 = (row_dy - ref_row) / kLLDelta;

    const double gt0 = ref_col - gt1 * ref_lon - gt2 * ref_lat;
    const double gt3 = ref_row - gt4 * ref_lon - gt5 * ref_lat;

    // Invert the affine (GDALInvGeoTransform).
    const double det = gt1 * gt5 - gt2 * gt4;
    if (det == 0.0) {
        throw std::runtime_error(
            "rpc: cannot invert the LL->PL affine (zero determinant)");
    }

    RpcInverseInit init;
    init.lon_c0 = (gt2 * gt3 - gt0 * gt5) / det;
    init.lon_c1 = gt5 / det;
    init.lon_c2 = -gt2 / det;
    init.lat_c0 = (gt0 * gt4 - gt1 * gt3) / det;
    init.lat_c1 = -gt4 / det;
    init.lat_c2 = gt1 / det;
    return init;
}

// A [N, width] double tensor used as transform input/output.
void CheckPointTensor(const zt::Tensor& t, int64_t width, const char* name) {
    ZT_CHECK_EQ(t.dim(), 2);
    ZT_CHECK_EQ(t.size(1), width);
    ZT_CHECK(t.is_contiguous(), "{} must be contiguous", name);
    ZT_CHECK(t.scalar_type() == zt::ScalarType::Double,
             "{} must use double type",
             name);
}

}  // namespace

RpcModel::RpcModel(RpcInfo info, RpcOptions options)
    : info_(info), options_(options), inverse_init_(ComputeInverseInit(info_)) {
    // GDAL semantics: non-positive threshold / iterations fall back to the
    // no-DEM defaults.
    if (options_.pixel_error_threshold <= 0.0) {
        options_.pixel_error_threshold = 0.1;
    }
    if (options_.max_iterations <= 0) {
        options_.max_iterations = 10;
    }
}

void RpcModel::lonlatalt_to_colrow(const zt::Tensor& in,
                                   zt::Tensor& dst) const {
    CheckPointTensor(in, 3, "in");

    if (!dst.defined()) {
        dst = zt::empty({in.size(0), 2},
                        zt::dtype(zt::kDouble).device(in.device()));
    }
    CheckPointTensor(dst, 2, "dst");
    ZT_CHECK_LE(in.size(0), dst.size(0));

    if (in.is_cpu() && dst.is_cpu()) {
        rpc_forward_cpu(info_, in, dst);
        return;
    }

#ifdef BUILD_CUDA_MODULE
    if (in.is_cuda() && dst.is_cuda()) {
        rpc_forward_cuda(info_, in, dst);
        return;
    }
#endif  // BUILD_CUDA_MODULE

    ZT_LOG_ERROR("rpc: unsupported device (in {}, dst {})",
                 in.device().string(),
                 dst.device().string());
}

void RpcModel::colrowalt_to_lonlat(const zt::Tensor& in,
                                   zt::Tensor& dst) const {
    CheckPointTensor(in, 3, "in");

    if (!dst.defined()) {
        dst = zt::empty({in.size(0), 2},
                        zt::dtype(zt::kDouble).device(in.device()));
    }
    CheckPointTensor(dst, 2, "dst");
    ZT_CHECK_LE(in.size(0), dst.size(0));

    if (in.is_cpu() && dst.is_cpu()) {
        rpc_inverse_cpu(info_, inverse_init_, options_, in, dst);
        return;
    }

#ifdef BUILD_CUDA_MODULE
    if (in.is_cuda() && dst.is_cuda()) {
        rpc_inverse_cuda(info_, inverse_init_, options_, in, dst);
        return;
    }
#endif  // BUILD_CUDA_MODULE

    ZT_LOG_ERROR("rpc: unsupported device (in {}, dst {})",
                 in.device().string(),
                 dst.device().string());
}

}  // namespace zproj::rpc
