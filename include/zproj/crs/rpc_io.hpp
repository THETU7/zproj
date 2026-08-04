// RPC00 metadata text/file parsing (the standard interchange format used by
// DigitalGlobe RPB, GeoEye RPC, EROS .rpc, OrbView, ... files -- the same
// input GDAL's GDALLoadRPCFile() reads).
//
// The format is a list of "KEY: value" lines. Scalars (LINE_OFF, LONG_SCALE,
// ...) may carry a trailing unit ("+003577.86 pixels", "-25.46203790
// degrees"). The 20 coefficients of each rational polynomial plane come
// either as numbered keys (LINE_NUM_COEFF_1 .. LINE_NUM_COEFF_20) or as a
// single space-separated list under the bare key. Bounds (MIN_LONG ...) and
// ERR_BIAS/ERR_RAND are optional.
#pragma once

#include <string>
#include <string_view>

#include "zproj/crs/rpc.hpp"

namespace zproj::crs {

// Parse RPC metadata text into RpcInfo. Throws std::runtime_error on missing
// required fields or malformed numbers.
RpcInfo RpcInfoFromRpcText(std::string_view text);

// Read an RPC metadata file (e.g. *.rpc / *_rpc.txt) and parse it.
// Throws std::runtime_error on I/O or parse failures.
RpcInfo RpcInfoFromRpcFile(const std::string& path);

}  // namespace zproj::crs
