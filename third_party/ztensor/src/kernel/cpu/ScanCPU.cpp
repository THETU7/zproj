// ztensor/kernel/ScanCPU.cpp
//
// CPU inclusive scan along a single dimension.  Outer axes are parallelized
// via OpenMP; the scan axis is walked serially per row.

#include <cstdint>
#include <type_traits>
#include <vector>

#include "ztensor/zt/utility/Log.h"

#include "kernel/Scan.h"

namespace zt::kernel {
namespace {

template<typename T>
T scan_combine(ScanOpCode op, T acc, T v) {
    switch (op) {
        case ScanOpCode::CumSum:
            return static_cast<T>(acc + v);
        case ScanOpCode::CumProd:
            // For bool, product == logical AND. g++-11's -Wint-in-bool-context
            // flags `acc * v` for the bool instantiation, so the bool path is
            // diverted via `if constexpr`. The non-bool multiply MUST live in
            // the `else` branch: g++-11 otherwise still analyzes the statement
            // (it follows the `if`, so it is not a discarded branch) and emits
            // the warning anyway.
            if constexpr (std::is_same_v<T, bool>) {
                return acc && v;
            } else {
                return static_cast<T>(acc * v);
            }
        case ScanOpCode::CumMax:
            return v > acc ? v : acc;
        case ScanOpCode::CumMin:
            return v < acc ? v : acc;
    }
    return v;  // unreachable
}

template<typename scalar_t>
void scan_cpu_typed(const Tensor& src,
                    Tensor& dst,
                    int64_t dim,
                    ScanOpCode op) {
    const int64_t ndim = src.dim();
    const int64_t scan_len = src.size(dim);
    const int64_t num_rows = src.numel() / scan_len;
    const int64_t scan_stride = src.stride(dim);

    // Copy src → dst.
    const auto* sptr = src.data_ptr<scalar_t>();
    auto* dptr = dst.data_ptr<scalar_t>();
    const int64_t total = src.numel();
#pragma omp parallel for
    for (int64_t i = 0; i < total; ++i) {
        dptr[i] = sptr[i];
    }

    // Build strides/sizes for non-scan axes.
    std::vector<int64_t> ns_strides;
    std::vector<int64_t> ns_sizes;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            ns_strides.push_back(src.stride(d));
            ns_sizes.push_back(src.size(d));
        }
    }
    const int64_t ns_ndim = static_cast<int64_t>(ns_sizes.size());

    // For each row, walk the scan axis serially.
#pragma omp parallel for
    for (int64_t row = 0; row < num_rows; ++row) {
        // Decode row index → base offset within the scan axis.
        int64_t base = 0;
        int64_t r = row;
        for (int64_t d = ns_ndim - 1; d >= 0; --d) {
            const int64_t sz = ns_sizes[static_cast<std::size_t>(d)];
            base += (r % sz) * ns_strides[static_cast<std::size_t>(d)];
            r /= sz;
        }
        // Inclusive scan: first element is identity, subsequent accumulate.
        scalar_t acc = dptr[base];
        for (int64_t j = 1; j < scan_len; ++j) {
            const int64_t off = base + (j * scan_stride);
            acc = scan_combine(op, acc, dptr[off]);
            dptr[off] = acc;
        }
    }
}

}  // namespace

void ScanCPU(const Tensor& src, Tensor& dst, int64_t dim, ScanOpCode op) {
    const int64_t scan_len = src.size(dim);
    if (scan_len <= 0) {
        return;
    }

    // Dispatch on dtype without the DISPATCH macro so we can use #pragma omp.
    const ScalarType dt = src.scalar_type();
    if (dt == ScalarType::Float) {
        scan_cpu_typed<float>(src, dst, dim, op);
    } else if (dt == ScalarType::Double) {
        scan_cpu_typed<double>(src, dst, dim, op);
    } else if (dt == ScalarType::Int) {
        scan_cpu_typed<std::int32_t>(src, dst, dim, op);
    } else if (dt == ScalarType::Long) {
        scan_cpu_typed<std::int64_t>(src, dst, dim, op);
    } else if (dt == ScalarType::Short) {
        scan_cpu_typed<std::int16_t>(src, dst, dim, op);
    } else if (dt == ScalarType::Char) {
        scan_cpu_typed<std::int8_t>(src, dst, dim, op);
    } else if (dt == ScalarType::Byte) {
        scan_cpu_typed<std::uint8_t>(src, dst, dim, op);
    } else if (dt == ScalarType::Bool) {
        scan_cpu_typed<bool>(src, dst, dim, op);
    } else if (dt == ScalarType::Half) {
        scan_cpu_typed<Half>(src, dst, dim, op);
    } else if (dt == ScalarType::BFloat16) {
        scan_cpu_typed<BFloat16>(src, dst, dim, op);
    } else {
        ZT_LOG_ERROR("ScanCPU: unsupported dtype {}", toString(dt));
    }
}

}  // namespace zt::kernel
