// ztensor/zt/MemoryFormat.h
//
// How a freshly-allocated / contiguous tensor should be laid out. ztensor
// currently only supports plain row-major (Contiguous); the enum mirrors
// PyTorch's MemoryFormat for API parity.

#pragma once

#include <cstdint>

namespace zt {

enum class MemoryFormat : int8_t {
    Contiguous = 0,
    // Preserve / ChannelsLast reserved for future phases.
};

inline constexpr MemoryFormat kContiguous = MemoryFormat::Contiguous;

}  // namespace zt
