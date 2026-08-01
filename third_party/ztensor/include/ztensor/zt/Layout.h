// ztensor/zt/Layout.h
//
// Memory layout of a Tensor. ztensor only supports the dense strided layout
// (mirrors PyTorch's kStrided); the enum exists for API parity with PyTorch.

#pragma once

#include <cstdint>

namespace zt {

enum class Layout : int8_t {
    Strided = 0,
    // Sparse / SparseCsr reserved for future phases.
};

inline constexpr Layout kStrided = Layout::Strided;

}  // namespace zt
