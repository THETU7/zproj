// ztensor/core/TensorKey.cpp
//
// Out-of-line definitions of the Tensor-holding parts of TensorKey. These
// need the complete zt::Tensor type, so they live in a .cpp (not the header,
// which only forward-declares Tensor to break the include cycle with
// Tensor.h).

#include "ztensor/zt/TensorKey.h"

#include <utility>

#include "ztensor/zt/Tensor.h"

namespace zt {

TensorKey::TensorKey(const zt::Tensor& t)
    : impl_(std::make_shared<zt::Tensor>(t)) {}

TensorKey TensorKey::Tensor(const zt::Tensor& idx) { return TensorKey(idx); }

const zt::Tensor& TensorKey::GetTensor() const {
    ZT_CHECK(IsTensor(), "TensorKey::GetTensor: key is not a Tensor");
    return *std::get<std::shared_ptr<zt::Tensor>>(impl_);
}

}  // namespace zt
