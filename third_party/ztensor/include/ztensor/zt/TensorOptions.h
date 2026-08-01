// ztensor/zt/TensorOptions.h
//
// Builder for tensor creation options: dtype, device, layout. Modeled on
// PyTorch's c10::TensorOptions, trimmed to ztensor's supported subset.
//
// Usage:
//   zt::TensorOptions o =
//   zt::TensorOptions().dtype(zt::kFloat).device("cuda:0"); zt::empty({2, 3},
//   o);

#pragma once

#include <type_traits>

#include "ztensor/zt/Device.h"
#include "ztensor/zt/Layout.h"
#include "ztensor/zt/MemoryFormat.h"
#include "ztensor/zt/ScalarType.h"

namespace zt {

class TensorOptions {
public:
    constexpr TensorOptions() noexcept = default;

    // ---- setters (return a new TensorOptions, mirroring PyTorch) ----
    constexpr TensorOptions dtype(ScalarType dt) const noexcept {
        TensorOptions o = *this;
        o.dtype_ = dt;
        o.has_dtype_ = true;
        return o;
    }
    template<
        typename T,
        typename = typename std::enable_if<std::is_arithmetic<T>::value ||
                                           std::is_same<T, bool>::value>::type>
    constexpr TensorOptions dtype() const noexcept {
        return dtype(CppTypeToScalarType<T>::value);
    }

    constexpr TensorOptions device(Device dev) const noexcept {
        TensorOptions o = *this;
        o.device_ = dev;
        o.has_device_ = true;
        return o;
    }
    // Convenience: parse a device string inline.
    TensorOptions device(const std::string& dev_str) const {
        return device(Device(dev_str));
    }

    constexpr TensorOptions layout(Layout l) const noexcept {
        TensorOptions o = *this;
        o.layout_ = l;
        o.has_layout_ = true;
        return o;
    }

    constexpr TensorOptions memory_format(MemoryFormat mf) const noexcept {
        TensorOptions o = *this;
        o.memory_format_ = mf;
        return o;
    }

    // ---- getters ----
    constexpr ScalarType dtype() const noexcept { return dtype_; }
    constexpr Device device() const noexcept { return device_; }
    constexpr Layout layout() const noexcept { return layout_; }
    constexpr MemoryFormat memory_format() const noexcept {
        return memory_format_;
    }

    constexpr bool has_dtype() const noexcept { return has_dtype_; }
    constexpr bool has_device() const noexcept { return has_device_; }
    constexpr bool has_layout() const noexcept { return has_layout_; }

    // Merge `other` into `*this`: explicit fields of `other` win, defaults
    // of `other` are ignored. This is what tensor factories use to fold a
    // user-supplied options object into the default.
    constexpr TensorOptions merge(const TensorOptions& other) const noexcept {
        TensorOptions o = *this;
        if (other.has_dtype_) {
            o.dtype_ = other.dtype_;
            o.has_dtype_ = true;
        }
        if (other.has_device_) {
            o.device_ = other.device_;
            o.has_device_ = true;
        }
        if (other.has_layout_) {
            o.layout_ = other.layout_;
            o.has_layout_ = true;
        }
        o.memory_format_ = other.memory_format_;
        return o;
    }

private:
    Device device_ = Device(kCPU);
    ScalarType dtype_ = ScalarType::Float;
    Layout layout_ = Layout::Strided;
    MemoryFormat memory_format_ = MemoryFormat::Contiguous;
    bool has_dtype_ = false;
    bool has_device_ = false;
    bool has_layout_ = false;
};

// Free-function builders (PyTorch parity).
inline constexpr TensorOptions dtype(ScalarType dt) {
    return TensorOptions().dtype(dt);
}
template<typename T>
inline constexpr TensorOptions dtype() {
    return TensorOptions().dtype<T>();
}
inline TensorOptions device(Device dev) { return TensorOptions().device(dev); }
inline TensorOptions device(const std::string& dev_str) {
    return TensorOptions().device(dev_str);
}
inline constexpr TensorOptions layout(Layout l) {
    return TensorOptions().layout(l);
}
inline constexpr TensorOptions memory_format(MemoryFormat mf) {
    return TensorOptions().memory_format(mf);
}

}  // namespace zt
