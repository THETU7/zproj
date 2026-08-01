// ztensor/zt/Device.h
//
// Compute device abstraction. Phase 1 supports CPU and CUDA; the design
// mirrors PyTorch's c10::Device (a (type, index) pair with string parsing),
// trimmed to the device types ztensor cares about.

#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace zt {

// Phase-1 device types. Values are stable (0 = CPU, 1 = CUDA).
enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
    // SYCL / Metal / ROCm reserved for future phases.
};

inline constexpr DeviceType kCPU = DeviceType::CPU;
inline constexpr DeviceType kCUDA = DeviceType::CUDA;

// Index of a device within its type (-1 = "current/default device").
using DeviceIndex = int8_t;

// A compute device: a (type, index) pair.
//
// Examples:
//   zt::Device d1{"cpu"};        // CPU, any index
//   zt::Device d2{"cuda:0"};     // CUDA device 0
//   zt::Device d3{zt::kCUDA};    // CUDA, current device
class Device {
public:
    // ---- constructors ----
    constexpr Device() noexcept = default;

    // From (type, index). index == -1 means "the current/default device".
    constexpr Device(DeviceType type, DeviceIndex index = -1) noexcept
        : type_(type), index_(index) {}

    // Parse a device string such as "cpu", "cuda", "cuda:0", "CUDA:1".
    // Whitespace and case are normalized. Throws std::invalid_argument on a
    // malformed string or unknown device type.
    explicit Device(const std::string& dev_str);

    // ---- observers ----
    constexpr DeviceType type() const noexcept { return type_; }
    constexpr DeviceIndex index() const noexcept { return index_; }
    // True iff the index was explicitly set (i.e. >= 0).
    constexpr bool has_index() const noexcept { return index_ >= 0; }

    constexpr bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }
    constexpr bool is_cuda() const noexcept {
        return type_ == DeviceType::CUDA;
    }

    // A canonical string such as "cpu" or "cuda:0".
    std::string string() const;
    // A short string suitable for log messages.
    std::string str() const { return string(); }

    // Device equality. A device whose index was not pinned (index == -1,
    // meaning "the current/default device") compares equal to any device of
    // the same type — matching PyTorch, where `cuda` and `cuda:0` name the
    // same device. Two explicitly-indexed devices are equal iff their indices
    // match.
    friend constexpr bool operator==(const Device& a,
                                     const Device& b) noexcept {
        if (a.type_ != b.type_) return false;
        if (a.index_ < 0 || b.index_ < 0) return true;  // default == any
        return a.index_ == b.index_;
    }
    friend constexpr bool operator!=(const Device& a,
                                     const Device& b) noexcept {
        return !(a == b);
    }

private:
    DeviceType type_ = DeviceType::CPU;
    DeviceIndex index_ = -1;
};

// Validate that the current build supports `dev`'s type. Throws
// std::runtime_error if CUDA is requested while the CUDA backend is disabled.
void check_device_supported(const Device& dev);

}  // namespace zt

// std::hash specialization so Device can key associative containers.
namespace std {
template<>
struct hash<zt::Device> {
    std::size_t operator()(const zt::Device& d) const noexcept {
        return std::hash<int>()((static_cast<int>(d.type()) << 8) |
                                (d.index() & 0xff));
    }
};
}  // namespace std
