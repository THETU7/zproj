// ztensor/core/Device.cpp

#include "ztensor/zt/Device.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ztensor/zt/utility/Log.h"

namespace zt {

namespace {

// Lowercase + strip leading/trailing whitespace.
std::string normalize(std::string s) {
    auto to_lower = [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    };
    std::transform(s.begin(), s.end(), s.begin(), to_lower);
    const auto first = s.find_first_not_of(" \t\r\n");
    const auto last = s.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return s.substr(first, last - first + 1);
}

}  // namespace

Device::Device(const std::string& dev_str) {
    const std::string s = normalize(dev_str);
    if (s.empty()) {
        throw std::invalid_argument("Device: empty device string");
    }

    // Split "type[:index]".
    const auto colon = s.find(':');
    std::string type_part =
        (colon == std::string::npos) ? s : s.substr(0, colon);

    if (type_part == "cpu") {
        type_ = DeviceType::CPU;
    } else if (type_part == "cuda") {
        type_ = DeviceType::CUDA;
    } else {
        throw std::invalid_argument("Device: unknown device type '" +
                                    type_part + "'");
    }

    if (colon != std::string::npos) {
        const std::string idx_part = s.substr(colon + 1);
        try {
            const long idx = std::stol(idx_part);
            if (idx < 0) {
                throw std::invalid_argument("Device: negative index '" +
                                            idx_part + "'");
            }
            index_ = static_cast<DeviceIndex>(idx);
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument("Device: bad index '" + idx_part + "'");
        } catch (const std::out_of_range&) {
            throw std::invalid_argument("Device: index out of range '" +
                                        idx_part + "'");
        }
    } else {
        index_ = -1;  // current/default device
    }
}

std::string Device::string() const {
    std::ostringstream oss;
    switch (type_) {
        case DeviceType::CPU:
            oss << "cpu";
            break;
        case DeviceType::CUDA:
            oss << "cuda";
            break;
    }
    if (index_ >= 0) oss << ":" << static_cast<int>(index_);
    return oss.str();
}

void check_device_supported(const Device& dev) {
#ifndef BUILD_CUDA_MODULE
    if (dev.is_cuda()) {
        // Defer to Logger so the throw site is consistent.
        ZT_LOG_ERROR(
            "CUDA device '{}' requested, but the CUDA backend was not built. "
            "Reconfigure with -DBUILD_CUDA_MODULE=ON.",
            dev.string());
    }
#else
    (void)dev;
#endif
}

}  // namespace zt
