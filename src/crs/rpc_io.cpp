#include "zproj/crs/rpc_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zproj::crs {

namespace {

std::string Trim(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

double ParseDouble(const std::string& key, const std::string& value) {
    // Values may carry a trailing unit ("+003577.86 pixels",
    // "-25.46203790 degrees", "0000.00 meters"): take the first token.
    std::istringstream iss(value);
    std::string token;
    iss >> token;
    try {
        return std::stod(token);
    } catch (const std::exception&) {
        std::string msg = "rpc: cannot parse '" + key + "' value '";
        msg += value;
        msg += "'";
        throw std::runtime_error(msg);
    }
}

std::vector<double> ParseCoeffList(const std::string& key,
                                   const std::string& value) {
    std::istringstream iss(value);
    std::vector<double> coeffs;
    std::string token;
    while (iss >> token) {
        try {
            coeffs.push_back(std::stod(token));
        } catch (const std::exception&) {
            std::string msg = "rpc: cannot parse '" + key + "' value '";
            msg += value;
            msg += "'";
            throw std::runtime_error(msg);
        }
    }
    if (coeffs.size() != static_cast<std::size_t>(kRpcCoeffCount)) {
        throw std::runtime_error("rpc: expected 20 coefficients for '" + key +
                                 "', got " + std::to_string(coeffs.size()));
    }
    return coeffs;
}

}  // namespace

RpcInfo RpcInfoFromRpcText(std::string_view text) {
    RpcInfo info;
    std::istringstream iss{std::string(text)};
    std::string line;
    while (std::getline(iss, line)) {
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = Trim(line.substr(0, colon));
        const std::string value = Trim(line.substr(colon + 1));

        // Coefficient planes arrive either as KEY_n (n = 1..20) or as a
        // single space-separated KEY list.
        auto* coeffs = [&]() -> std::array<double, 20>* {
            if (key == "LINE_NUM_COEFF") {
                return &info.line_num_coeff;
            }
            if (key == "LINE_DEN_COEFF") {
                return &info.line_den_coeff;
            }
            if (key == "SAMP_NUM_COEFF") {
                return &info.samp_num_coeff;
            }
            if (key == "SAMP_DEN_COEFF") {
                return &info.samp_den_coeff;
            }
            return nullptr;
        }();
        if (coeffs != nullptr) {
            const std::vector<double> parsed = ParseCoeffList(key, value);
            std::copy(parsed.begin(), parsed.end(), coeffs->begin());
            continue;
        }

        const std::string kLineNum = "LINE_NUM_COEFF_";
        const std::string kLineDen = "LINE_DEN_COEFF_";
        const std::string kSampNum = "SAMP_NUM_COEFF_";
        const std::string kSampDen = "SAMP_DEN_COEFF_";
        const auto assign = [&](const std::string& prefix,
                                std::array<double, 20>* plane) {
            if (key.starts_with(prefix)) {
                const int idx = std::stoi(key.substr(prefix.size())) - 1;
                if (idx < 0 || idx >= kRpcCoeffCount) {
                    throw std::runtime_error("rpc: bad coefficient index '" +
                                             key + "'");
                }
                (*plane)[static_cast<std::size_t>(idx)] =
                    ParseDouble(key, value);
                return true;
            }
            return false;
        };
        if (assign(kLineNum, &info.line_num_coeff) ||
            assign(kLineDen, &info.line_den_coeff) ||
            assign(kSampNum, &info.samp_num_coeff) ||
            assign(kSampDen, &info.samp_den_coeff)) {
            continue;
        }

        if (key == "LINE_OFF") {
            info.line_off = ParseDouble(key, value);
        } else if (key == "SAMP_OFF") {
            info.samp_off = ParseDouble(key, value);
        } else if (key == "LAT_OFF") {
            info.lat_off = ParseDouble(key, value);
        } else if (key == "LONG_OFF") {
            info.long_off = ParseDouble(key, value);
        } else if (key == "HEIGHT_OFF") {
            info.height_off = ParseDouble(key, value);
        } else if (key == "LINE_SCALE") {
            info.line_scale = ParseDouble(key, value);
        } else if (key == "SAMP_SCALE") {
            info.samp_scale = ParseDouble(key, value);
        } else if (key == "LAT_SCALE") {
            info.lat_scale = ParseDouble(key, value);
        } else if (key == "LONG_SCALE") {
            info.long_scale = ParseDouble(key, value);
        } else if (key == "HEIGHT_SCALE") {
            info.height_scale = ParseDouble(key, value);
        } else if (key == "MIN_LONG") {
            info.min_lon = ParseDouble(key, value);
        } else if (key == "MIN_LAT") {
            info.min_lat = ParseDouble(key, value);
        } else if (key == "MAX_LONG") {
            info.max_lon = ParseDouble(key, value);
        } else if (key == "MAX_LAT") {
            info.max_lat = ParseDouble(key, value);
        }
        // (all branches above already use braces)
        // ERR_BIAS / ERR_RAND and any other keys are ignored.
    }
    return info;
}

RpcInfo RpcInfoFromRpcFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("rpc: cannot open '" + path + "'");
    }
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    return RpcInfoFromRpcText(text);
}

}  // namespace zproj::crs
