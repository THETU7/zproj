// ztensor/zt/utility/Log.cpp — implementation of zt::Logger.

#include "ztensor/zt/utility/Log.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <spdlog/sinks/stdout_color_sinks.h>

namespace zt {

namespace {

// Owns the shared logger; created on first use so that Logger::Init() and
// the log macros are safe to call before main() if needed.
spdlog::logger& default_logger() {
    static auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    static auto logger = std::make_shared<spdlog::logger>("zt", sink);
    static bool initialized = [] {
        logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
#ifdef NDEBUG
        logger->set_level(spdlog::level::warn);
#else
        logger->set_level(spdlog::level::debug);
#endif
        return true;
    }();
    (void)initialized;
    return *logger;
}

}  // namespace

std::string short_basename(std::string_view file) {
    // Find the last path separator.
    const auto pos = file.find_last_of("/\\");
    return std::string(pos == std::string_view::npos ? file
                                                     : file.substr(pos + 1));
}

void Logger::Init() { ensure_initialized_(); }

void Logger::ensure_initialized_() {
    // Touching the static `default_logger()` triggers one-time init.
    (void)default_logger();
}

void Logger::SetLevel(spdlog::level::level_enum level) {
    default_logger().set_level(level);
}

spdlog::level::level_enum Logger::GetLevel() {
    return default_logger().level();
}

void Logger::log_(spdlog::level::level_enum lvl,
                  const char* file,
                  int line,
                  const char* fn,
                  fmt::string_view fmt_str,
                  fmt::format_args args) {
    auto& logger = default_logger();
    if (!logger.should_log(lvl)) return;
    const std::string body = fmt::vformat(fmt_str, args);
    spdlog::source_loc loc{file, line, fn};
    logger.log(loc, lvl, body);
}

std::string Logger::format_msg_(const char* file,
                                int line,
                                const char* fn,
                                fmt::string_view fmt_str,
                                fmt::format_args args) {
    const std::string body = fmt::vformat(fmt_str, args);
    return fmt::format("[{}:{}] {}: {}", short_basename(file), line, fn, body);
}

}  // namespace zt
