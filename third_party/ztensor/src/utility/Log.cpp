// ztensor/zt/utility/Log.cpp — implementation of zt::Logger.

#include "ztensor/zt/utility/Log.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace zt {

namespace {

// Map zt::LogLevel (declared fmt-free in Log.h) to/from spdlog's level enum.
// spdlog core is confined to this TU, so these are the only places the two
// enumerations meet.
spdlog::level::level_enum to_spdlog(LogLevel l) {
    switch (l) {
        case LogLevel::Trace:
            return spdlog::level::trace;
        case LogLevel::Debug:
            return spdlog::level::debug;
        case LogLevel::Info:
            return spdlog::level::info;
        case LogLevel::Warn:
            return spdlog::level::warn;
        case LogLevel::Error:
            return spdlog::level::err;
        case LogLevel::Critical:
            return spdlog::level::critical;
        case LogLevel::Off:
            return spdlog::level::off;
    }
    return spdlog::level::info;
}

LogLevel from_spdlog(spdlog::level::level_enum l) {
    switch (l) {
        case spdlog::level::trace:
            return LogLevel::Trace;
        case spdlog::level::debug:
            return LogLevel::Debug;
        case spdlog::level::info:
            return LogLevel::Info;
        case spdlog::level::warn:
            return LogLevel::Warn;
        case spdlog::level::err:
            return LogLevel::Error;
        case spdlog::level::critical:
            return LogLevel::Critical;
        case spdlog::level::off:
            return LogLevel::Off;
        default:
            break;  // n_levels / any future sentinel
    }
    return LogLevel::Info;
}

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

void Logger::SetLevel(LogLevel level) {
    default_logger().set_level(to_spdlog(level));
}

LogLevel Logger::GetLevel() { return from_spdlog(default_logger().level()); }

void Logger::log_(LogLevel lvl,
                  const char* file,
                  int line,
                  const char* fn,
                  fmt::string_view fmt_str,
                  fmt::format_args args) {
    auto& logger = default_logger();
    const auto slevel = to_spdlog(lvl);
    if (!logger.should_log(slevel)) {
        return;
    }
    const std::string body = fmt::vformat(fmt_str, args);
    spdlog::source_loc loc{file, line, fn};
    logger.log(loc, slevel, body);
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
