// ztensor/zt/utility/Log.h
//
// ztensor's logging facade. Built on spdlog (compiled static library), but
// wrapped behind a small Logger class + macros so call sites are insulated
// from spdlog and gain compile-time format-string checking via fmt.
//
// Modeled on Open3D's open3d/utility/Logging.h.
//
// Usage:
//   zt::Logger::Init();                 // optional; called lazily otherwise
//   ZT_LOG_INFO("loaded {} tensors", n);
//   ZT_CHECK(shape.size() == 2, "expected 2-D, got {}-D", shape.size());
//   ZT_LOG_ERROR("device {} not available", dev.string());  // throws

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

// A portable __PRETTY_FUNCTION__-like name for log provenance.
#if defined(__GNUC__) || defined(__clang__)
#define ZT_FUNCTION __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define ZT_FUNCTION __FUNCSIG__
#else
#define ZT_FUNCTION __func__
#endif

namespace zt {

// Centralized logger. Holds the single spdlog logger used by all ZT_LOG_*
// macros. Thread-safe after Init().
class Logger {
public:
    // Initialize the default logger. Idempotent. Sets a sensible pattern and
    // a default level that depends on NDEBUG (debug -> debug, release -> warn).
    static void Init();

    // Replace the default level at runtime.
    static void SetLevel(spdlog::level::level_enum level);
    static spdlog::level::level_enum GetLevel();

    // ---- formatted log helpers ----
    // Each takes a compile-time-checked fmt format string. ZT_LOG_ERROR
    // constructs a std::runtime_error with the formatted message and throws;
    // it never returns.
    template<typename... Args>
    static void LogInfo_(const char* file,
                         int line,
                         const char* fn,
                         fmt::format_string<Args...> fmt_str,
                         Args&&... args) {
        log_(spdlog::level::info,
             file,
             line,
             fn,
             fmt_str,
             fmt::make_format_args(args...));
    }
    template<typename... Args>
    static void LogWarning_(const char* file,
                            int line,
                            const char* fn,
                            fmt::format_string<Args...> fmt_str,
                            Args&&... args) {
        log_(spdlog::level::warn,
             file,
             line,
             fn,
             fmt_str,
             fmt::make_format_args(args...));
    }
    template<typename... Args>
    [[noreturn]] static void LogError_(const char* file,
                                       int line,
                                       const char* fn,
                                       fmt::format_string<Args...> fmt_str,
                                       Args&&... args) {
        const std::string msg = format_msg_(
            file, line, fn, fmt_str, fmt::make_format_args(args...));
        throw std::runtime_error(msg);
    }

private:
    // Lazily ensures the logger is initialized; called from every log path.
    static void ensure_initialized_();

    static void log_(spdlog::level::level_enum lvl,
                     const char* file,
                     int line,
                     const char* fn,
                     fmt::string_view fmt_str,
                     fmt::format_args args);

    static std::string format_msg_(const char* file,
                                   int line,
                                   const char* fn,
                                   fmt::string_view fmt_str,
                                   fmt::format_args args);

    Logger() = delete;  // static-only utility.
};

// Prepend a compact basename to a log line.
std::string short_basename(std::string_view file);

}  // namespace zt

// ---------------------------------------------------------------------------
// Public macros. Use these from call sites; never call Logger methods directly.
// ---------------------------------------------------------------------------

#define ZT_LOG_INFO(...)                                                      \
    do {                                                                      \
        ::zt::Logger::LogInfo_(__FILE__, __LINE__, ZT_FUNCTION, __VA_ARGS__); \
    } while (0)

#define ZT_LOG_WARNING(...)                                \
    do {                                                   \
        ::zt::Logger::LogWarning_(                         \
            __FILE__, __LINE__, ZT_FUNCTION, __VA_ARGS__); \
    } while (0)

// ZT_LOG_ERROR formats the message and throws std::runtime_error.
#define ZT_LOG_ERROR(...)                                                      \
    do {                                                                       \
        ::zt::Logger::LogError_(__FILE__, __LINE__, ZT_FUNCTION, __VA_ARGS__); \
    } while (0)

// Condition + message. Throws when `cond` is false.
#define ZT_CHECK(cond, ...)                                    \
    do {                                                       \
        if (!(cond)) {                                         \
            ::zt::Logger::LogError_(                           \
                __FILE__, __LINE__, ZT_FUNCTION, __VA_ARGS__); \
        }                                                      \
    } while (0)

// Comparison shorthands. They log both operands on failure.
#define ZT_CHECK_OP(lhs, rhs, op, fmt_spec)                           \
    do {                                                              \
        auto _zt_a = (lhs);                                           \
        auto _zt_b = (rhs);                                           \
        if (!(_zt_a op _zt_b)) {                                      \
            ::zt::Logger::LogError_(__FILE__,                         \
                                    __LINE__,                         \
                                    ZT_FUNCTION,                      \
                                    "check failed: " fmt_spec " " #op \
                                    " " fmt_spec " (lhs={}, rhs={})", \
                                    _zt_a,                            \
                                    _zt_b,                            \
                                    _zt_a,                            \
                                    _zt_b);                           \
        }                                                             \
    } while (0)

#define ZT_CHECK_EQ(a, b) ZT_CHECK_OP(a, b, ==, "{}")
#define ZT_CHECK_NE(a, b) ZT_CHECK_OP(a, b, !=, "{}")
#define ZT_CHECK_LT(a, b) ZT_CHECK_OP(a, b, <, "{}")
#define ZT_CHECK_LE(a, b) ZT_CHECK_OP(a, b, <=, "{}")
#define ZT_CHECK_GT(a, b) ZT_CHECK_OP(a, b, >, "{}")
#define ZT_CHECK_GE(a, b) ZT_CHECK_OP(a, b, >=, "{}")
