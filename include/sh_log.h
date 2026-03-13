/*
 * sh_log.h - Structured Logging Library
 *
 * Thread-safe logging with JSON and plain text output formats.
 * Supports log levels, trace IDs, and structured context fields.
 *
 * Usage:
 *   sh_log_init(NULL);  // Use defaults
 *   SH_LOG_INFO("Request received", "method", "GET", "path", "/api/health");
 *   SH_LOG_ERROR("Database error", "code", "ECONNREFUSED", "retry", "3");
 *
 * Environment variables:
 *   SH_LOG_LEVEL  - Minimum log level (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
 *   SH_LOG_FORMAT - Output format (json, text) - default: text
 *   SH_LOG_COLOR  - Enable ANSI colors in text mode (0, 1) - default: 1 if tty
 *
 * Log output always goes to stderr (12-factor app style).
 */

#ifndef SH_LOG_H
#define SH_LOG_H

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Log Levels
 * ============================================================================ */

typedef enum {
    SH_LOG_LEVEL_TRACE = 0,
    SH_LOG_LEVEL_DEBUG = 1,
    SH_LOG_LEVEL_INFO  = 2,
    SH_LOG_LEVEL_WARN  = 3,
    SH_LOG_LEVEL_ERROR = 4,
    SH_LOG_LEVEL_FATAL = 5,
    SH_LOG_LEVEL_OFF   = 6   /* Disable all logging */
} ShLogLevel;

/* ============================================================================
 * Log Format
 * ============================================================================ */

typedef enum {
    SH_LOG_FORMAT_TEXT = 0,  /* Human-readable: [LEVEL] [time] [trace_id] msg */
    SH_LOG_FORMAT_JSON = 1   /* Machine-parseable JSON lines */
} ShLogFormat;

/* ============================================================================
 * Configuration
 * ============================================================================ */

typedef struct {
    ShLogLevel level;        /* Minimum level to log (default: INFO) */
    ShLogFormat format;      /* Output format (default: TEXT) */
    int color;               /* ANSI colors in text mode (-1=auto, 0=off, 1=on) */
    const char *service;     /* Service name for JSON output */
    const char *version;     /* Service version for JSON output */
} ShLogConfig;

/* Default configuration initializer */
#define SH_LOG_CONFIG_DEFAULT { \
    .level = SH_LOG_LEVEL_INFO, \
    .format = SH_LOG_FORMAT_TEXT, \
    .color = -1, \
    .service = NULL, \
    .version = NULL \
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

/*
 * Initialize the logging system.
 * If config is NULL, uses defaults and reads from environment variables.
 * Thread-safe, can be called multiple times (reconfigures).
 */
void sh_log_init(const ShLogConfig *config);

/*
 * Shutdown logging system and flush any buffered output.
 */
void sh_log_shutdown(void);

/*
 * Get current log level.
 */
ShLogLevel sh_log_get_level(void);

/*
 * Set log level at runtime.
 */
void sh_log_set_level(ShLogLevel level);

/*
 * Check if a log level is enabled (for expensive log message construction).
 */
int sh_log_enabled(ShLogLevel level);

/* ============================================================================
 * Core Logging Functions
 * ============================================================================ */

/*
 * Log a message with optional structured fields.
 *
 * Fields are passed as key-value pairs (both const char*), terminated by NULL.
 * Example: sh_log(SH_LOG_LEVEL_INFO, "file.c", 42, "User logged in",
 *                 "user_id", "123", "ip", "192.168.1.1", NULL);
 *
 * The trace_id is automatically included from thread-local storage if set.
 */
void sh_log(ShLogLevel level, const char *file, int line,
            const char *message, ...);

/*
 * Log with explicit trace ID (overrides thread-local).
 */
void sh_log_with_trace(ShLogLevel level, const char *file, int line,
                       const char *trace_id, const char *message, ...);

/*
 * Log with va_list for wrapper functions.
 */
void sh_log_v(ShLogLevel level, const char *file, int line,
              const char *trace_id, const char *message, va_list fields);

/* ============================================================================
 * Convenience Macros
 * ============================================================================ */

/*
 * Main logging macros. Usage:
 *   SH_LOG_INFO("message");
 *   SH_LOG_INFO("message", "key1", "val1", "key2", "val2");
 *
 * Fields are optional key-value string pairs.
 */
#define SH_LOG_TRACE(msg, ...) \
    do { if (sh_log_enabled(SH_LOG_LEVEL_TRACE)) \
        sh_log(SH_LOG_LEVEL_TRACE, __FILE__, __LINE__, msg, ##__VA_ARGS__, NULL); \
    } while(0)

#define SH_LOG_DEBUG(msg, ...) \
    do { if (sh_log_enabled(SH_LOG_LEVEL_DEBUG)) \
        sh_log(SH_LOG_LEVEL_DEBUG, __FILE__, __LINE__, msg, ##__VA_ARGS__, NULL); \
    } while(0)

#define SH_LOG_INFO(msg, ...) \
    do { if (sh_log_enabled(SH_LOG_LEVEL_INFO)) \
        sh_log(SH_LOG_LEVEL_INFO, __FILE__, __LINE__, msg, ##__VA_ARGS__, NULL); \
    } while(0)

#define SH_LOG_WARN(msg, ...) \
    do { if (sh_log_enabled(SH_LOG_LEVEL_WARN)) \
        sh_log(SH_LOG_LEVEL_WARN, __FILE__, __LINE__, msg, ##__VA_ARGS__, NULL); \
    } while(0)

#define SH_LOG_ERROR(msg, ...) \
    do { if (sh_log_enabled(SH_LOG_LEVEL_ERROR)) \
        sh_log(SH_LOG_LEVEL_ERROR, __FILE__, __LINE__, msg, ##__VA_ARGS__, NULL); \
    } while(0)

#define SH_LOG_FATAL(msg, ...) \
    do { if (sh_log_enabled(SH_LOG_LEVEL_FATAL)) \
        sh_log(SH_LOG_LEVEL_FATAL, __FILE__, __LINE__, msg, ##__VA_ARGS__, NULL); \
    } while(0)

/* ============================================================================
 * Trace ID Integration
 * ============================================================================ */

/*
 * Set thread-local trace ID. Pass NULL to clear.
 * The string is copied internally.
 */
void sh_log_set_trace_id(const char *trace_id);

/*
 * Get current thread-local trace ID (may be NULL).
 */
const char *sh_log_get_trace_id(void);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/*
 * Parse log level from string (case-insensitive).
 * Returns SH_LOG_LEVEL_INFO on invalid input.
 */
ShLogLevel sh_log_level_from_string(const char *str);

/*
 * Get string name for log level.
 */
const char *sh_log_level_to_string(ShLogLevel level);

/*
 * Parse log format from string (case-insensitive).
 * Returns SH_LOG_FORMAT_TEXT on invalid input.
 */
ShLogFormat sh_log_format_from_string(const char *str);

#ifdef __cplusplus
}
#endif

#endif /* SH_LOG_H */
