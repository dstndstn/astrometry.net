#ifndef _LOG_H
#define _LOG_H

enum log_level {
	LOG_NONE,
	LOG_ERROR,
	LOG_MSG,
	LOG_VERB,
	LOG_ALL
};

struct log_t {
	enum log_level level;
};
typedef struct log_t log_t;

/**
 * Initialize global logging object. Must be called before any of the other
 * log_* functions.
 */
void log_init(enum log_level level);

/**
 * Create a new logger.
 *
 * Parameters:
 *   
 *   level - LOG_NONE  don't show anything
 *           LOG_ERROR only log errors
 *           LOG_MSG   log errors and important messages
 *           LOG_VERB  log verbose messages
 *           LOG_ALL   log debug messages
 *
 * Returns:
 *
 *   A new logger object
 *
 */
log_t* log_create(const enum log_level level);

/**
 * Close and free a logger object.
 */
void log_free(log_t* logger);

/**
 * Log a message
 */

#define LOG_TEMPLATE(name) \
	void name(const char* format, ...)                      \
		__attribute__ ((format (printf, 1, 2)));             \
	void name##_(const log_t* log, const char* format, ...) \
		__attribute__ ((format (printf, 2, 3)));             \

// Note that these must match corresponding templates in log.c
LOG_TEMPLATE(logmsg);
LOG_TEMPLATE(logerr);
LOG_TEMPLATE(logverb);
LOG_TEMPLATE(debug);

extern log_t _logger_global;

#endif // _LOG_H
