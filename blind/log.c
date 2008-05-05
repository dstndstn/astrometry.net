#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>
#include "log.h"

log_t _logger_global;

void log_init_structure(log_t* logger, enum log_level level) {
	logger->level = level;
}

void log_init(enum log_level level) {
	log_init_structure(&_logger_global, level);
}

log_t* log_create(enum log_level level) {
	log_t* logger = calloc(1, sizeof(log_t));
	return logger;
}

void log_free(log_t* log) {
	assert(log);
	free(log);
}

static void loglvl(const log_t* logger, enum log_level level,
                   const char* format, va_list va) {
	// FIXME: add pthread synchronization
	if (level > logger->level)
		return;
	vfprintf(stderr, format, va);
	fflush(stderr);
}


#define LOGGER_TEMPLATE(name, level)                  \
	void                                                \
	name##_(const log_t* logger, const char* format, ...) { \
		va_list va;                                       \
		va_start(va, format);                             \
		loglvl(logger, level, format, va);                \
		va_end(va);                                       \
	}                                                   \
	void                                                \
	name(const char* format, ...) {                     \
		va_list va;                                       \
		va_start(va, format);                             \
		loglvl(&_logger_global, level, format, va);       \
		va_end(va);                                       \
	}                                                   \

LOGGER_TEMPLATE(logerr,  LOG_ERROR);
LOGGER_TEMPLATE(logmsg,  LOG_MSG);
LOGGER_TEMPLATE(logverb, LOG_ALL);
LOGGER_TEMPLATE(debug,   LOG_ALL);
