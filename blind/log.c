#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>
#include "log.h"

log_t _logger_global;

void log_init_structure(log_t* logger, enum log_level level) {
	logger->level = level;
    logger->f = stdout;
}

void log_init(enum log_level level) {
	log_init_structure(&_logger_global, level);
}

void log_set_level(enum log_level level) {
    _logger_global.level = level;
}

void log_to(FILE* fid) {
	_logger_global.f = fid;
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
	vfprintf(logger->f, format, va);
	fflush(logger->f);
}

void loglevel(enum log_level level,
              const char* format, ...) {
    va_list va;
    va_start(va, format);
    loglvl(&_logger_global, level, format, va);
    va_end(va);
}

int log_get_level() {
    return _logger_global.level;
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
LOGGER_TEMPLATE(logverb, LOG_VERB);
LOGGER_TEMPLATE(debug,   LOG_ALL);
