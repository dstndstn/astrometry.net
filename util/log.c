#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "log.h"
#include "an-thread.h"

static int g_thread_specific = 0;
static log_t g_logger;

void log_set_thread_specific() {
    g_thread_specific = 1;
}

static void* logts_init_key(void* user) {
    log_t* l = malloc(sizeof(log_t));
    if (user)
        memcpy(l, user, sizeof(log_t));
    return l;
}
#define TSNAME logts
#include "thread-specific.inc"

static log_t* get_logger() {
    if (g_thread_specific)
        return logts_get_key(&g_logger);
    return &g_logger;
}



void log_init_structure(log_t* logger, enum log_level level) {
	logger->level = level;
    logger->f = stdout;
}

void log_init(enum log_level level) {
	log_init_structure(get_logger(), level);
}

void log_set_level(enum log_level level) {
    get_logger()->level = level;
}

void log_to(FILE* fid) {
	get_logger()->f = fid;
}

void log_to_fd(int fd) {
    // MEMLEAK
    FILE* fid = fdopen(fd, "a");
    log_to(fid);
}

log_t* log_create(enum log_level level) {
	log_t* logger = calloc(1, sizeof(log_t));
	return logger;
}

void log_free(log_t* log) {
	assert(log);
	free(log);
}

AN_THREAD_DECLARE_STATIC_MUTEX(loglock);

static void loglvl(const log_t* logger, enum log_level level,
                   const char* format, va_list va) {
	if (level > logger->level)
		return;
	AN_THREAD_LOCK(loglock);
	vfprintf(logger->f, format, va);
	fflush(logger->f);
	AN_THREAD_UNLOCK(loglock);
}

void loglevel(enum log_level level,
              const char* format, ...) {
    va_list va;
    va_start(va, format);
    loglvl(get_logger(), level, format, va);
    va_end(va);
}

int log_get_level() {
    return get_logger()->level;
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
		loglvl(get_logger(), level, format, va);       \
		va_end(va);                                       \
	}                                                   \

LOGGER_TEMPLATE(logerr,  LOG_ERROR);
LOGGER_TEMPLATE(logmsg,  LOG_MSG);
LOGGER_TEMPLATE(logverb, LOG_VERB);
LOGGER_TEMPLATE(debug,   LOG_ALL);
