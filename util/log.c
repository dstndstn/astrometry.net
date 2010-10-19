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
	logger->logfunc = NULL;
	logger->baton = NULL;
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

void log_use_function(logfunc_t func, void* baton) {
	log_t* l = get_logger();
	l->logfunc = func;
	l->baton = baton;
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
				   const char* file, int line,
                   const char* format, va_list va) {
	if (level > logger->level)
		return;
	AN_THREAD_LOCK(loglock);
	if (logger->f) {
		//fprintf(logger->f, "%s:%i ", file, line);
		vfprintf(logger->f, format, va);
		fflush(logger->f);
	}
	if (logger->logfunc) {
		logger->logfunc(logger->baton, level, file, line, format, va);
	}
	AN_THREAD_UNLOCK(loglock);
}

void log_loglevel(enum log_level level,
				  const char* file, int line,
				  const char* format, ...) {
    va_list va;
    va_start(va, format);
    loglvl(get_logger(), level, file, line, format, va);
    va_end(va);
}

int log_get_level() {
    return get_logger()->level;
}

FILE* log_get_fid() {
	return get_logger()->f;
}

#define LOGGER_TEMPLATE(name, level)									\
	void																\
	name##_(const log_t* logger, const char* file, int line, const char* format, ...) {	\
		va_list va;														\
		va_start(va, format);											\
		loglvl(logger, level, file, line, format, va);					\
		va_end(va);														\
	}																	\
	void																\
	name(const char* file, int line, const char* format, ...) {			\
		va_list va;														\
		va_start(va, format);											\
		loglvl(get_logger(), level, file, line, format, va);			\
		va_end(va);														\
	}																	\
	
LOGGER_TEMPLATE(log_logerr,  LOG_ERROR);
LOGGER_TEMPLATE(log_logmsg,  LOG_MSG);
LOGGER_TEMPLATE(log_logverb, LOG_VERB);
LOGGER_TEMPLATE(log_logdebug,LOG_ALL);



