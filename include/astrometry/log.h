/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef _LOG_H
#define _LOG_H

#include <stdio.h>
#include <stdarg.h>
#include "astrometry/an-bool.h"

enum log_level {
    LOG_NONE,
    LOG_ERROR,
    LOG_MSG,
    LOG_VERB,
    LOG_ALL
};

typedef void (*logfunc_t)(void* baton, enum log_level, const char* file, int line, const char* func, const char* format, va_list va);
//typedef void (*logfunc2_t)(void* baton, enum log_level, const char* file, int line, const char* string);

struct log_t {
    enum log_level level;
    FILE* f;
    anbool timestamp;
    double t0;
    // User-specified logging functions
    logfunc_t logfunc;
    void* baton;
};
typedef struct log_t log_t;

/**
 Send log messages to the user-specified logging function (as well as
 to the FILE*, if it is set; use log_to(NULL) to only send messages to
 the user-specified function.
 */
void log_use_function(logfunc_t func, void* baton);

/**
 Make all logging commands thread-specific rather than global.
 */
void log_set_thread_specific(void);

void log_set_timestamp(anbool b);

/**
 * Initialize global logging object. Must be called before any of the other
 * log_* functions.
 */
void log_init(enum log_level level);

void log_set_level(enum log_level level);

/**
 Sends global logging to the given FILE*.
 */
void log_to(FILE* fid);

void log_to_fd(int fd);

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

#define LOG_TEMPLATE(name)                                              \
    void log_##name(const char* file, int line, const char* func, const char* format, ...) \
         __attribute__ ((format (printf, 4, 5)));
LOG_TEMPLATE(logmsg);
LOG_TEMPLATE(logerr);
LOG_TEMPLATE(logverb);
LOG_TEMPLATE(logdebug);

void log_loglevel(enum log_level level, const char* file, int line, const char* func, const char* format, ...)
    __attribute__ ((format (printf, 5, 6)));


/**
 * Log a message:
 */

#define logerr(  x, ...) log_logerr(  __FILE__, __LINE__, __func__, x, ##__VA_ARGS__)
#define logmsg(  x, ...) log_logmsg(  __FILE__, __LINE__, __func__, x, ##__VA_ARGS__)
#define logverb( x, ...) log_logverb( __FILE__, __LINE__, __func__, x, ##__VA_ARGS__)
#define debug(   x, ...) log_logdebug(__FILE__, __LINE__, __func__, x, ##__VA_ARGS__)
#define logdebug(x, ...) log_logdebug(__FILE__, __LINE__, __func__, x, ##__VA_ARGS__)

// log at a particular level.
#define loglevel(loglvl, format, ...) log_loglevel(loglvl, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)

int log_get_level(void);

FILE* log_get_fid(void);

extern log_t _logger_global;

#endif // _LOG_H
