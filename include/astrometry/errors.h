/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef AN_ERRORS_H
#define AN_ERRORS_H

#include <stdarg.h>
#include <stdio.h>
#include <sys/types.h>
#include <regex.h>

#include "astrometry/an-bool.h"
#include "astrometry/bl.h"
#include "astrometry/keywords.h"

// forward declaration
struct errors;
typedef struct errors err_t;

typedef void (errfunc_t)(void* baton, err_t* errstate, const char* file, int line, const char* func, const char* format, va_list va);

struct errentry {
    char* file;
    int line;
    char* func;
    char* str;
};
typedef struct errentry errentry_t;

struct errors {
    FILE* print;
    anbool save;
    bl* errstack;

    errfunc_t* errfunc;
    void* baton;
};

/***    Global functions    ***/

err_t* errors_get_state();

// takes a (deep) snapshot of the current error handling state and pushes it onto the
// stack.
void errors_push_state();

// 
void errors_pop_state();

void
ATTRIB_FORMAT(printf,4,5)
    report_error(const char* modfile, int modline, const char* modfunc, const char* fmt, ...);

void report_errno();

#define ERROR(fmt, ...) report_error(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define SYSERROR(fmt, ...) do { report_errno(); report_error(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__); } while(0)

void errors_log_to(FILE* f);

/* Sends all errors to the given function for processing;
 turns off printing and saving (ie, err_t.print and err_t.save)
 */
void errors_use_function(errfunc_t* func, void* baton);

void errors_print_stack(FILE* f);

void errors_clear_stack();

int errors_print_on_exit(FILE* fid);

// free globals.
void errors_free();

/*
 A convenience routine for times when you want to suppress printing error
 messages and instead capture them to a string.  Use in conjunction with
 the following...
 */
void errors_start_logging_to_string();

/*
 Reverts the error-processing system to its previous state and returns the
 captured error string.
 Returns a newly-allocated string which you must free().
 */
char* errors_stop_logging_to_string(const char* separator);

/*
 Convenience function to report an error from the regex module.
 */
void errors_regex_error(int errcode, const regex_t* re);

/***    End globals   ***/


err_t* error_new();

void error_free(err_t* e);

void error_stack_add_entryv(err_t* e, const char* file, int line, const char* func, const char* format, va_list va);

void error_stack_add_entry(err_t* e, const char* file, int line, const char* func, const char* str);

errentry_t* error_stack_get_entry(const err_t* e, int i);

int error_stack_N_entries(const err_t* e);

int error_nerrs(const err_t* e);

char* error_get_errstr(const err_t* e, int i);

void error_stack_clear(err_t* e);

void
ATTRIB_FORMAT(printf,5,6)
    error_report(err_t* e, const char* module, int line, const char* func,
                 const char* fmt, ...);

void error_reportv(err_t* e, const char* module, int line,
                   const char* func, const char* fmt, va_list va);

void error_print_stack(err_t* e, FILE* f);

// returns the error messages (not module:lines) in a newly-allocated string
char* error_get_errs(err_t* e, const char* separator);


#endif

