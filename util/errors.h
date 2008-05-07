/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef AN_ERRORS_H
#define AN_ERRORS_H

#include <stdarg.h>
#include <stdio.h>

#include "an-bool.h"
#include "bl.h"
#include "keywords.h"

struct errors {
    FILE* print;
    bool save;
    sl* modstack;
    sl* errstack;
};
typedef struct errors err_t;


/***    Global functions    ***/

err_t* errors_get_state();

// takes a (deep) snapshot of the current error handling state and pushes it onto the
// stack.
void errors_push_state();

// 
void errors_pop_state();

void
ATTRIB_FORMAT(printf,3,4)
report_error(const char* modfile, int modline, const char* fmt, ...);

void report_errno();

#define ERROR(fmt, ...) report_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define SYSERROR(fmt, ...) do { report_errno(); report_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__); } while(0)

// free globals.
void errors_free();

void errors_log_to(FILE* f);

void errors_print_stack(FILE* f);

void errors_clear_stack();

int errors_print_on_exit(FILE* fid);

/***    End globals   ***/






err_t* error_new();

void error_free(err_t* e);

int error_nerrs(err_t* e);

char* error_get_errstr(err_t* e, int i);

void
ATTRIB_FORMAT(printf,4,5)
error_report(err_t* e, const char* module, int line, const char* fmt, ...);

void error_reportv(err_t* e, const char* module, int line, const char* fmt, va_list va);

void error_print_stack(err_t* e, FILE* f);

// returns the error messages (not module:lines) in a newly-allocated string
char* error_get_errs(err_t* e, const char* separator);

void error_clear_stack(err_t* e);

#endif

