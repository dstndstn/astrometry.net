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

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>

#include "errors.h"
#include "ioutils.h"
#include "an-bool.h"

static pl* estack = NULL;
static bool atexit_registered = FALSE;

static err_t* error_copy(err_t* e) {
    err_t* copy = error_new();
    copy->print = e->print;
    copy->save = e->save;
    sl_append_contents(copy->errstack, e->errstack);
    sl_append_contents(copy->modstack, e->modstack);
    return copy;
}

static FILE* print_errs_fid;
static void print_errs(void) {
    FILE* fid = print_errs_fid;
    //fprintf(fid, "Error traceback:\n");
    errors_print_stack(fid);
}

void errors_start_logging_to_string() {
    err_t* err;
    errors_push_state();
    err = errors_get_state();
    err->print = NULL;
    err->save = TRUE;
}

char* errors_stop_logging_to_string(const char* separator) {
    err_t* err;
    char* rtn;
    err = errors_get_state();
    rtn = error_get_errs(err, separator);
    errors_pop_state();
    return rtn;
}

int errors_print_on_exit(FILE* fid) {
    err_t* e;
    errors_push_state();
    e = errors_get_state();
    e->save = TRUE;
    e->print = NULL;
    print_errs_fid = fid;
    return atexit(print_errs);
}

void errors_log_to(FILE* f) {
    err_t* e;
    e = errors_get_state();
    e->print = f;
}

void errors_clear_stack() {
    error_clear_stack(errors_get_state());
}

void error_clear_stack(err_t* e) {
    sl_remove_all(e->modstack);
    sl_remove_all(e->errstack);
}

err_t* errors_get_state() {
    if (!estack) {
        estack = pl_new(4);
        // register an atexit() function to clean up.
        if (!atexit_registered) {
            if (atexit(errors_free) == 0)
                atexit_registered = TRUE;
        }
    } 
    if (!pl_size(estack)) {
        err_t* e = error_new();
        e->print = stderr;
        pl_append(estack, e);
    }
    return pl_get(estack, pl_size(estack)-1);
}

void errors_free() {
    int i;
    if (!estack)
        return;
    for (i=0; i<pl_size(estack); i++) {
        err_t* e = pl_get(estack, i);
        error_free(e);
    }
    pl_free(estack);
    estack = NULL;
}

void errors_push_state() {
    err_t* now;
    err_t* snapshot;
    // make sure the stack and current state are initialized
    errors_get_state();
    now = pl_pop(estack);
    snapshot = error_copy(now);
    pl_push(estack, snapshot);
    pl_push(estack, now);
}

void errors_pop_state() {
    err_t* now = pl_pop(estack);
    error_free(now);
}

void errors_print_stack(FILE* f) {
    error_print_stack(errors_get_state(), f);
}

void report_error(const char* modfile, int modline,
                  const char* modfunc, const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    error_reportv(errors_get_state(), modfile, modline, modfunc, fmt, va);
    va_end(va);
}

void report_errno() {
    error_report(errors_get_state(), "system", -1, "", "%s", strerror(errno));
}

err_t* error_new() {
    err_t* e = calloc(1, sizeof(err_t));
    e->modstack = sl_new(4);
    e->errstack = sl_new(4);
    return e;
}

void error_free(err_t* e) {
    if (!e) return;
    sl_free2(e->modstack);
    sl_free2(e->errstack);
    free(e);
}

int error_nerrs(err_t* e) {
    return sl_size(e->errstack);
}

char* error_get_errstr(err_t* e, int i) {
    return sl_get(e->errstack, i);
}

void error_report(err_t* e, const char* module, int line, const char* func, 
                  const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    error_reportv(errors_get_state(), module, line, func, fmt, va);
    va_end(va);
}

void error_reportv(err_t* e, const char* module, int line,
                   const char* func, const char* fmt, va_list va) {
    if (e->print) {
        if (line == -1)
            fprintf(e->print, "%s: ", module);
        else
            fprintf(e->print, "%s:%i:%s: ", module, line, func);
        vfprintf(e->print, fmt, va);
        fprintf(e->print, "\n");
    }
    if (e->save) {
        sl_appendvf(e->errstack, fmt, va);
        if (line >= 0) {
            sl_appendf(e->modstack, "%s:%i:%s", module, line, func);
        } else {
            sl_appendf(e->modstack, "%s", module);
        }
    }
}

void error_print_stack(err_t* e, FILE* f) {
    int i;
    for (i=sl_size(e->modstack)-1; i>=0; i--) {
        char* mod = sl_get(e->modstack, i);
        char* err = sl_get(e->errstack, i);
        if (i < sl_size(e->modstack)-1)
            fprintf(f, "  ");
        fprintf(f, "%s: %s\n", mod, err);
    }
}

char* error_get_errs(err_t* e, const char* separator) {
    return sl_join_reverse(e->errstack, separator);
}

void errors_regex_error(int errcode, const regex_t* re) {
    char str[256];
    regerror(errcode, re, str, sizeof(str));
    error_report(errors_get_state(), "regex", -1, NULL, "%s", str);
}
