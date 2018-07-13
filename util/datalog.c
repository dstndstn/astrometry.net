/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "datalog.h"

static data_log_t g_datalog;

static data_log_t* get_logger() {
    return &g_datalog;
}

void data_log_init(int level) {
    data_log_t* log = get_logger();
    log->level = level;
    log->f = stdout;
    log->mask = 0;
    log->nitems = 0;
}

void data_log_start() {
    data_log_t* log = get_logger();
    fprintf(log->f, "[\n");
}

void data_log_end() {
    data_log_t* log = get_logger();
    fprintf(log->f, "]\n");
}

void data_log_start_item(data_mask_t mask, int level, const char* name) {
    data_log_t* log = get_logger();
    data_log(mask, level, "%s{\"%s\": ", (log->nitems ? ",\n" : ""), name);
}

void data_log_end_item(data_mask_t mask, int level) {
    data_log_t* log = get_logger();
    data_log(mask, level, "}");
    if (data_log_passes(mask, level))
        log->nitems++;
}

void data_log_enable(data_mask_t mask) {
    data_log_t* log = get_logger();
    log->mask |= mask;
}

void data_log_enable_all() {
    data_log_t* log = get_logger();
    log->mask |= DATALOG_MASK_ALL;
}

void data_log_set_level(int level) {
    data_log_t* log = get_logger();
    log->level = level;
}

void data_log_to(FILE* fid) {
    data_log_t* log = get_logger();
    log->f = fid;
}

anbool data_log_passes(data_mask_t mask, int level) {
    data_log_t* log = get_logger();
    if (level > log->level)
        return FALSE;
    if ((mask & log->mask) == 0)
        return FALSE;
    return TRUE;
}

void data_log(data_mask_t mask, int level, const char* format, ...) {
    data_log_t* log = get_logger();
    va_list va;
    if (!data_log_passes(mask, level))
        return;
    va_start(va, format);
    vfprintf(log->f, format, va);
    va_end(va);
    fflush(log->f);
}

