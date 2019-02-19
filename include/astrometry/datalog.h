/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef DATA_LOG_H
#define DATA_LOG_H

#include <stdio.h>
#include "astrometry/an-bool.h"
#include "astrometry/keywords.h"

typedef uint32_t data_mask_t;

#define DATALOG_MASK_ALL UINT32_MAX



// FIXME -- duh, datalog should use log!

struct data_log_t {
    data_mask_t mask;
    int level;
    FILE* f;
    int nitems;
};
typedef struct data_log_t data_log_t;

/**
 * Initialize global logging object. Must be called before any of the other
 * log_* functions.
 */
void data_log_init(int level);

void data_log_start();
void data_log_end();

void data_log_start_item(data_mask_t mask, int level, const char* name);
void data_log_end_item(data_mask_t mask, int level);

void data_log_enable(data_mask_t mask);

void data_log_enable_all();

void data_log_set_level(int level);

/**
 Sends global data_logging to the given FILE*.
 */
void data_log_to(FILE* fid);

anbool data_log_passes(data_mask_t mask, int level);

void
ATTRIB_FORMAT(printf, 3, 4)
    data_log(data_mask_t mask, int level, const char* format, ...);

#endif
