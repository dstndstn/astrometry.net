/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang

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
#ifndef DATA_LOG_H
#define DATA_LOG_H

#include <stdio.h>
#include "an-bool.h"
#include "keywords.h"

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
