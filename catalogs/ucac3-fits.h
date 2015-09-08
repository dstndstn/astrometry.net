/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef UCAC3_FITS_H
#define UCAC3_FITS_H

#include <stdio.h>

#include "astrometry/ucac3.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_UCAC3 "UCAC3"

typedef fitstable_t ucac3_fits;

ucac3_fits* ucac3_fits_open(char* fn);

ucac3_fits* ucac3_fits_open_for_writing(char* fn);

int ucac3_fits_write_headers(ucac3_fits* ucac3);

int ucac3_fits_fix_headers(ucac3_fits* ucac3);

int ucac3_fits_count_entries(ucac3_fits* ucac3);

ucac3_entry* ucac3_fits_read_entry(ucac3_fits* t);

int ucac3_fits_read_entries(ucac3_fits* ucac3, int offset,
                            int count, ucac3_entry* entries);

int ucac3_fits_close(ucac3_fits* ucac3);

int ucac3_fits_write_entry(ucac3_fits* ucac3, ucac3_entry* entry);

#endif
