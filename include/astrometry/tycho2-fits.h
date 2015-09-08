/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef TYCHO2_FITS_H
#define TYCHO2_FITS_H

#include <stdio.h>

#include "astrometry/tycho2.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_TYCHO2 "TYCHO2"

typedef struct fitstable_t tycho2_fits;

tycho2_fits* tycho2_fits_open(char* fn);

tycho2_fits* tycho2_fits_open_for_writing(char* fn);

qfits_header* tycho2_fits_get_header(tycho2_fits* tycho2);

int tycho2_fits_write_headers(tycho2_fits* tycho2);

int tycho2_fits_fix_headers(tycho2_fits* tycho2);

int tycho2_fits_count_entries(tycho2_fits* tycho2);

tycho2_entry* tycho2_fits_read_entry(tycho2_fits* t);

int tycho2_fits_read_entries(tycho2_fits* tycho2, int offset,
							 int count, tycho2_entry* entries);

int tycho2_fits_close(tycho2_fits* tycho2);

int tycho2_fits_write_entry(tycho2_fits* tycho2, tycho2_entry* entry);

#endif
