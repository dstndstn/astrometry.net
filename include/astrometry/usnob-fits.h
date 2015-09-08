/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef USNOB_FITS_H
#define USNOB_FITS_H

#include <stdio.h>

#include "astrometry/qfits_header.h"
#include "astrometry/usnob.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_USNOB "USNOB"

typedef fitstable_t usnob_fits;

usnob_fits* usnob_fits_open(char* fn);

usnob_fits* usnob_fits_open_for_writing(char* fn);

qfits_header* usnob_fits_get_header(usnob_fits* usnob);

int usnob_fits_write_headers(usnob_fits* usnob);

int usnob_fits_fix_headers(usnob_fits* usnob);

usnob_entry* usnob_fits_read_entry(usnob_fits* u);

int usnob_fits_read_entries(usnob_fits* usnob, int offset,
							int count, usnob_entry* entries);

int usnob_fits_count_entries(usnob_fits* usnob);

int usnob_fits_close(usnob_fits* usnob);

int usnob_fits_write_entry(usnob_fits* usnob, usnob_entry* entry);

int usnob_fits_remove_an_diffraction_spike_column(usnob_fits* usnob);

#endif
