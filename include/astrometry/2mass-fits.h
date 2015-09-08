/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef TWOMASS_FITS_H
#define TWOMASS_FITS_H

#include "astrometry/anqfits.h"
#include "astrometry/2mass.h"
#include "astrometry/fitstable.h"
#include "astrometry/ioutils.h"

#define AN_FILETYPE_2MASS "2MASS"

typedef fitstable_t twomass_fits;

twomass_fits* twomass_fits_open(char* fn);

twomass_fits* twomass_fits_open_for_writing(char* fn);

int twomass_fits_write_headers(twomass_fits* cat);

int twomass_fits_fix_headers(twomass_fits* cat);

int twomass_fits_read_entries(twomass_fits* cat, int offset,
                              int count, twomass_entry* entries);

twomass_entry* twomass_fits_read_entry(twomass_fits* cat);

int twomass_fits_count_entries(twomass_fits* cat);

int twomass_fits_close(twomass_fits* cat);

int twomass_fits_write_entry(twomass_fits* cat, twomass_entry* entry);

qfits_header* twomass_fits_get_primary_header(const twomass_fits* cat);

#endif
