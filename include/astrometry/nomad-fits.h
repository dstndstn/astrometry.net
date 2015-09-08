/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef NOMAD_FITS_H
#define NOMAD_FITS_H

#include <stdio.h>

#include "astrometry/nomad.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_NOMAD "NOMAD"

typedef fitstable_t nomad_fits;

nomad_fits* nomad_fits_open(char* fn);

nomad_fits* nomad_fits_open_for_writing(char* fn);

int nomad_fits_write_headers(nomad_fits* nomad);

int nomad_fits_fix_headers(nomad_fits* nomad);

int nomad_fits_count_entries(nomad_fits* nomad);

nomad_entry* nomad_fits_read_entry(nomad_fits* t);

int nomad_fits_read_entries(nomad_fits* nomad, int offset,
                            int count, nomad_entry* entries);

int nomad_fits_close(nomad_fits* nomad);

int nomad_fits_write_entry(nomad_fits* nomad, nomad_entry* entry);

#endif
