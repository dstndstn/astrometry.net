/*
 This file is part of the Astrometry.net suite.
 Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef TWOMASS_FITS_H
#define TWOMASS_FITS_H

#include "qfits.h"
#include "2mass.h"
#include "fitstable.h"
#include "ioutils.h"

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
