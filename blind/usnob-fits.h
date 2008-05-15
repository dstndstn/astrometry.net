/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef USNOB_FITS_H
#define USNOB_FITS_H

#include <stdio.h>

#include "qfits.h"
#include "usnob.h"
#include "fitstable.h"

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
