/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.
  Copyright 2013 Michal Kočer, Klet Observatory.

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

#ifndef UCAC4_FITS_H
#define UCAC4_FITS_H

#include <stdio.h>

#include "astrometry/ucac4.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_UCAC4 "UCAC4"

typedef fitstable_t ucac4_fits;

ucac4_fits* ucac4_fits_open(char* fn);

ucac4_fits* ucac4_fits_open_for_writing(char* fn);

int ucac4_fits_write_headers(ucac4_fits* ucac4);

int ucac4_fits_fix_headers(ucac4_fits* ucac4);

int ucac4_fits_count_entries(ucac4_fits* ucac4);

ucac4_entry* ucac4_fits_read_entry(ucac4_fits* t);

int ucac4_fits_read_entries(ucac4_fits* ucac4, int offset,
                            int count, ucac4_entry* entries);

int ucac4_fits_close(ucac4_fits* ucac4);

int ucac4_fits_write_entry(ucac4_fits* ucac4, ucac4_entry* entry);

#endif
