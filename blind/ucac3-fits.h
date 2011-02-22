/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.

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

#ifndef UCAC3_FITS_H
#define UCAC3_FITS_H

#include <stdio.h>

#include "qfits.h"
#include "ucac3.h"
#include "fitstable.h"

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
