/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.
  Copyright 2013 Michal Kočer, Klet Observatory.
  Copyright 2021 Vladimir Kouprianov, Skynet RTN, University of North Carolina at Chapel Hill

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

// Author: Vladimir Kouprianov, Skynet RTN, University of North Carolina at Chapel Hill

#ifndef UCAC5_FITS_H
#define UCAC5_FITS_H

#include <stdio.h>

#include "astrometry/ucac5.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_UCAC5 "UCAC5"

typedef fitstable_t ucac5_fits;

ucac5_fits* ucac5_fits_open(char* fn, anbool full);

ucac5_fits* ucac5_fits_open_for_writing(char* fn, anbool full);

int ucac5_fits_write_headers(ucac5_fits* ucac5);

int ucac5_fits_fix_headers(ucac5_fits* ucac5);

int ucac5_fits_count_entries(ucac5_fits* ucac5);

ucac5_entry* ucac5_fits_read_entry(ucac5_fits* t);

int ucac5_fits_read_entries(ucac5_fits* ucac5, int offset,
                            int count, ucac5_entry* entries);

int ucac5_fits_close(ucac5_fits* ucac5);

int ucac5_fits_write_entry(ucac5_fits* ucac5, ucac5_entry* entry);

#endif
