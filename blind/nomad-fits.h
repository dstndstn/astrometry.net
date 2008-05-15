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

#ifndef NOMAD_FITS_H
#define NOMAD_FITS_H

#include <stdio.h>

#include "qfits.h"
#include "nomad.h"
#include "fitstable.h"

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
