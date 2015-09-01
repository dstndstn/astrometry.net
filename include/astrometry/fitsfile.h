/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef FITSFILE_H
#define FITSFILE_H

#include <stdio.h>

#include "astrometry/qfits_header.h"

int fitsfile_pad_with(FILE* fid, char pad);

int fitsfile_write_primary_header(FILE* fid, qfits_header* hdr,
                                  off_t* end_offset, const char* fn);

int fitsfile_fix_primary_header(FILE* fid, qfits_header* hdr,
                                off_t* end_offset, const char* fn);

// set ext = -1 if unknown.
int fitsfile_write_header(FILE* fid, qfits_header* hdr,
                          off_t* start_offset, off_t* end_offset,
                          int ext, const char* fn);

// set ext = -1 if unknown.
int fitsfile_fix_header(FILE* fid, qfits_header* hdr,
                        off_t* start_offset, off_t* end_offset,
                        int ext, const char* fn);



#endif

