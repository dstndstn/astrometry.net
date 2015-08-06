/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef BOILERPLATE_H
#define BOILERPLATE_H

#include <stdio.h>
#include "astrometry/qfits_header.h"

#define BOILERPLATE_HELP_HEADER(fid)                                         \
do {								             \
    fprintf(fid, "This program is part of the Astrometry.net suite.\n");     \
    fprintf(fid, "For details, visit http://astrometry.net.\n");	     \
    fprintf(fid, "Git URL %s\n", AN_GIT_URL);                      \
    fprintf(fid, "Revision %s, date %s.\n", AN_GIT_REVISION, AN_GIT_DATE); \
} while (0)

#define BOILERPLATE_ADD_FITS_HEADERS(hdr)                                        \
do {                                                                             \
    fits_add_long_history(hdr, "Created by the Astrometry.net suite.");          \
    fits_add_long_history(hdr, "For more details, see http://astrometry.net.");  \
    fits_add_long_history(hdr, "Git URL %s", AN_GIT_URL);          \
    fits_add_long_history(hdr, "Git revision %s", AN_GIT_REVISION); \
    fits_add_long_history(hdr, "Git date %s", AN_GIT_DATE);        \
} while (0)

#endif
