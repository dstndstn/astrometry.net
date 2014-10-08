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

#include <stdio.h>

#include "boilerplate.h"
#include "fitsioutils.h"
#include "git.h"

void boilerplate_help_header(FILE* fid) {
    fprintf(fid, "This program is part of the Astrometry.net suite.\n");
    fprintf(fid, "For details, visit  http://astrometry.net .\n");
    fprintf(fid, "Git URL %s\n", git_url());
    fprintf(fid, "Revision %s, date %s.\n", git_revision(), git_date());
}

void boilerplate_add_fits_headers(qfits_header* hdr) {
    fits_add_long_history(hdr, "Created by the Astrometry.net suite.");
    fits_add_long_history(hdr, "For more details, see http://astrometry.net .");
    fits_add_long_history(hdr, "Git URL %s", git_url());
    fits_add_long_history(hdr, "Git revision %s", git_revision());
    fits_add_long_history(hdr, "Git date %s", git_date());
}
