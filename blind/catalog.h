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

#ifndef CATUTILS_H
#define CATUTILS_H

#include <sys/types.h>

#include "qfits.h"
#include "fitsbin.h"

#define AN_FILETYPE_CATALOG "OBJS"

struct catalog {
	int numstars;
	double ramin;
	double ramax;
	double decmin;
	double decmax;
	int healpix;

	double* stars;

	// optional table: star magnitudes.
	float* mags;

    fitsbin_t* fb;
};
typedef struct catalog catalog;

// FIXME.
// if "modifiable" is non-zero, a private copy-on-write is made.
// (changes don't automatically get written to the file.)

catalog* catalog_open(char* catfn);

catalog* catalog_open_for_writing(char* catfn);

double* catalog_get_star(catalog* cat, int sid);

double* catalog_get_base(catalog* cat);

int catalog_write_star(catalog* cat, double* star);

int catalog_close(catalog* cat);

void catalog_compute_radecminmax(catalog* cat);

int catalog_write_header(catalog* cat);

qfits_header* catalog_get_header(catalog* cat);

int catalog_fix_header(catalog* cat);

int catalog_write_to_file(catalog* cat, char* fn);

/*
  This should be called after writing all the star positions and
  calling catalog_fix_header().  It appends the data in "cat->mags"
  to the file as an extra FITS table.
 */
int catalog_write_mags(catalog* cat);

#endif
