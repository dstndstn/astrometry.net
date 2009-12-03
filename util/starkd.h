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

#ifndef STAR_KD_H
#define STAR_KD_H

#include <stdint.h>
#include <stdio.h>

#include "kdtree.h"
#include "qfits.h"
#include "kdtree_fits_io.h"

#define AN_FILETYPE_STARTREE "SKDT"

#define STARTREE_NAME "stars"

struct startree_s {
	kdtree_t* tree;
	qfits_header* header;
	int* inverse_perm;
	uint8_t* sweep;

    // reading or writing?
    int writing;

    // optional tables: positional error ellipses, proper motions
    float* sigma_radec;
    float* proper_motion;
    float* sigma_pm;
    // optional tables: mag and mag error
    float* mag;
    float* mag_err;
    // optional table: star IDs
    uint64_t* starids;
};
typedef struct startree_s startree_t;


startree_t* startree_open(const char* fn);

void startree_search_for(const startree_t* s, const double* xyzcenter, double radius2,
						 double** xyzresults, double** radecresults,
						 int** starinds, int* nresults);

void startree_search(const startree_t* s, const double* xyzcenter, double radius2,
                     double** xyzresults, double** radecresults, int* nresults);

/*
 Retrieve parameters of the cut-an process, if they are available.
 Older index files may not have these header cards.
 */
// healpix nside, or -1
int startree_get_cut_nside(const startree_t* s);
int startree_get_cut_nsweeps(const startree_t* s);
// in arcsec; 0 if none.
double startree_get_cut_dedup(const startree_t* s);
// band (one of several static strings), or NULL
char* startree_get_cut_band(const startree_t* s);
// margin, in healpix, or -1
int startree_get_cut_margin(const startree_t* s);

double startree_get_jitter(const startree_t* s);


uint64_t startree_get_starid(const startree_t* s, int ind);

// returns the sweep number of star 'ind', or -1 if the index is out of bounds
// or the tree has no sweep numbers.
//int startree_get_sweep(const startree_t* s, int ind);

int startree_N(const startree_t* s);

int startree_nodes(const startree_t* s);

int startree_D(const startree_t* s);

qfits_header* startree_header(const startree_t* s);

int startree_get(startree_t* s, int starid, double* posn);

int startree_get_radec(startree_t* s, int starid, double* ra, double* dec);

int startree_close(startree_t* s);

void startree_compute_inverse_perm(startree_t* s);

int startree_check_inverse_perm(startree_t* s);

// for writing
startree_t* startree_new();

int startree_write_to_file(startree_t* s, const char* fn);

int startree_write_to_file_flipped(startree_t* s, const char* fn);

int startree_append_to(startree_t* s, FILE* fid);

#endif
