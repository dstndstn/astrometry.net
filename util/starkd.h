/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2009 Dustin Lang.

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
#include "fitstable.h"

#include "anqfits.h"

#define AN_FILETYPE_STARTREE "SKDT"

#define AN_FILETYPE_TAGALONG "TAGALONG"

#define STARTREE_NAME "stars"

struct startree_s {
	kdtree_t* tree;
	qfits_header* header;
	int* inverse_perm;
	uint8_t* sweep;

    // reading or writing?
    int writing;

	// reading: tagged-along data (a FITS BINTABLE with one row per star,
	// in the same order); access this via startree_get_tagalong() ONLY!
	fitstable_t* tagalong;
};
typedef struct startree_s startree_t;


startree_t* startree_open(const char* fn);

startree_t* startree_open_fits(anqfits_t* fits);

/**
   Searches for stars within a radius of a point.

 xyzcenter: double[3]: unit-sphere coordinates of point; see
 starutil.h : radecdeg2xyzarr() to convert RA,Decs to this form.

 radius2: radius-square on the unit sphere; see starutil.h :
 deg2distsq() or arcsec2distsq().

 xyzresults: if non-NULL, returns the xyz positions of the stars that
 are found, in a newly-allocated array.

 radecresults: if non-NULL, returns the RA,Dec positions (in degrees)
 of the stars within range.

 starinds: if non-NULL, returns the indices of stars within range.
 This can be used to retrieve extra information about the stars, using
 the 'startree_get_data_column()' function.
 
 */
void startree_search_for(const startree_t* s, const double* xyzcenter, double radius2,
						 double** xyzresults, double** radecresults,
						 int** starinds, int* nresults);

void startree_search(const startree_t* s, const double* xyzcenter, double radius2,
                     double** xyzresults, double** radecresults, int* nresults);

/**
 Reads a column of data from the "tag-along" table.

 The data should be freed using "startree_free_data_column"
 */
double* startree_get_data_column(startree_t* s, const char* colname, int* indices, int N);

/**
 Reads a column of data from the "tag-along" table.

 The column may be an array (that is, each row contains multiple
 entries); the array size is placed in "arraysize".
 */
double* startree_get_data_column_array(startree_t* s, const char* colname, int* indices, int N, int* arraysize);

void startree_free_data_column(startree_t* s, double* d);


bool startree_has_tagalong(startree_t* s);

fitstable_t* startree_get_tagalong(startree_t* s);

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

void startree_set_jitter(startree_t* s, double jitter_arcsec);

//uint64_t startree_get_starid(const startree_t* s, int ind);

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
