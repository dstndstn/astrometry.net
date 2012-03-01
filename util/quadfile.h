/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2009, 2010 Dustin Lang.

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

#ifndef QUADFILE_H
#define QUADFILE_H

#include <sys/types.h>
#include <stdint.h>

#include "qfits.h"
#include "fitsbin.h"
#include "anqfits.h"

struct quadfile {
	unsigned int numquads;
	unsigned int numstars;
    int dimquads;
	// upper bound of AB distance of quads in this index
	double index_scale_upper;
	// lower bound
	double index_scale_lower;
	// unique ID of this index
	int indexid;
	// healpix covered by this index
	int healpix;
    // Nside of the healpixelization
    int hpnside;

	fitsbin_t* fb;
	// when reading:
	uint32_t* quadarray;
};
typedef struct quadfile quadfile;

quadfile* quadfile_open(const char* fname);
quadfile* quadfile_open_fits(anqfits_t* fits);

char* quadfile_get_filename(const quadfile* qf);

quadfile* quadfile_open_for_writing(const char* quadfname);

quadfile* quadfile_open_in_memory(void);

int quadfile_switch_to_reading(quadfile* qf);

int quadfile_close(quadfile* qf);

// Look at each quad, and ensure that the star ids it contains are all
// less than the number of stars ("numstars").  Returns 0=ok, -1=problem
int quadfile_check(const quadfile* qf);

// Copies the star ids of the stars that comprise quad "quadid".
// There will be qf->dimquads such stars.
// (this will be less than starutil.h : DQMAX, for ease of static
// allocation of arrays that will hold quads of stars)
int quadfile_get_stars(const quadfile* qf, unsigned int quadid,
                       unsigned int* stars);

int quadfile_write_quad(quadfile* qf, unsigned int* stars);

int quadfile_dimquads(const quadfile* qf);

int quadfile_nquads(const quadfile* qf);

int quadfile_fix_header(quadfile* qf);

int quadfile_write_header(quadfile* qf);

double quadfile_get_index_scale_upper_arcsec(const quadfile* qf);

double quadfile_get_index_scale_lower_arcsec(const quadfile* qf);

qfits_header* quadfile_get_header(const quadfile* qf);

int quadfile_write_header_to(quadfile* qf, FILE* fid);

int quadfile_write_all_quads_to(quadfile* qf, FILE* fid);

#endif
