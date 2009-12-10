/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
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

#ifndef MATCHOBJ_H
#define MATCHOBJ_H

#include <stdint.h>

#include "starutil.h"
#include "sip.h"
#include "bl.h"

struct match_struct {
    unsigned int quadno;
	unsigned int star[DQMAX];
	unsigned int field[DQMAX];
	uint64_t ids[DQMAX];
	// actually code error ^2.
    float code_err;

	// Pixel positions of the quad stars.
	double quadpix[2 * DQMAX];
	// Star positions of the quad stars.
	double quadxyz[3 * DQMAX];

	uint8_t dimquads;

	// the center of the field in xyz coords
	double center[3];
	// radius of the bounding circle, in distance on the unit sphere.
	double radius;
    // radius of the bounding circle in degrees
    double radius_deg;

	// WCS params
	bool wcs_valid;
	tan_t wcstan;

	// arcseconds per pixel; computed: scale=3600*sqrt(abs(det(wcstan.cd)))
	double scale;
    
    // How many quads were matched to this single quad (not a running total)
    // (only counts a single ABCD permutation)
    int16_t quad_npeers;

	int16_t nmatch;
	int16_t ndistractor;
	int16_t nconflict;
	int16_t nfield;
	int16_t nindex;

	// nbest = besti+1 = nmatch + ndistractor + nconflict <= nfield
	int16_t nbest;

	float logodds;

	float worstlogodds;

	// how many other matches agreed with this one *at the time it was found*
	int16_t nagree;

	int fieldnum;
	int fieldfile;
	int16_t indexid;
	int16_t healpix;
	int16_t hpnside;

	char fieldname[32];

	bool parity;

	// how many field quads did we try before finding this one?
	int quads_tried;
	// how many matching quads from the index did we find before this one?
	int quads_matched;
	// how many matching quads had the right scale?
	int quads_scaleok;
	// how many field objects did we have to look at?
	//  (this isn't stored in the matchfile, it's max(field))
	int objs_tried;
	// how many matches have we run verification on?
	int nverified;
	// how many seconds of CPU time have we spent on this field?
	float timeused;

    // stuff used by blind...
    sip_t* sip;
    double* indexrdls;
    int nindexrdls;

	bl* tagalong;

    // in arcsec.
    double index_jitter;

    il* corr_field;
    il* corr_index;
    dl* corr_field_xy;
    dl* corr_index_rd;
};
typedef struct match_struct MatchObj;

void matchobj_compute_overlap(MatchObj* mo);

// compute all derived fields.
void matchobj_compute_derived(MatchObj* mo);

#endif
