/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef CONSTELLATIONS_H
#define CONSTELLATIONS_H

#include "bl.h"

int constellations_n();

const char* constellations_get_shortname(int constellation_num);

const char* constellations_get_longname(int constellation_num);

const char* constellations_short_to_longname(const char* shortname);

int constellations_get_nlines(int constellation_num);

il* constellations_get_lines(int constellation_num);

il* constellations_get_unique_stars(int constellation_num);

/*
 Returns the star IDs of the line_num'th line.
 */
void constellations_get_line(int constellation_num, int line_num,
							 int* ep1, int* ep2);

/*
 Returns a newly-allocated dl* which is a list of (ra1, dec1), (ra2, dec2) coordinates
 of the line endpoints.
 */
dl* constellations_get_lines_radec(int constellation_num);

/*
 RA,Dec in degrees
 */
void constellations_get_star_radec(int starnum, double* ra, double* dec);

#endif
