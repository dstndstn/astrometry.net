/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef CONSTELLATIONS_H
#define CONSTELLATIONS_H

#include "astrometry/bl.h"

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
