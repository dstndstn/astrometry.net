/*
 This file is part of the Astrometry.net suite.
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

#ifndef HEALPIX_UTILS_H
#define HEALPIX_UTILS_H

#include "bl.h"

/**
 Returns healpixes that are / may be within range of the given point, resp.
 */
il* healpix_rangesearch_xyz(const double* xyz, double radius, int Nside, il* hps);
il* healpix_rangesearch_xyz_approx(const double* xyz, double radius, int Nside, il* hps);
il* healpix_rangesearch_radec_approx(double ra, double dec, double radius, int Nside, il* hps);
il* healpix_rangesearch_radec(double ra, double dec, double radius, int Nside, il* hps);

/**
 Starting from a "seed" or list of "seeds" healpixes, grows a region
 by looking at healpix neighbours.  Accepts healpixes for which the
 "accept" function returns 1.  Returns the healpixes that are
 accepted.  The accepted results are placed in "accepted", if
 non-NULL, or in a newly-allocated list.

 If "rejected" is non-NULL, the healpixes that are rejected will be
 put there.

 If "depth" is non-zero, that number of neighbour steps will be taken.
 Zero means no limit.

 NOTE that any existing entries in the "accepted" list will be treated
 as having already been accepted: when the search reaches them, their
 neighbours will not be added to the frontier to explore.
 */
il* healpix_region_search(int seed, il* seeds, int Nside,
						  il* accepted, il* rejected,
						  int (*accept)(int hp, void* token),
						  void* token,
						  int depth);


#endif
