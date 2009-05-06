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

#include "bl.h"
#include "healpix.h"
#include "mathutil.h"
#include "starutil.h"

il* healpix_approx_rangesearch(double* xyz, double radius, int Nside, il* hps) {
	int hp;
	double hprad = arcmin2dist(healpix_side_length_arcmin(Nside));
	il* frontier = il_new(256);
	il* bad = il_new(256);
	if (!hps)
		hps = il_new(256);

	hp = xyzarrtohealpix(xyz, Nside);
	il_append(frontier, hp);
	il_append(hps, hp);
	while (il_size(frontier)) {
		int nn, neighbours[8];
		int i;
		hp = il_pop(frontier);
		nn = healpix_get_neighbours(hp, neighbours, Nside);
		for (i=0; i<nn; i++) {
			double nxyz[3];
			if (il_contains(frontier, neighbours[i]))
				continue;
			if (il_contains(bad, neighbours[i]))
				continue;
			if (il_contains(hps, neighbours[i]))
				continue;
			healpix_to_xyzarr(neighbours[i], Nside, 0.5, 0.5, nxyz);
			if (sqrt(distsq(xyz, nxyz, 3)) - hprad < radius) {
				// in range!
				il_append(frontier, neighbours[i]);
				il_append(hps, neighbours[i]);
			} else
				il_append(bad, neighbours[i]);
		}
	}

	return hps;
}
