/*
 This file is part of the Astrometry.net suite.
 Copyright 2009 Dustin Lang.

 The Astrometry.net suite is free software; you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2, or
 (at your option) any later version.

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

il* healpix_region_search(int seed, il* seeds, int Nside,
						  il* accepted, il* rejected,
						  int (*accept)(int hp, void* token),
						  void* token, int depth) {
	il* frontier;
	anbool allocd_rej = FALSE;
	int d;

	if (!accepted)
		accepted = il_new(256);
	if (!rejected) {
		rejected = il_new(256);
		allocd_rej = TRUE;
	}

	if (seeds)
		//frontier = seeds;
		frontier = il_dupe(seeds);
	else {
		frontier = il_new(256);
		il_append(frontier, seed);
	}

	for (d=0; !depth || d<depth; d++) {
		int j, N;
		N = il_size(frontier);
		if (N == 0)
			break;
		for (j=0; j<N; j++) {
			int hp;
			int i, nn, neigh[8];
			hp = il_get(frontier, j);
			nn = healpix_get_neighbours(hp, neigh, Nside);
			for (i=0; i<nn; i++) {
				if (il_contains(frontier, neigh[i]))
					continue;
				if (il_contains(rejected, neigh[i]))
					continue;
				if (il_contains(accepted, neigh[i]))
					continue;
				if (accept(neigh[i], token)) {
					il_append(accepted, neigh[i]);
					il_append(frontier, neigh[i]);
				} else
					il_append(rejected, neigh[i]);
			}
		}
		il_remove_index_range(frontier, 0, N);
	}

	il_free(frontier);
	if (allocd_rej)
		il_free(rejected);
	return accepted;
}


static il* hp_rangesearch(const double* xyz, double radius, int Nside, il* hps, anbool approx) {
	int hp;
	double hprad = arcmin2dist(healpix_side_length_arcmin(Nside)) * sqrt(2);
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
			anbool tst;
			double nxyz[3];
			if (il_contains(frontier, neighbours[i]))
				continue;
			if (il_contains(bad, neighbours[i]))
				continue;
			if (il_contains(hps, neighbours[i]))
				continue;
			if (approx) {
				healpix_to_xyzarr(neighbours[i], Nside, 0.5, 0.5, nxyz);
				tst = (sqrt(distsq(xyz, nxyz, 3)) - hprad <= radius);
			} else {
				tst = healpix_within_range_of_xyz(neighbours[i], Nside, xyz, radius);
			}
			if (tst) {
				// in range!
				il_append(frontier, neighbours[i]);
				il_append(hps, neighbours[i]);
			} else
				il_append(bad, neighbours[i]);
		}
	}

	il_free(bad);
	il_free(frontier);

	return hps;
}

il* healpix_rangesearch_xyz_approx(const double* xyz, double radius, int Nside, il* hps) {
	return hp_rangesearch(xyz, radius, Nside, hps, TRUE);
}

il* healpix_rangesearch_xyz(const double* xyz, double radius, int Nside, il* hps) {
	return hp_rangesearch(xyz, radius, Nside, hps, FALSE);
}

il* healpix_rangesearch_radec_approx(double ra, double dec, double radius, int Nside, il* hps) {
	double xyz[3];
	radecdeg2xyzarr(ra, dec, xyz);
	return hp_rangesearch(xyz, radius, Nside, hps, TRUE);
}

il* healpix_rangesearch_radec(double ra, double dec, double radius, int Nside, il* hps) {
	double xyz[3];
	radecdeg2xyzarr(ra, dec, xyz);
	return hp_rangesearch(xyz, radius, Nside, hps, FALSE);
}
