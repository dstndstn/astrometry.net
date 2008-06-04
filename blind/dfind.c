/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg, Sam Roweis
  and Dustin Lang.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
#include <assert.h>

#include "simplexy-common.h"
#include "dimage.h"
#include "bl.h"

/*
 * dfind.c
 *
 * Find non-zero objects in a binary image.
 *
 * Mike Blanton
 * 1/2006
 *
 * dfind2() Keir Mierle 2007
 */

#define DEBUG_DFIND 0

int initial_max_groups = 50;

/*
 * This code does connected component analysis, but instead of returning a list
 * of components, it does the following crazy thing: it returns an image where
 * each component is all numbered the same. For example, if your input is:
 *
 *  . . . . . . . . .
 *  . . . 1 . . . . .
 *  . . 1 1 1 . . . .
 *  . . . 1 . . . . .
 *  . . . . . . . . .
 *  . 1 1 . . . 1 . .
 *  . 1 1 . . 1 1 1 .
 *  . 1 1 . . . 1 . .
 *  . . . . . . . . .
 *
 * where . is 0. Then your output is:
 *
 *  . . . . . . . . .
 *  . . . 0 . . . . .
 *  . . 0 0 0 . . . .
 *  . . . 0 . . . . .
 *  . . . . . . . . .
 *  . 1 1 . . . 2 . .
 *  . 1 1 . . 2 2 2 .
 *  . 1 1 . . . 2 . .
 *  . . . . . . . . .
 *
 * where . is now -1. Diagonals are considered connections, so the following is
 * a single component:
 *  . . . . .
 *  . 1 . 1 .
 *  . . 1 . .
 *  . 1 . 1 .
 *  . . . . .
 */

/* Finds the root of this set (which is the min label) but collapses
 * intermediate labels as it goes. */
label_t collapsing_find_minlabel(label_t label,
                                 label_t *equivs) {
	int min;
	min = label;
	while (equivs[min] != min)
		min = equivs[min];
	while (label != min) {
		int next = equivs[label];
		equivs[label] = min;
		label = next;
	}
	return min;
}

// Yummy preprocessor goodness!

static label_t relabel_image(il* on_pixels,
                             int maxlabel,
                             label_t* equivs,
                             int* object) {
    int i;
	label_t maxcontiguouslabel = 0;
	label_t *number;
	number = malloc(sizeof(label_t) * maxlabel);
	assert(number);
	for (i = 0; i < maxlabel; i++)
		number[i] = LABEL_MAX;
	for (i=0; i<il_size(on_pixels); i++) {
        int onpix;
		int minlabel;
        onpix = il_get(on_pixels, i);
        minlabel = collapsing_find_minlabel(object[onpix], equivs);
		if (number[minlabel] == LABEL_MAX)
			number[minlabel] = maxcontiguouslabel++;
        object[onpix] = number[minlabel];
	}
	free(number);
    return maxcontiguouslabel;
}

#define DFIND2 dfind2
#define IMGTYPE int
#include "dfind2.c"
#undef DFIND2
#undef IMGTYPE

#define DFIND2 dfind2_u8
#define IMGTYPE unsigned char
#include "dfind2.c"
#undef DFIND2
#undef IMGTYPE


int dfind(int *image,
          int nx,
          int ny,
          int *object) {
	int i, ip, j, jp, k, kp, l, ist, ind, jst, jnd, igroup, minearly, checkearly, tmpearly;
	int ngroups;

	int* mapgroup = (int *) malloc((size_t) nx * ny * sizeof(int));
	int* matches = (int *) malloc((size_t) nx * ny * 9 * sizeof(int));
	int* nmatches = (int *) malloc((size_t) nx * ny * sizeof(int));

	if (!mapgroup || !matches || !nmatches) {
		fprintf(stderr, "Failed to allocate memory in dfind.c\n");
		exit(-1);
	}

	for (k = 0;k < nx*ny;k++)
		object[k] = -1;
	for (k = 0;k < nx*ny;k++)
		mapgroup[k] = -1;
	for (k = 0;k < nx*ny;k++)
		nmatches[k] = 0;
	for (k = 0;k < nx*ny*9;k++)
		matches[k] = -1;

	/* find matches */
	for (j = 0;j < ny;j++) {
		jst = j - 1;
		jnd = j + 1;
		if (jst < 0)
			jst = 0;
		if (jnd > ny - 1)
			jnd = ny - 1;
		for (i = 0;i < nx;i++) {
			ist = i - 1;
			ind = i + 1;
			if (ist < 0)
				ist = 0;
			if (ind > nx - 1)
				ind = nx - 1;
			k = i + j * nx;
			if (image[k]) {
				for (jp = jst;jp <= jnd;jp++)
					for (ip = ist;ip <= ind;ip++) {
						kp = ip + jp * nx;
						if (image[kp]) {
							matches[9*k + nmatches[k]] = kp;
							nmatches[k]++;
						}
					}
			} /* end if */
		}
	}

	/* group pixels on matches */
	igroup = 0;
	for (k = 0;k < nx*ny;k++) {
		if (image[k]) {
			minearly = igroup;
			for (l = 0;l < nmatches[k];l++) {
				kp = matches[9 * k + l];
				checkearly = object[kp];
				if (checkearly >= 0) {
					while (mapgroup[checkearly] != checkearly) {
						checkearly = mapgroup[checkearly];
					}
					if (checkearly < minearly)
						minearly = checkearly;
				}
			}

			if (minearly == igroup) {
				mapgroup[igroup] = igroup;
				for (l = 0;l < nmatches[k];l++) {
					kp = matches[9 * k + l];
					object[kp] = igroup;
				}
				igroup++;
			} else {
				for (l = 0;l < nmatches[k];l++) {
					kp = matches[9 * k + l];
					checkearly = object[kp];
					if (checkearly >= 0) {
						while (mapgroup[checkearly] != checkearly) {
							tmpearly = mapgroup[checkearly];
							mapgroup[checkearly] = minearly;
							checkearly = tmpearly;
						}
						mapgroup[checkearly] = minearly;
					}
				}
				for (l = 0;l < nmatches[k];l++) {
					kp = matches[9 * k + l];
					object[kp] = minearly;
				}
			}
		}
	}

	ngroups = 0;
	for (i = 0;i < nx*ny;i++) {
		if (mapgroup[i] >= 0) {
			if (mapgroup[i] == i) {
				mapgroup[i] = ngroups;
				ngroups++;
			} else {
				mapgroup[i] = mapgroup[mapgroup[i]];
			}
		}
	}

	if (ngroups == 0)
		goto bail;

	for (i = 0;i < nx*ny;i++)
        if (object[i] >= 0)
            object[i] = mapgroup[object[i]];

	for (i = 0;i < nx*ny;i++)
		mapgroup[i] = -1;
	igroup = 0;
	for (k = 0;k < nx*ny;k++) {
		if (image[k] > 0 && mapgroup[object[k]] == -1) {
			mapgroup[object[k]] = igroup;
			igroup++;
		}
	}

	for (i = 0;i < nx*ny;i++)
		if (image[i] > 0)
			object[i] = mapgroup[object[i]];
		else
			object[i] = -1;

bail:
	FREEVEC(matches);
	FREEVEC(nmatches);
	FREEVEC(mapgroup);

	return (1);
} /* end dfind */
