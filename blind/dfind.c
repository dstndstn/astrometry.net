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

// Yummy preprocessor templating goodness!

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

