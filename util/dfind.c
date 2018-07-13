/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
dimage_label_t collapsing_find_minlabel(dimage_label_t label,
                                        dimage_label_t *equivs) {
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

static dimage_label_t relabel_image(il* on_pixels,
                                    int maxlabel,
                                    dimage_label_t* equivs,
                                    int* object) {
    int i;
    dimage_label_t maxcontiguouslabel = 0;
    dimage_label_t *number;
    number = malloc(sizeof(dimage_label_t) * maxlabel);
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

