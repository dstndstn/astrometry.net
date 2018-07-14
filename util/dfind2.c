/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

// This file gets #included in dfind.c, with
// DFIND2 and IMGTYPE defined appropriately.
// I did this so that we can handle int* and
// unsigned char* images using the same code.

#include "os-features.h"
#include "errors.h"
#include "log.h"

int DFIND2(const IMGTYPE* image,
           int nx,
           int ny,
           int* object,
           int* pnobjects) {
    int ix, iy, i;
    int maxgroups = initial_max_groups;
    dimage_label_t *equivs = malloc(sizeof(dimage_label_t) * maxgroups);
    int maxlabel = 0;

    /* Keep track of 'on' pixels to avoid later rescanning */
    il* on_pixels = il_new(256);

    /* Find blobs and track equivalences */
    for (iy = 0; iy < ny; iy++) {
        for (ix = 0; ix < nx; ix++) {
            int thislabel, thislabelmin;

            object[nx*iy+ix] = -1;
            if (!image[nx*iy+ix])
                continue;

            /* Store location of each 'on' pixel. */
            il_append(on_pixels, nx * iy + ix);

            /* If the pixel to the left [exists and] is on, this pixel
             joins its group. */
            if (ix && image[nx*iy+ix-1]) {
                /* Old group */
                object[nx*iy+ix] = object[nx*iy+ix-1];

            } else {
                /* New blob */
                // FIXME this part should become uf_new_group()
                if (maxlabel >= maxgroups) {
                    maxgroups *= 2;
                    equivs = realloc(equivs, sizeof(dimage_label_t) * maxgroups);
                    assert(equivs);
                }
                object[nx*iy+ix] = maxlabel;
                equivs[maxlabel] = maxlabel;
                maxlabel++;

                if (maxlabel == LABEL_MAX) {
                    logverb("Ran out of labels.  Relabelling...\n");
                    maxlabel = relabel_image(on_pixels, maxlabel, equivs, object);
                    logverb("After relabelling, we need %i labels\n", maxlabel);
                    if (maxlabel == LABEL_MAX) {
                        ERROR("Ran out of labels.");
                        exit(-1);
                    }
                }
            }

            thislabel  = object[nx*iy + ix];

            /* Compute minimum equivalence label for this pixel */
            thislabelmin = collapsing_find_minlabel(thislabel, equivs);

            if (iy == 0)
                continue;

            /* Check three pixels above this one which are 'neighbours' */
            for (i = MAX(0, ix - 1); i <= MIN(ix + 1, nx - 1); i++) {
                if (image[nx*(iy-1)+i]) {
                    int otherlabel = object[nx*(iy-1) + i];

                    /* Find min of the other */
                    int otherlabelmin = collapsing_find_minlabel(otherlabel, equivs);

                    /* Merge groups if necessary */
                    if (thislabelmin != otherlabelmin) {
                        int oldlabelmin = MAX(thislabelmin, otherlabelmin);
                        int newlabelmin = MIN(thislabelmin, otherlabelmin);
                        thislabelmin = newlabelmin;
                        equivs[oldlabelmin] = newlabelmin;
                        equivs[thislabel] = newlabelmin;
                        /* Update other pixel too */
                        object[nx*(iy-1) + i] = newlabelmin;
                    } 
                }
            }
            object[nx*iy + ix] = thislabelmin;
        }
    }

    /* Re-label the groups before returning */
    maxlabel = relabel_image(on_pixels, maxlabel, equivs, object);
    //logverb("After final relabelling, %i labels were used.\n", maxlabel);
    if (pnobjects)
        *pnobjects = maxlabel;

    free(equivs);
    il_free(on_pixels);
    return 1;
}
