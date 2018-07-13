/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "keywords.h"
#include "dimage.h"
#include "permutedsort.h"
#include "simplexy-common.h"

/*
 * dpeaks.c
 *
 * Find peaks in an image, for the purposes of deblending children.
 *
 * Mike Blanton
 * 1/2006 */

int dpeaks(float *image,
           int nx,
           int ny,
           int *npeaks,
           int *xcen,
           int *ycen,
           float sigma,    /* sky sigma */
           float dlim,     /* limiting distance */
           float saddle,   /* number of sigma for allowed saddle */
           int maxnpeaks,
           int smoothimage,
           int checkpeaks,
           float minpeak)
{
    int i, j, ip, jp, ist, jst, ind, jnd, highest, tmpnpeaks;
    float dx, dy, level;
        
    float *smooth = NULL;
    int *peaks = NULL;
    int *indx = NULL;
    int *object = NULL;
    int *keep = NULL;
    int *mask = NULL;
    int *fullxcen = NULL;
    int *fullycen = NULL;

    /* 1. smooth image */
    smooth = (float *) malloc(sizeof(float) * nx * ny);
    if (smoothimage) {
        dsmooth2(image, nx, ny, 1, smooth);
    } else {
        for (j = 0;j < ny;j++)
            for (i = 0;i < nx;i++)
                smooth[i + j*nx] = image[i + j*nx];
    }

    /* 2. find peaks (highest in the 3x3 neighbourhood) */
    peaks = (int *) malloc(sizeof(int) * nx * ny);
    *npeaks = 0;
    for (j = 1; j < ny - 1; j++) {
        jst = j - 1;
        jnd = j + 1;
        for (i = 1; i < nx - 1; i++) {
            if (smooth[i + j*nx] < minpeak)
                continue;
            ist = i - 1;
            ind = i + 1;
            highest = 1;
            for (ip = ist; ip <= ind; ip++)
                for (jp = jst; jp <= jnd; jp++)
                    if (smooth[ip + jp*nx] > smooth[i + j*nx])
                        highest = 0;
            if (highest) {
                peaks[*npeaks] = i + j * nx;
                (*npeaks)++;
            }
        }
    }

    // DEBUG
    for (i=0; i<(*npeaks); i++) {
        Unused float pk = smooth[peaks[i]];
        assert((peaks[i] % nx) >= 1);
        assert((peaks[i] % nx) <= (nx-2));
        assert((peaks[i] / nx) >= 1);
        assert((peaks[i] / nx) <= (ny-2));
        assert(pk >= minpeak);
        assert(pk >= smooth[peaks[i]-1]);
        assert(pk >= smooth[peaks[i]+1]);
        assert(pk >= smooth[peaks[i]-nx]);
        assert(pk >= smooth[peaks[i]+nx]);
        assert(pk >= smooth[peaks[i]+nx+1]);
        assert(pk >= smooth[peaks[i]+nx-1]);
        assert(pk >= smooth[peaks[i]-nx+1]);
        assert(pk >= smooth[peaks[i]-nx-1]);
    }

    /* 2. sort peaks */
    indx = realloc(peaks, sizeof(int) * (*npeaks));
    peaks = NULL;
    permuted_sort(smooth, sizeof(float), compare_floats_desc, indx, *npeaks);

    // DEBUG
    for (i=0; i<(*npeaks); i++) {
        Unused float pk = smooth[indx[i]];
        assert((indx[i] % nx) >= 1);
        assert((indx[i] % nx) <= (nx-2));
        assert((indx[i] / nx) >= 1);
        assert((indx[i] / nx) <= (ny-2));
        assert(pk >= minpeak);
        assert(pk >= smooth[indx[i]-1]);
        assert(pk >= smooth[indx[i]+1]);
        assert(pk >= smooth[indx[i]-nx]);
        assert(pk >= smooth[indx[i]+nx]);
        assert(pk >= smooth[indx[i]+nx+1]);
        assert(pk >= smooth[indx[i]+nx-1]);
        assert(pk >= smooth[indx[i]-nx+1]);
        assert(pk >= smooth[indx[i]-nx-1]);
    }
    for (i=1; i<(*npeaks); i++) {
        assert(smooth[indx[i-1]] >= smooth[indx[i]]);
    }

    if ((*npeaks) > maxnpeaks)
        *npeaks = maxnpeaks;

    fullxcen = (int *) malloc((*npeaks) * sizeof(int));
    fullycen = (int *) malloc((*npeaks) * sizeof(int));
    for (i = 0;i < (*npeaks);i++) {
        fullxcen[i] = indx[i] % nx;
        fullycen[i] = indx[i] / nx;
    }
    FREEVEC(indx);

    // DEBUG
    for (i = 0;i < (*npeaks);i++) {
        assert(fullxcen[i] >= 1);
        assert(fullxcen[i] <= nx-2);
        assert(fullycen[i] >= 1);
        assert(fullycen[i] <= ny-2);
    }


    /* 3. trim close peaks and joined peaks */
    mask = (int *) malloc(sizeof(int) * nx * ny);
    object = (int *) malloc(sizeof(int) * nx * ny);
    keep = (int *) malloc(sizeof(int) * (*npeaks));
    for (i = (*npeaks) - 1;i >= 0;i--) {
        keep[i] = 1;

        if (checkpeaks) {
            /* look for peaks joined by a high saddle to brighter peaks */
            level = (smooth[ fullxcen[i] + fullycen[i] * nx] - saddle * sigma);
            if (level < sigma)
                level = sigma;
            if (level > 0.99*smooth[ fullxcen[i] + fullycen[i] * nx]) 
                level= 0.99*smooth[ fullxcen[i] + fullycen[i] * nx]; 
            for (jp = 0;jp < ny;jp++)
                for (ip = 0;ip < nx;ip++)
                    mask[ip + jp*nx] = smooth[ip + jp * nx] > level;
            dfind2(mask, nx, ny, object, NULL);
            for (j = i - 1;j >= 0;j--)
                if (object[ fullxcen[j] + fullycen[j]*nx] ==
                    object[ fullxcen[i] + fullycen[i]*nx] ||
                    object[ fullxcen[i] + fullycen[i]*nx] == -1 ) {
                    keep[i] = 0;
                }
        }

        /* look for close peaks */
        for (j = i - 1;j >= 0;j--) {
            dx = (float)(fullxcen[j] - fullxcen[i]);
            dy = (float)(fullycen[j] - fullycen[i]);
            if (dx*dx + dy*dy < dlim*dlim)
                keep[i] = 0;
        }
    }

    // Grab just the keepers.
    tmpnpeaks = 0;
    for (i=0; i < (*npeaks); i++) {
        if (!keep[i])
            continue;
        xcen[tmpnpeaks] = fullxcen[i];
        ycen[tmpnpeaks] = fullycen[i];
        tmpnpeaks++;
        if (tmpnpeaks >= maxnpeaks)
            break;
    }
    (*npeaks) = tmpnpeaks;

    FREEVEC(smooth);
    FREEVEC(keep);
    FREEVEC(object);
    FREEVEC(mask);
    FREEVEC(fullxcen);
    FREEVEC(fullycen);

    return (1);
} /* end dpeaks */
