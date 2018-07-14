/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "simplexy-common.h"

/*
 * dcen3x3.c
 *
 * Find center of a star inside a 3x3 image
 *
 * COMMENTS:
 *   - Convention is to make the CENTER of the first pixel (0,0).
 * BUGS:
 *
 * Mike Blanton
 * 1/2006 */

int dcen3a(float f0, float f1, float f2, float *xcen,
           float* maxval
           ) {
    float s, d, aa, sod, kk;

    kk = (4.0/3.0);
    s = 0.5 * (f2 - f0);
    d = 2. * f1 - (f0 + f2);

    if (d <= 1.e-10*f0) {
        return (0);
    }

    aa = f1 + 0.5 * s * s / d;
    sod = s / d;
    if (!isnormal(aa) || !isnormal(s))
        return 0;
    (*xcen) = sod * (1. + kk * (0.25 * d / aa) * (1. - 4. * sod * sod)) + 1.;

    return (1);
}

/* given points at (0, f0), (1, f1), (2, f2), assume there is a
 quadratic passing through the three points; return the peak position
 and value of the quadratic:

 f = a x^2 + b x + c

 df/dx = 2ax + b = 0  =>  x* = -b/2a

 f* = a x*^2 + b x* + c

 */
int dcen3b(float f0, float f1, float f2, float *xcen
           //float *maxval
           ) {
    float a, b;
    /*
     f0 = c
     f1 = a + b + c
     f2 = 4a + 2b + c
     */
    a = 0.5 * (f2 - 2*f1 + f0);
    if (a == 0.0)
        return 0;
    b = f1 - a - f0;
    *xcen = -0.5 * b / a;
    if ((*xcen < 0.0) || (*xcen > 2.0))
        return 0;
    //*maxval = a * (*xcen)*(*xcen) + b * (*xcen) + f0;
    return 1;
}

#define dcen3 dcen3b

int dcen3x3(float *image, float *xcen, float *ycen)
{
    float mx0=0, mx1=0, mx2=0;
    float my0=0, my1=0, my2=0;
    float bx, by, mx , my;
    int badcen = 0;

    // Find the peak of the quadratic along each row...
    badcen += dcen3(image[0 + 3*0], image[1 + 3*0], image[2 + 3*0], &mx0);
    badcen += dcen3(image[0 + 3*1], image[1 + 3*1], image[2 + 3*1], &mx1);
    badcen += dcen3(image[0 + 3*2], image[1 + 3*2], image[2 + 3*2], &mx2);

    // Now along each column...
    badcen += dcen3(image[0 + 3*0], image[0 + 3*1], image[0 + 3*2], &my0);
    badcen += dcen3(image[1 + 3*0], image[1 + 3*1], image[1 + 3*2], &my1);
    badcen += dcen3(image[2 + 3*0], image[2 + 3*1], image[2 + 3*2], &my2);

    /* are we not okay? */
    if (badcen != 6)
        return 0;

    // Fit straight line to peak positions along the rows...
    /* x = (y-1) mx + bx */
    bx = (mx0 + mx1 + mx2) / 3.;
    mx = (mx2 - mx0) / 2.;

    // ... and along the columns...
    /* y = (x-1) my + by */
    by = (my0 + my1 + my2) / 3.;
    my = (my2 - my0) / 2.;

    /* find intersection */
    (*xcen) = (mx * (by - my - 1.) + bx) / (1. + mx * my);
    (*ycen) = ((*xcen) - 1.) * my + by;

    /* check that we are in the box */
    if (((*xcen) < 0.0) || ((*xcen) > 2.0) ||
        ((*ycen) < 0.0) || ((*ycen) > 2.0)){
        return (0);
    }

    /* check for nan's and inf's */
    if (!isnormal(*xcen) || !isnormal(*ycen))
        return 0;

    return (1);
} /* end dcen3x3 */
