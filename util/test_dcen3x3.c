/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cutest.h"
#include "dimage.h"

void test_dcen3x3_1(CuTest* tc) {
    float image[] = {
        0, 1, 0,
        1, 2, 1,
        0, 1, 0 };
    float xc, yc;
    int rtn;

    rtn = dcen3x3(image, &xc, &yc);
    CuAssertIntEquals(tc, 1, rtn);
    CuAssertDblEquals(tc, 1.0, xc, 1e-6);
    CuAssertDblEquals(tc, 1.0, yc, 1e-6);
}

void test_dcen3x3_2(CuTest* tc) {
    float image[] = {
        1, 2, 1,
        2, 3, 2,
        1, 2, 1 };
    float xc, yc;
    int rtn;

    rtn = dcen3x3(image, &xc, &yc);
    CuAssertIntEquals(tc, 1, rtn);
    CuAssertDblEquals(tc, 1.0, xc, 1e-6);
    CuAssertDblEquals(tc, 1.0, yc, 1e-6);
}

void test_dcen3x3_3(CuTest* tc) {
    float a1 = -3.;
    float XX = 1.1;

    float a2 = -3.;
    float YY = 1.1;

    float c = 40.;

    float image[9];
    float xc, yc;
    int rtn;
    int x,y;

    for (y=0; y<3; y++) {
        for (x=0; x<3; x++) {
            float dx = (x - XX);
            float dy = (y - YY);
            image[y*3 + x] = a1 * dx*dx + a2 * dy*dy + c;
        }
        printf("%i %i %i\n",
               (int)image[y*3+0],
               (int)image[y*3+1],
               (int)image[y*3+2]);
    }

    rtn = dcen3x3(image, &xc, &yc);
    CuAssertIntEquals(tc, 1, rtn);
    CuAssertDblEquals(tc, XX, xc, 1e-6);
    CuAssertDblEquals(tc, YY, yc, 1e-6);
    printf("(%g,%g) -> (%g,%g)\n", XX, YY, xc, yc);
}

