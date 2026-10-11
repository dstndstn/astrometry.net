/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stddef.h>

#include "dimage.h"
#include "cutest.h"

void test_dpeaks_single_peak(CuTest* tc) {
    float image[49] = {0};
    int n, x[4], y[4];
    image[3 + 3*7] = 10.0;
    dpeaks(image, 7, 7, &n, x, y, 1.0, 0.0, 2.0, 4, 0, 1, 1.0);
    CuAssertIntEquals(tc, 1, n);
    CuAssertIntEquals(tc, 3, x[0]);
    CuAssertIntEquals(tc, 3, y[0]);
}

void test_dpeaks_close_peak(CuTest* tc) {
    float image[49] = {0};
    int n, x[4], y[4];
    image[2 + 3*7] = 10.0;
    image[4 + 3*7] = 9.0;
    dpeaks(image, 7, 7, &n, x, y, 1.0, 3.0, 2.0, 4, 0, 1, 1.0);
    CuAssertIntEquals(tc, 1, n);
    CuAssertIntEquals(tc, 2, x[0]);
    CuAssertIntEquals(tc, 3, y[0]);
}

void test_dpeaks_rejected_candidate_distance(CuTest* tc) {
    float image[63] = {0};
    int n, x[4], y[4];
    image[2 + 3*9] = 10.0;
    image[4 + 3*9] = 9.0;
    image[6 + 3*9] = 8.0;
    /* The middle peak rejects the faintest, even though it is also rejected. */
    dpeaks(image, 9, 7, &n, x, y, 1.0, 2.5, 2.0, 4, 0, 1, 1.0);
    CuAssertIntEquals(tc, 1, n);
    CuAssertIntEquals(tc, 2, x[0]);
    CuAssertIntEquals(tc, 3, y[0]);
}

void test_dpeaks_high_saddle(CuTest* tc) {
    float image[49] = {0};
    int n, x[4], y[4];
    image[2 + 3*7] = 10.0;
    image[3 + 3*7] = 8.0;
    image[4 + 3*7] = 9.0;
    dpeaks(image, 7, 7, &n, x, y, 1.0, 0.0, 2.0, 4, 0, 1, 1.0);
    CuAssertIntEquals(tc, 1, n);
    CuAssertIntEquals(tc, 2, x[0]);
    CuAssertIntEquals(tc, 3, y[0]);
}

void test_dpeaks_low_saddle(CuTest* tc) {
    float image[49] = {0};
    int n, x[4], y[4];
    image[2 + 3*7] = 10.0;
    image[3 + 3*7] = 1.0;
    image[4 + 3*7] = 9.0;
    dpeaks(image, 7, 7, &n, x, y, 1.0, 0.0, 2.0, 4, 0, 1, 1.0);
    CuAssertIntEquals(tc, 2, n);
    CuAssertIntEquals(tc, 2, x[0]);
    CuAssertIntEquals(tc, 4, x[1]);
    CuAssertIntEquals(tc, 3, y[0]);
    CuAssertIntEquals(tc, 3, y[1]);
}

void test_dpeaks_saddle_check_disabled(CuTest* tc) {
    float image[49] = {0};
    int n, x[4], y[4];
    image[2 + 3*7] = 10.0;
    image[3 + 3*7] = 8.0;
    image[4 + 3*7] = 9.0;
    dpeaks(image, 7, 7, &n, x, y, 1.0, 0.0, 2.0, 4, 0, 0, 1.0);
    CuAssertIntEquals(tc, 2, n);
    CuAssertIntEquals(tc, 2, x[0]);
    CuAssertIntEquals(tc, 4, x[1]);
}
