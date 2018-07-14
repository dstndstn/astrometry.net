/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ctmf.h"
#include "cutest.h"

void test_simple_median(CuTest* tc) {
    int i=0;
    /* should come out all zeros */
    unsigned char test_data[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned char results[sizeof(test_data)];
    memset(results,9,sizeof(test_data));
    ctmf(test_data, results, 10,10, 10,10, 4 /*r*/, 1, 512*1024);
    for (i=0; i<sizeof(test_data); i++)
        CuAssertIntEquals(tc, 0, results[i]);
}
