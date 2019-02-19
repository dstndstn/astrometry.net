/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>

#include "cutest.h"
#include "hd.h"
#include "starutil.h"
#include "ioutils.h"
#include "constellation-boundaries.h"

void test_bdy_1(CuTest* tc) {
    // "Suhail" (Lambda Vela)
    int con = constellation_containing(136.9992, -43.4325);
    printf("Got: %i\n", con);
    CuAssertIntEquals(tc, CON_VEL, con);

    // Naos (xi Pup?)
    con = constellation_containing(120.8963, -40.0033);
    printf("Got: %i\n", con);
    CuAssertIntEquals(tc, CON_PUP, con);
}
