/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>

#include "brightstars.h"

static brightstar_t bs[] =
#include "brightstars-data.c"
    ;

int bright_stars_n() {
    return sizeof(bs) / sizeof(brightstar_t);
}

const brightstar_t* bright_stars_get(int starindex) {
    assert(starindex >= 0);
    assert(starindex < bright_stars_n());
    return bs + starindex;
}

