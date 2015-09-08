/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef CFITSIO_UTILS_H
#define CFITSIO_UTILS_H

// from cfitsio
#include "fitsio.h"

#include "astrometry/bl.h"
#include "astrometry/errors.h"

static void cfitserr(int status) {
    sl* msgs = sl_new(4);
    char errmsg[FLEN_ERRMSG];
    int i;
    // pop the cfitsio error message stack...
    fits_get_errstatus(status, errmsg);
    sl_append(msgs, errmsg);
    while (fits_read_errmsg(errmsg))
        sl_append(msgs, errmsg);
    // ... and push it onto the astrometry.net error message stack... sigh.
    for (i=sl_size(msgs)-1; i>=0; i--)
        ERROR("%s", sl_get(msgs, i));
    sl_free2(msgs);
}

#define CFITS_CHECK(msg, ...) \
do { \
if (status) { \
cfitserr(status); \
ERROR(msg, ##__VA_ARGS__); \
goto bailout; \
} \
} while(0)

#endif
