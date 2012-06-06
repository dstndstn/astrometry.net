/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef CFITSIO_UTILS_H
#define CFITSIO_UTILS_H

#include "bl.h"
#include "fitsio.h"
#include "errors.h"

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
