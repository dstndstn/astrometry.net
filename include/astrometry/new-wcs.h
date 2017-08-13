/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/
#ifndef NEW_WCS_H
#define NEW_WCS_H

#include "astrometry/an-bool.h"

int new_wcs(const char* infn, int extension,
            const char* wcsfn, const char* outfn,
            anbool include_data);

#endif

