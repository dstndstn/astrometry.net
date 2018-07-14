/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef WCS_RD2XY_H
#define WCS_RD2XY_H

#include "astrometry/bl.h"

int wcs_rd2xy(const char* wcsfn, int wcsext,
              const char* rdlsfn, const char* xylsfn,
              const char* racol, const char* deccol,
              int forcetan,
              int forcewcslib,
              il* fields);

#endif
