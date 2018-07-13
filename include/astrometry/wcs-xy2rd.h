/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef WCS_XY2RD_H
#define WCS_XY2RD_H

#include "astrometry/bl.h"

int wcs_xy2rd(const char* wcsfn, int wcsext,
              const char* xylsfn, const char* rdlsfn,
              const char* xcol, const char* ycol,
              int forcetan,
              int forcewcslib,
              il* fields);

#endif
