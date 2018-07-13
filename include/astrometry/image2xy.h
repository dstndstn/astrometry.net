/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef IMAGE2XY_H
#define IMAGE2XY_H

#include <stdint.h>

#include "astrometry/simplexy.h"

int image2xy_run(simplexy_t* s,
                 int downsample, int downsample_as_required);

#endif
