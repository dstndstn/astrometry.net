/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "dimage.h"
#include "simplexy-common.h"
#include "log.h"

/*
 * dsigma.c
 *
 * Simple guess at the sky sigma
 *
 * Mike Blanton
 * 1/2006 */


#define IMGTYPE float
#define DSIGMA_SUFF
#include "dsigma.inc"
#undef DSIGMA_SUFF
#undef IMGTYPE

#define IMGTYPE uint8_t
#define DSIGMA_SUFF _u8
#include "dsigma.inc"
#undef IMGTYPE
#undef DSIGMA_SUFF

