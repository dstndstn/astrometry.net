/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg, Sam Roweis
  and Dustin Lang.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
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

