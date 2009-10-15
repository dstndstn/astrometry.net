/*
 This file is part of the Astrometry.net suite.
 Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg,
 Sam Roweis and Dustin Lang.

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
#include <assert.h>
#include <sys/param.h>

#include "dimage.h"
#include "permutedsort.h"
#include "simplexy-common.h"

/*
 * dallpeaks.c
 *
 * Take image and list of objects, and produce list of all peaks (and
 * which object they are in).
 *
 * BUGS:
 *   - Returns no error analysis if the centroid sux.
 *   - Uses dead-reckon pixel center if dcen3x3 sux.
 *   - No out-of-memory checks
 *
 * Mike Blanton
 * 1/2006 */

/* Finds all peaks in the image by cutting a bounding box out around
 each one */

#define IMGTYPE float
#define SUFFIX
#include "dallpeaks.inc"
#undef SUFFIX
#undef IMGTYPE

#define IMGTYPE uint8_t
#define SUFFIX _u8
#include "dallpeaks.inc"
#undef IMGTYPE
#undef SUFFIX

