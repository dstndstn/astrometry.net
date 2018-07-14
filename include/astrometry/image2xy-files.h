/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef IMAGE2XY_FILES_H
#define IMAGE2XY_FILES_H

#include <stdint.h>

#include "astrometry/an-bool.h"
#include "astrometry/simplexy.h"

/**
 Reads an input FITS image (possibly multi-HDU), runs simplexy on each
 image and places the results in an output file containing a FITS BINTABLE.

 If you want to look at just a single HDU, set "extension".  Note
 that it follows the QFITS convention that the primary extension is 0,
 the first extension is 1, etc.  This is different than the CFITSIO
 convention which is 1-based: 1 is the primary extension, 2 is the
 first extension, etc.
 */
int image2xy_files(const char* infn, const char* outfn,
                   anbool do_u8, int downsample,
                   int downsample_as_required,
                   int extension, int plane,
                   simplexy_t* params);

#endif
