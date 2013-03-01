/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Michael Blanton, Keir Mierle, David W. Hogg,
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

#ifndef IMAGE2XY_FILES_H
#define IMAGE2XY_FILES_H

#include <stdint.h>

#include "an-bool.h"
#include "simplexy.h"

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
