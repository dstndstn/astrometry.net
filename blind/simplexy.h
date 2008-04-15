/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute it
  and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty
  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA
*/
#ifndef SIMPLEXY_H
#define SIMPLEXY_H

int simplexy(float *image, int nx, int ny, float dpsf, float plim,
             float dlim, float saddle, int maxper, int maxnpeaks,
	     int maxsize, int halfbox,
             float *sigma, float *x, float *y, float *flux, int *npeaks, int verbose);

int simplexy_u8(unsigned char *image, int nx, int ny, float dpsf, float plim,
             float dlim, float saddle, int maxper, int maxnpeaks,
	     int maxsize, int halfbox,
             float *sigma, float *x, float *y, float *flux, int *npeaks, int verbose);

#endif

