/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2010 Dustin Lang.

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

#ifndef WCS_RESAMPLE_H
#define WCS_RESAMPLE_H

#include "anwcs.h"

int resample_wcs_files(const char* infitsfn, int infitsext,
					   const char* inwcsfn, int inwcsext,
					   const char* outwcsfn, int outwcsext,
					   const char* outfitsfn, int lanczos_order);

int resample_wcs(const anwcs_t* inwcs, const float* inimg, int inW, int inH,
				 const anwcs_t* outwcs, float* outimg, int outW, int outH,
				 int weighted, int lanczos_order);

int resample_wcs_rgba(const anwcs_t* inwcs, const unsigned char* inimg,
					  int inW, int inH,
					  const anwcs_t* outwcs, unsigned char* outimg,
					  int outW, int outH);

#endif

