/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef WCS_RESAMPLE_H
#define WCS_RESAMPLE_H

#include "anwcs.h"

int resample_wcs_files(const char* infitsfn, int infitsext,
					   const char* inwcsfn, int inwcsext,
					   const char* outwcsfn, int outwcsext,
					   const char* outfitsfn, int lanczos_order,
                       int zero_inf);

int resample_wcs(const anwcs_t* inwcs, const float* inimg, int inW, int inH,
				 const anwcs_t* outwcs, float* outimg, int outW, int outH,
				 int weighted, int lanczos_order);

int resample_wcs_rgba(const anwcs_t* inwcs, const unsigned char* inimg,
					  int inW, int inH,
					  const anwcs_t* outwcs, unsigned char* outimg,
					  int outW, int outH);

#endif

