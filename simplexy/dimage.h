/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg,
  Sam Roweis and Dustin Lang.

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

#ifndef DIMAGE_H
#define DIMAGE_H

#include <stdint.h>

// this is only really included here so that it can be tested :)
typedef uint16_t label_t;
#define LABEL_MAX UINT16_MAX
label_t collapsing_find_minlabel(label_t label, label_t *equivs);

int dfind2(const int* image, int nx, int ny, int* objectimg, int* p_nobjects);
int dfind2_u8(const unsigned char* image, int nx, int ny, int* objectimg, int* p_nobjects);

float dselip(unsigned long k, unsigned long n, const float *arr);
void dselip_cleanup(void);

int dsmooth(float *image, int nx, int ny, float sigma, float *smooth);

void dsmooth2(float *image, int nx, int ny, float sigma, float *smooth);
void dsmooth2_u8(uint8_t *image, int nx, int ny, float sigma, float *smooth);
void dsmooth2_i16(int16_t *image, int nx, int ny, float sigma, float *smooth);

int dobjects(float *image, int nx, int ny, float limit,
			 float dpsf, int *objects);

int dmask(float *image, int nx, int ny, float limit,
		  float dpsf, uint8_t* mask);

int dpeaks(float *image, int nx, int ny, int *npeaks, int *xcen,
           int *ycen, float sigma, float dlim, float saddle, int maxnpeaks,
           int smooth, int checkpeaks, float minpeak);

int dcen3x3(float *image, float *xcen, float *ycen);

int dsigma(float *image, int nx, int ny, int sp, int gridsize, float *sigma);
int dsigma_u8(uint8_t *image, int nx, int ny, int sp, int gridsize, float *sigma);

int dmedsmooth(const float *image, const uint8_t *masked,
               int nx, int ny, int halfbox, float *smooth);

int dallpeaks(float *image, int nx, int ny, int *objects, float *xcen,
              float *ycen, int *npeaks, float dpsf, float sigma,
			  float dlim, float saddle,
              int maxper, int maxnpeaks, float minpeak, int maxsize);
int dallpeaks_u8(uint8_t *image, int nx, int ny, int *objects, float *xcen,
				 float *ycen, int *npeaks, float dpsf, float sigma,
				 float dlim, float saddle,
				 int maxper, int maxnpeaks, float minpeak, int maxsize);
int dallpeaks_i16(int16_t *image, int nx, int ny, int *objects, float *xcen,
				  float *ycen, int *npeaks, float dpsf, float sigma,
				  float dlim, float saddle,
				  int maxper, int maxnpeaks, float minpeak, int maxsize);

#endif
