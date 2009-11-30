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

int dfluxes(float *image, float *templates, float *weights, int nx, int ny,
            float *xcen, float *ycen, int nchild, float *children,
            float sigma);
int dweights(float *image, float *invvar, int nx, int ny, int ntemplates,
             float *templates, int nonneg, float *weights);

int dfind2(const int* image, int nx, int ny, int* objectimg, int* p_nobjects);
int dfind2_u8(const unsigned char* image, int nx, int ny, int* objectimg, int* p_nobjects);

float dselip(unsigned long k, unsigned long n, const float *arr);
void dselip_cleanup();

int dsmooth(float *image, int nx, int ny, float sigma, float *smooth);

void dsmooth2(float *image, int nx, int ny, float sigma, float *smooth);
void dsmooth2_u8(uint8_t *image, int nx, int ny, float sigma, float *smooth);
void dsmooth2_i16(int16_t *image, int nx, int ny, float sigma, float *smooth);

/*
 int dobjects(float *image, float *smooth, int nx, int ny,
 float dpsf, float plim, int *objects);
 */
int dobjects(float *image, int nx, int ny, float limit,
			 float dpsf, int *objects);

int dmask(float *image, int nx, int ny, float limit,
		  float dpsf, uint8_t* mask);

int dnonneg(float *xx, float *invcovar, float *bb, float offset,
            int nn, float tolerance, int maxiter, int *niter, float *chi2,
            int verbose);
int dpeaks(float *image, int nx, int ny, int *npeaks, int *xcen,
           int *ycen, float sigma, float dlim, float saddle, int maxnpeaks,
           int smooth, int checkpeaks, float minpeak);
int dcentral(float *image, int nx, int ny, int npeaks, float *xcen,
             float *ycen, int *central, float sigma, float dlim,
             float saddle, int maxnpeaks);
int dmsmooth(float *image, int nx, int ny, int box, float *smooth);
int deblend(float *image,
            float *invvar,
            int nx,
            int ny,
            int *nchild,
            int *xcen,
            int *ycen,
            float *cimages,
            float *templates,
            float sigma,
            float dlim,
            float tsmooth,   /* smoothing of template */
            float tlimit,    /* lowest template value in units of sigma */
            float tfloor,    /* vals < tlimit*sigma are set to tfloor*sigma */
            float saddle,    /* number of sigma for allowed saddle */
            float parallel,  /* how parallel you allow templates to be */
            int maxnchild,
            float minpeak,
            int starstart,
            float *psf,
            int pnx,
            int pny,
            int dontsettemplates);
int dcen3x3(float *image, float *xcen, float *ycen);

int dsigma(float *image, int nx, int ny, int sp, int gridsize, float *sigma);
int dsigma_u8(uint8_t *image, int nx, int ny, int sp, int gridsize, float *sigma);

int dmedsmooth(float *image,
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

int dtemplates(float *image, int nx, int ny, int *ntemplates, int *xcen,
               int *ycen, float *templates, float sigma, float parallel);

#endif
