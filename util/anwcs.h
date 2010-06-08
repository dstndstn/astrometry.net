/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

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

#ifndef ANWCSLIB_H
#define ANWCSLIB_H

#include "sip.h"
#include "an-bool.h"

/** Interface to Mark Calabretta's wcslib, if available, and
 Astrometry.net's TAN/SIP implementation. */

#define ANWCS_TYPE_WCSLIB 1
#define ANWCS_TYPE_SIP 2

struct anwcs_t {
	/**
	 If type == ANWCS_TYPE_WCSLIB:
	   data is a private struct containing a wcslib  "struct wcsprm*".
	 If type == ANWCS_TYPE_SIP:
	   data is a "sip_t*"
	 */
	int type;
	void* data;
};
typedef struct anwcs_t anwcs_t;


anwcs_t* anwcs_open(const char* filename, int ext);

anwcs_t* anwcs_open_wcslib(const char* filename, int ext);

anwcs_t* anwcs_open_sip(const char* filename, int ext);

anwcs_t* anwcs_new_sip(const sip_t* sip);

anwcs_t* anwcs_new_tan(const tan_t* tan);

int anwcs_write(const anwcs_t* wcs, const char* filename);

int anwcs_write_to(const anwcs_t* wcs, FILE* fid);

int anwcs_radec2pixelxy(const anwcs_t* wcs, double ra, double dec, double* px, double* py);

int anwcs_pixelxy2radec(const anwcs_t* wcs, double px, double py, double* ra, double* dec);

int anwcs_pixelxy2xyz(const anwcs_t* wcs, double px, double py, double* xyz);

int anwcs_xyz2pixelxy(const anwcs_t* wcs, const double* xyz, double *px, double *py);

bool anwcs_radec_is_inside_image(const anwcs_t* wcs, double ra, double dec);

void anwcs_get_radec_bounds(const anwcs_t* wcs, int stepsize,
							double* pramin, double* pramax,
							double* pdecmin, double* pdecmax);

void anwcs_print(const anwcs_t* wcs, FILE* fid);

// Center and radius of the field.
// RA,Dec,radius in degrees.
int anwcs_get_radec_center_and_radius(anwcs_t* anwcs,
									  double* p_ra, double* p_dec, double* p_radius);

void anwcs_walk_image_boundary(const anwcs_t* wcs, double stepsize,
							   void (*callback)(const anwcs_t* wcs, double x, double y, double ra, double dec, void* token),
							   void* token);

double anwcs_imagew(const anwcs_t* anwcs);
double anwcs_imageh(const anwcs_t* anwcs);

void anwcs_set_size(anwcs_t* anwcs, int W, int H);

// Approximate pixel scale, in arcsec/pixel, at the reference point.
double anwcs_pixel_scale(const anwcs_t* anwcs);

void anwcs_free(anwcs_t* wcs);


#endif
