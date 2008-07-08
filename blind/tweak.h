/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Keir Mierle, David W. Hogg, Sam Roweis and Dustin Lang.

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

#ifndef _TWEAK_INTERNAL_H
#define _TWEAK_INTERNAL_H

#include "an-bool.h"
#include "kdtree.h"
#include "bl.h"
#include "sip.h"
#include "starutil.h"
#include "starxy.h"

enum opt_flags {
	OPT_CRVAL         = 1,
	OPT_CRPIX         = 2,
	OPT_CD            = 4,
	OPT_SIP           = 8,
	OPT_SIP_INVERSE   = 16,
	OPT_SHIFT         = 32 };

// These flags represent the work already done on a tweak problem
enum tweak_flags {
	TWEAK_HAS_SIP              = 1,
	TWEAK_HAS_IMAGE_XY         = 2,
	TWEAK_HAS_IMAGE_XYZ        = 4,
	TWEAK_HAS_IMAGE_AD         = 8,
	TWEAK_HAS_REF_XY           = 16, 
	TWEAK_HAS_REF_XYZ          = 32, 
	TWEAK_HAS_REF_AD           = 64, 
	TWEAK_HAS_AD_BAR_AND_R     = 128,
	TWEAK_HAS_CORRESPONDENCES  = 256,
	TWEAK_HAS_RUN_OPT          = 512,
	TWEAK_HAS_RUN_RANSAC_OPT   = 1024,
	TWEAK_HAS_COARSLY_SHIFTED  = 2048,
	TWEAK_HAS_FINELY_SHIFTED   = 4096,
	TWEAK_HAS_REALLY_FINELY_SHIFTED   = 8192,
	TWEAK_HAS_HEALPIX_PATH     = 16384,
	TWEAK_HAS_LINEAR_CD        = 32768
};
// FIXME add a method to print out the state in a readable way

typedef struct tweak_s {
	sip_t* sip;
	unsigned int state; // bitfield of tweak_flags

	// For sources in the image
	int n;
	double *a;
	double *d;
	double *x;
	double *y;
	double *xyz;

	// Center of field estimate
	double a_bar;  // degrees
	double d_bar;  // degrees
	double radius; // radians (genius!)

	// Cached values of sources in the catalog
	int n_ref;
	double *x_ref;
	double *y_ref;
	double *a_ref;
	double *d_ref;
	double *xyz_ref;

	// Correspondences
	il* image;
	il* ref;
	dl* dist2;
	dl* weight;

	// Correspondence subsets for RANSAC
	il* maybeinliers;
	il* bestinliers;
	il* included;

	int opt_flags;
//	int n;
//	int m;
//	double parameters[500];
	double err;

	// Size of Hough space for shift
	double mindx, mindy, maxdx, maxdy;

	// Size of last run shift operation
	double xs, ys;

	// Trees used for finding correspondences
	kdtree_t* kd_image;
	kdtree_t* kd_ref;

	// path for raw star data
	char* hppath;
	// Nside for healpixed star kdtree... necessary?
	int Nside;
	// star jitter, in arcseconds.
	double jitter;

	// (computed from jitter); star jitter in distance-squared on the unit sphere.
	double jitterd2;

	// Weighted or unweighted fit?
	int weighted_fit;

    bool verbose;
    bool quiet;
} tweak_t;

tweak_t* tweak_new();
void tweak_init(tweak_t*);
void tweak_push_wcs_tan(tweak_t* t, tan_t* wcs);
void tweak_push_ref_xyz(tweak_t* t, double* xyz, int n);
unsigned int tweak_advance_to(tweak_t* t, unsigned int flag);
void tweak_clear(tweak_t* t);
void tweak_dump_ascii(tweak_t* t);
void tweak_skip_shift(tweak_t* t);

//void tweak_push_image_xy(tweak_t* t, const double* x, const double *y, int n);
void tweak_push_image_xy(tweak_t* t, const starxy_t* xy);

void tweak_push_hppath(tweak_t* t, char* hppath);
void tweak_push_ref_ad(tweak_t* t, double* a, double *d, int n);
void tweak_print_state(tweak_t* t);
void tweak_go_to(tweak_t* t, unsigned int flag);
void tweak_clear_correspondences(tweak_t* t);
void tweak_clear_on_sip_change(tweak_t* t);
void tweak_clear_image_ad(tweak_t* t);
void tweak_clear_ref_xy(tweak_t* t);
void tweak_clear_image_xyz(tweak_t* t);
void tweak_free(tweak_t* t);

void tweak_print_rms_curve(tweak_t* t);

#endif
