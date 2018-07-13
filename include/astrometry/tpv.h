/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef ANTPV_H
#define ANTPV_H

#include <stdio.h>
#include "astrometry/sip.h"
#include "astrometry/keywords.h"

#define N_PV_TERMS 40

// TPV (TAN + PV distortion) structure.
typedef struct {

    // A basic TAN header.
    tan_t wcstan;

    double pv1[N_PV_TERMS];
    double pv2[N_PV_TERMS];

} tpv_t;

tpv_t* tpv_new(void);
void   tpv_free(tpv_t* tpv);

void sip_copy(sip_t* dest, const sip_t* src);

// Set the given TPV wcs to the given TAN wcs.
void tpv_wrap_tan(const tan_t* tan, tpv_t* tpv);

double tpv_imagew(tpv_t* tpv);
double tpv_imageh(tpv_t* tpv);

// Pixels to RA,Dec in degrees.
void   tpv_pixelxy2radec(const tpv_t* tpv, double px, double py, double *a, double *d);

// Pixels to XYZ unit vector.
void   tpv_pixelxy2xyzarr(const tpv_t* tpv, double px, double py, double *xyz);

// RA,Dec in degrees to Pixels.
// Returns FALSE if the point is on the opposite side of the sphere (and hence the point
// does not project onto the tangent plane)
WarnUnusedResult
anbool tpv_radec2pixelxy(const tpv_t* tpv, double a, double d, double *px, double *py);

WarnUnusedResult
anbool tpv_radec2pixelxy_check(const tpv_t* tpv, double ra, double dec, double *px, double *py);

WarnUnusedResult
anbool tpv_xyzarr2pixelxy(const tpv_t* tpv, const double* xyz, double *px, double *py);

WarnUnusedResult
anbool tpv_xyz2pixelxy(const tpv_t* tpv, double x, double y, double z, double *px, double *py);

// Pixels to Intermediate World Coordinates in degrees.
void tpv_pixelxy2iwc(const tpv_t* tpv, double px, double py,
                     double *iwcx, double* iwcy);


double tpv_det_cd(const tpv_t* tpv);
// returns pixel scale in arcseconds/pixel (NOT arcsec^2)
double tpv_pixel_scale(const tpv_t* tpv);

// these take *relative* pixel coords (WRT crpix)
void   tpv_calc_inv_distortion(const tpv_t* tpv, double U, double V, double* u, double *v);
void   tpv_calc_distortion(const tpv_t* tpv, double u, double v, double* U, double *V);
      
// Applies forward TPV distortion to pixel coords.
// This applies the A,B matrix terms;
// This is the distortion applied in the pixel-to-RA,Dec direction.
//   (pix -> "un"distorted -> TAN -> RA,Dec)
void tpv_pixel_distortion(const tpv_t* tpv, double x, double y, double* X, double *Y);

// Reverses tpv_pixel_distortion;
// Applies "reverse" TPV distortion: the AP, BP matrices;
// This is the distortion applied in the RA,Dec-to-pixel direction:
//   (RA,Dec -> TAN -> undistorted -> pix)
void tpv_pixel_undistortion(const tpv_t* tpv, double x, double y, double* X, double *Y);

anbool tpv_xyzarr2iwc(const tpv_t* tpv, const double* xyz,
                      double* iwcx, double* iwcy);
anbool tpv_radec2iwc(const tpv_t* tpv, double ra, double dec,
                     double* iwcx, double* iwcy);

void tpv_iwc2pixelxy(const tpv_t* tpv, double iwcx, double iwcy,
                     double *px, double* py);

void tpv_iwc2radec(const tpv_t* tpv, double x, double y, double *p_ra, double *p_dec);

void   tpv_print(const tpv_t*);
void   tpv_print_to(const tpv_t*, FILE* fid);

// for python
void tpv_get_crval(const tpv_t* tpv, double* ra, double* dec);

double tpv_get_orientation(const tpv_t* tpv);

#endif
