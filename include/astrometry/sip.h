/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef ANSIP_H
#define ANSIP_H

#include <stdio.h>
#include "astrometry/an-bool.h"
#include "astrometry/keywords.h"

#define SIP_MAXORDER 10

// WCS TAN header.
typedef struct {

    // World coordinate of the tangent point, in ra,dec.
    double crval[2];

    // Tangent point location in pixel (CCD) coordinates
    // This may not be in the image; consider a telescope with an array of
    // CCD's, where the tangent point is in one or none of the CCD's.
    double crpix[2];

    // Matrix for the linear transformation of relative pixel coordinates
    // (u,v) onto "intermediate world coordinates", which are in degrees
    // (x,y). The x,y coordinates are on the tangent plane. If the SIP
    // terms are all zero, then the equation to get from pixel coordinates
    // to intermediate world coordinates is:
    //
    //   u = pixel_x - crpix0
    //   v = pixel_y - crpix1
    // 
    //   x  = [cd00 cd01] * u
    //   y    [cd10 cd11]   v
    // 
    // where x,y are in intermediate world coordinates (i.e. x points
    // along negative ra and y points to positive dec) and u,v are in pixel
    // coordinates.
    double cd[2][2];

    // size of the image in pixels.  Not strictly part of the WCS, but useful!
    double imagew;
    double imageh;

    // SIN projection rather than TAN.
    anbool sin;

} tan_t;

// Flat structure for minimal SIP wcs. This structure should contain enough
// information to effectively represent any image, provided it is possible to
// convert that image's projecton to a TAN projection and the distortion to SIP
// distortion.
typedef struct {

    // A basic TAN header.
    tan_t wcstan;

    // Forward SIP coefficients
    // The transformation from relative pixel coordinates to intermediate
    // world coordinates[1] is:
    // 
    //   x  = [cd00 cd01] * (u + f(u,v))          x,y are intermediate coordinates on the sky (in deg)
    //   y    [cd10 cd11]   (v + g(u,v))            which are just a "translation" away from final WCS
    //                                            u,v are original (unwarped) pixel coordinates
    // where
    //                                       p    q
    //   U = u + f(u,v) = u + SUM a[p][q] * u  * v  ,  p+q <= a_order
    //                        p,q
    //
    //                                       p    q
    //   V = v + g(u,v) = v + SUM b[p][q] * u  * v  ,  p+q <= b_order
    //                        p,q
    // 
    // [1] The SIP convention for representing distortion in FITS image
    // headers. D. L. Shupe, M.Moshir, J. Li, D. Makovoz, R. Narron, R. N.
    // Hook. Astronomical Data Analysis Software and Systems XIV.
    // http://ssc.spitzer.caltech.edu/postbcd/doc/shupeADASS.pdf
    //
    // Note: These matricies are larger than they strictly need to be
    // because aij = 0 if i+j > a_order and similarily for b.
    // 
    // Note: The convention for indicating that no SIP polynomial is
    // present is to simply set [ab]_order to zero.
    int a_order, b_order;
    double a[SIP_MAXORDER][SIP_MAXORDER];
    double b[SIP_MAXORDER][SIP_MAXORDER];

    // Inverse SIP coefficients
    // To convert from world coordinates back into image coordinates, the
    // inverse transformation may be stored. To convert from intermediate
    // world coordinates, first we calculate the linear pixel coordinates:
    // 
    //                   -1
    //    U = [cd00 cd01]    * x
    //    V   [cd10 cd11]      y
    //
    // Then, the original pixel coordinates are computed as:
    // 
    //                            p    q
    //   u  = U + SUM ap[p][q] * U  * V  ,  p+q <= ap_order
    //            p,q
    //
    //                            p    q
    //   v  = V + SUM bp[p][q] * U  * V  ,  p+q <= ap_order
    //            p,q
    // 
    // Note: ap_order does not necessarily equal a_order, because the
    // inverse of a nth-order polynomial may be of higer order than n.
    // 
    // Note: The convention for indicating that no inverse SIP polynomial
    // is present is to simply set [ab]p_order to zero.
    int ap_order, bp_order;
    double ap[SIP_MAXORDER][SIP_MAXORDER];
    double bp[SIP_MAXORDER][SIP_MAXORDER];
} sip_t;

sip_t* sip_create(void);
void   sip_free(sip_t* sip);

void sip_copy(sip_t* dest, const sip_t* src);

// Set the given SIP wcs to the given TAN wcs.
void sip_wrap_tan(const tan_t* tan, sip_t* sip);

double sip_imagew(sip_t* sip);
double sip_imageh(sip_t* sip);

// Pixels to RA,Dec in degrees.
void   sip_pixelxy2radec(const sip_t* sip, double px, double py, double *a, double *d);

// Pixels to XYZ unit vector.
void   sip_pixelxy2xyzarr(const sip_t* sip, double px, double py, double *xyz);

// RA,Dec in degrees to Pixels.
// Returns FALSE if the point is on the opposite side of the sphere (and hence the point
// does not project onto the tangent plane)
WarnUnusedResult
anbool sip_radec2pixelxy(const sip_t* sip, double a, double d, double *px, double *py);

WarnUnusedResult
anbool sip_radec2pixelxy_check(const sip_t* sip, double ra, double dec, double *px, double *py);

WarnUnusedResult
anbool sip_xyzarr2pixelxy(const sip_t* sip, const double* xyz, double *px, double *py);

WarnUnusedResult
anbool sip_xyz2pixelxy(const sip_t* sip, double x, double y, double z, double *px, double *py);

// Pixels to Intermediate World Coordinates in degrees.
void sip_pixelxy2iwc(const sip_t* sip, double px, double py,
                     double *iwcx, double* iwcy);


double tan_det_cd(const tan_t* tan);
double sip_det_cd(const sip_t* sip);
// returns pixel scale in arcseconds/pixel (NOT arcsec^2)
double tan_pixel_scale(const tan_t* tn);
double sip_pixel_scale(const sip_t* sip);

// these take *relative* pixel coords (WRT crpix)
void   sip_calc_inv_distortion(const sip_t* sip, double U, double V, double* u, double *v);
void   sip_calc_distortion(const sip_t* sip, double u, double v, double* U, double *V);
      
// Applies forward SIP distortion to pixel coords.
// This applies the A,B matrix terms;
// This is the distortion applied in the pixel-to-RA,Dec direction.
//   (pix -> "un"distorted -> TAN -> RA,Dec)
void sip_pixel_distortion(const sip_t* sip, double x, double y,
                          double *p_x, double *p_y);

// Reverses sip_pixel_distortion;
// Applies "reverse" SIP distortion: the AP, BP matrices;
// This is the distortion applied in the RA,Dec-to-pixel direction:
//   (RA,Dec -> TAN -> undistorted -> pix)
void sip_pixel_undistortion(const sip_t* sip, double x, double y,
                            double *p_x, double *p_y);

// Pixels to XYZ unit vector.
void   tan_pixelxy2xyzarr(const tan_t* tan, double px, double py, double *xyz);

// Pixels to RA,Dec in degrees.
void   tan_pixelxy2radec(const tan_t* wcs_tan, double px, double py, double *ra, double *dec);

// Pixels to RA,Dec in degrees.
void   tan_pixelxy2radecarr(const tan_t* wcs_tan, double px, double py, double *radec);

// RA,Dec in degrees to Pixels.
// Returns FALSE if the point is on the opposite side of the sphere.
WarnUnusedResult
anbool   tan_radec2pixelxy(const tan_t* wcs_tan, double ra, double dec, double *px, double *py);

// xyz unit vector to Pixels.
// Returns TRUE if all is good.
WarnUnusedResult
anbool   tan_xyzarr2pixelxy(const tan_t* wcs_tan, const double* xyz, double *px, double *py);

void tan_iwc2pixelxy(const tan_t* tan, double iwcx, double iwcy,
                     double *px, double* py);
void tan_iwc2xyzarr(const tan_t* tan, double x, double y, double *xyz);

void tan_iwc2radec(const tan_t* tan, double x, double y, double *p_ra, double *p_dec);

/**
 Subtracts off CRPIX, multiplies by CD matrix.
 Results are in degrees.
 */
void tan_pixelxy2iwc(const tan_t* tan, double px, double py, double *iwcx, double* iwcy);

anbool tan_xyzarr2iwc(const tan_t* tan, const double* xyz,
                      double* iwcx, double* iwcy);
anbool tan_radec2iwc(const tan_t* tan, double ra, double dec,
                     double* iwcx, double* iwcy);

anbool sip_xyzarr2iwc(const sip_t* sip, const double* xyz,
                      double* iwcx, double* iwcy);
anbool sip_radec2iwc(const sip_t* sip, double ra, double dec,
                     double* iwcx, double* iwcy);

void sip_iwc2pixelxy(const sip_t* sip, double iwcx, double iwcy,
                     double *px, double* py);

void sip_iwc2radec(const sip_t* sip, double x, double y, double *p_ra, double *p_dec);

void   sip_print(const sip_t*);
void   sip_print_to(const sip_t*, FILE* fid);

void tan_print(const tan_t* tan);
void tan_print_to(const tan_t* tan, FILE* f);

// for python
void sip_get_crval(const sip_t* sip, double* ra, double* dec);

double tan_get_orientation(const tan_t* tan);
double sip_get_orientation(const sip_t* sip);

#endif
