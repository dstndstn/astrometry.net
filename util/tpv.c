/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tpv.h"
#include "starutil.h"
#include "mathutil.h"

/*******************************

 NOTE, this does not work yet!!!

 I copied this from sip.c, which applies the distortion in PIXEL space,
 while TPV applies the distortion in IWC space.



 The TPV projection is evaluated as follows.

 Compute the first order standard coordinates xi and eta from the
 linear part of the solution stored in CRPIX and the CD matrix.

 xi = CD1_1 * (x - CRPIX1) + CD1_2 * (y - CRPIX2)
 eta = CD2_1 * (x - CRPIX1) + CD2_2 * (y - CRPIX2)

 Apply the distortion transformation using the coefficients in the
 PV keywords as described below.

 xi' = f_xi (xi, eta)
 eta' = f_eta (xi, eta)

 Apply the tangent plane projection to xi' and eta' as described in
 Calabretta and Greisen . The reference tangent point given by the
 CRVAL values lead to the final RA and DEC in degrees. Note that
 the units of xi, eta, f_xi, and f_eta are also degrees.

 ********************************/


anbool tpv_xyz2pixelxy(const tpv_t* tpv, double x, double y, double z, double *px, double *py) {
    double xyz[3];
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
    return tpv_xyzarr2pixelxy(tpv, xyz, px, py);
}

void tpv_wrap_tan(const tan_t* tan, tpv_t* tpv) {
    memset(tpv, 0, sizeof(tpv_t));
    memcpy(&(tpv->wcstan), tan, sizeof(tan_t));
}

void tpv_get_crval(const tpv_t* tpv, double* ra, double* dec) {
    *ra = tpv->wcstan.crval[0];
    *dec = tpv->wcstan.crval[1];
}

double tpv_imagew(tpv_t* tpv) {
    assert(tpv);
    return tpv->wcstan.imagew;
}
double tpv_imageh(tpv_t* tpv) {
    assert(tpv);
    return tpv->wcstan.imageh;
}
tpv_t* tpv_create() {
    tpv_t* tpv = calloc(1, sizeof(tpv_t));

    tpv->wcstan.cd[0][0] = 1;
    tpv->wcstan.cd[0][1] = 0;
    tpv->wcstan.cd[1][0] = 0;
    tpv->wcstan.cd[1][1] = 1;

    return tpv;
}

void tpv_free(tpv_t* tpv) {
    free(tpv);
}

void tpv_copy(tpv_t* dest, const tpv_t* src) {
    memcpy(dest, src, sizeof(tpv_t));
}

static void tpv_distortion(const tpv_t* tpv, double x, double y,
                           double* X, double* Y) {
    // Get pixel coordinates relative to reference pixel
    double u = x - tpv->wcstan.crpix[0];
    double v = y - tpv->wcstan.crpix[1];
    tpv_calc_distortion(tpv, u, v, X, Y);
    *X += tpv->wcstan.crpix[0];
    *Y += tpv->wcstan.crpix[1];
}

// Pixels to RA,Dec in degrees.
void tpv_pixelxy2radec(const tpv_t* tpv, double px, double py,
                       double *ra, double *dec) {
    double U, V;
    tpv_distortion(tpv, px, py, &U, &V);
    // Run a normal TAN conversion on the distorted pixel coords.
    tan_pixelxy2radec(&(tpv->wcstan), U, V, ra, dec);
}

// Pixels to Intermediate World Coordinates in degrees.
void tpv_pixelxy2iwc(const tpv_t* tpv, double px, double py,
                     double *iwcx, double* iwcy) {
    double U, V;
    tpv_distortion(tpv, px, py, &U, &V);
    // Run a normal TAN conversion on the distorted pixel coords.
    tan_pixelxy2iwc(&(tpv->wcstan), U, V, iwcx, iwcy);
}

// Pixels to XYZ unit vector.
void tpv_pixelxy2xyzarr(const tpv_t* tpv, double px, double py, double *xyz) {
    double U, V;
    tpv_distortion(tpv, px, py, &U, &V);
    // Run a normal TAN conversion on the distorted pixel coords.
    tan_pixelxy2xyzarr(&(tpv->wcstan), U, V, xyz);
}

void tpv_iwc2pixelxy(const tpv_t* tpv, double u, double v,
                     double *px, double* py) {
    double x,y;
    tan_iwc2pixelxy(&(tpv->wcstan), u, v, &x, &y);
    tpv_pixel_undistortion(tpv, x, y, px, py);
}

// RA,Dec in degrees to Pixels.
anbool tpv_radec2pixelxy(const tpv_t* tpv, double ra, double dec, double *px, double *py) {
    double x,y;
    if (!tan_radec2pixelxy(&(tpv->wcstan), ra, dec, &x, &y))
        return FALSE;
    tpv_pixel_undistortion(tpv, x, y, px, py);
    return TRUE;
}

void tpv_iwc2radec(const tpv_t* tpv, double x, double y, double *p_ra, double *p_dec) {
    tan_iwc2radec(&(tpv->wcstan), x, y, p_ra, p_dec);
}

anbool tpv_xyzarr2pixelxy(const tpv_t* tpv, const double* xyz, double *px, double *py) {
    double ra, dec;
    xyzarr2radecdeg(xyz, &ra, &dec);
    return tpv_radec2pixelxy(tpv, ra, dec, px, py);
}


anbool tpv_xyzarr2iwc(const tpv_t* tpv, const double* xyz,
                      double* iwcx, double* iwcy) {
    return tan_xyzarr2iwc(&(tpv->wcstan), xyz, iwcx, iwcy);
}
anbool tpv_radec2iwc(const tpv_t* tpv, double ra, double dec,
                     double* iwcx, double* iwcy) {
    return tan_radec2iwc(&(tpv->wcstan), ra, dec, iwcx, iwcy);
}

void tpv_calc_distortion(const tpv_t* tpv, double xi, double eta,
                         double* XI, double *ETA) {
    // Do TPV distortion (in intermediate world coords)
    // xi,eta  ->  xi',eta'

    /**
     From http://iraf.noao.edu/projects/mosaic/tpv.html

     p = PV1_

     xi' = p0 +
     p1 * xi + p2 * eta + p3 * r +
     p4 * xi^2 + p5 * xi * eta + p6 * eta^2 +
     p7 * xi^3 + p8 * xi^2 * eta + p9 * xi * eta^2 +
     p10 * eta^3 + p11 * r^3 + 
     p12 * xi^4 + p13 * xi^3 * eta + p14 * xi^2 * eta^2 +
     p15 * xi * eta^3 + p16 * eta^4 +
     p17 * xi^5 + p18 * xi^4 * eta + p19 * xi^3 * eta^2 +
     p20 * xi^2 * eta^3 + p21 * xi * eta^4 + p22 * eta^5 + p23 * r^5 +
     p24 * xi^6 + p25 * xi^5 * eta + p26 * xi^4 * eta^2 +
     p27 * xi^3 * eta^3 + p28 * xi^2 * eta^4 + p29 * xi * eta^5 +
     p30 * eta^6
     p31 * xi^7 + p32 * xi^6 * eta + p33 * xi^5 * eta^2 +
     p34 * xi^4 * eta^3 + p35 * xi^3 * eta^4 + p36 * xi^2 * eta^5 +
     p37 * xi * eta^6 + p38 * eta^7 + p39 * r^7

     p = PV2_
     eta' = p0 +
     p1 * eta + p2 * xi + p3 * r +
     p4 * eta^2 + p5 * eta * xi + p6 * xi^2 +
     p7 * eta^3 + p8 * eta^2 * xi + p9 * eta * xi^2 + p10 * xi^3 +
     p11 * r^3 +
     p12 * eta^4 + p13 * eta^3 * xi + p14 * eta^2 * xi^2 +
     p15 * eta * xi^3 + p16 * xi^4 +
     p17 * eta^5 + p18 * eta^4 * xi + p19 * eta^3 * xi^2 +
     p20 * eta^2 * xi^3 + p21 * eta * xi^4 + p22 * xi^5 +
     p23 * r^5 +
     p24 * eta^6 + p25 * eta^5 * xi + p26 * eta^4 * xi^2 +
     p27 * eta^3 * xi^3 + p28 * eta^2 * xi^4 + p29 * eta * xi^5 +
     p30 * xi^6
     p31 * eta^7 + p32 * eta^6 * xi + p33 * eta^5 * xi^2 +
     p34 * eta^4 * xi^3 + p35 * eta^3 * xi^4 + p36 * eta^2 * xi^5 +
     p37 * eta * xi^6 + p38 * xi^7 + p39 * r^7

     Note the "cross-over" -- the xi' powers are in terms of xi,eta
     while the eta' powers are in terms of eta,xi.
     */

    //           1  x  y  r x2 xy y2 x3 x2y xy2 y3 r3 x4 x3y x2y2 xy3 y4
    //          x5 x4y x3y2 x2y3 xy4 y5 r5 x6 x5y x4y2, x3y3 x2y4 xy5 y6
    //          x7 x6y x5y2 x4y3 x3y4 x2y5 xy6 y7 r7
    int xp[] = {
        0,
        1, 0, 0,
        2, 1, 0,
        3, 2, 1, 0, 0,
        4, 3, 2, 1, 0,
        5, 4, 3, 2, 1, 0, 0,
        6, 5, 4, 3, 2, 1, 0,
        7, 6, 5, 4, 3, 2, 1, 0, 0};
    int yp[] = {
        0,
        0, 1, 0,
        0, 1, 2,
        0, 1, 2, 3, 0,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 5, 0,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6, 7, 0};
    int rp[] = {
        0,
        0, 0, 1,
        0, 0, 0,
        0, 0, 0, 0, 3,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 5,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 7};
    double xipows[8];
    double etapows[8];
    double rpows[8];
    double r;
    int i;
    double px, py;

    r = sqrt(xi*xi + eta*eta);
    xipows[0] = etapows[0] = rpows[0] = 1.0;
    for (i=1; i<sizeof(xipows)/sizeof(double); i++) {
        xipows[i] = xipows[i-1]*xi;
        etapows[i] = etapows[i-1]*eta;
        rpows[i] = rpows[i-1]*r;
    }
    px = py = 0;
    for (i=0; i<sizeof(xp)/sizeof(int); i++) {
        px += tpv->pv1[i] *  xipows[xp[i]] * etapows[yp[i]]
            * rpows[rp[i]];
        // here's the "cross-over" mentioned above
        py += tpv->pv2[i] * etapows[xp[i]] * xipows[yp[i]]
            * rpows[rp[i]];
    }
    *XI = px;
    *ETA = py;
}

void tpv_pixel_distortion(const tpv_t* tpv, double x, double y, double* X, double *Y) {
    tpv_distortion(tpv, x, y, X, Y);
}

void tpv_pixel_undistortion(const tpv_t* tpv, double x, double y, double* X, double *Y) {
    // Get pixel coordinates relative to reference pixel
    double u = x - tpv->wcstan.crpix[0];
    double v = y - tpv->wcstan.crpix[1];
    tpv_calc_inv_distortion(tpv, u, v, X, Y);
    *X += tpv->wcstan.crpix[0];
    *Y += tpv->wcstan.crpix[1];
}

void tpv_calc_inv_distortion(const tpv_t* tpv, double U, double V, double* u, double *v)
{
    assert(0);
    printf("tpv_calc_inv_distortion not implemented!\n");
    exit(-1);
}

double tpv_det_cd(const tpv_t* tpv) {
    return tan_det_cd(&(tpv->wcstan));
}

// returns pixel scale in arcseconds (NOT arcsec^2)
double tpv_pixel_scale(const tpv_t* tpv) {
    return tan_pixel_scale(&(tpv->wcstan));
}

void tpv_print_to(const tpv_t* tpv, FILE* f) {
    double det,pixsc;

    print_to(&(tpv->wcstan), f, "TPV");
   
    /*
     if (tpv->a_order > 0) {
     int p, q;
     for (p=0; p<=tpv->a_order; p++) {
     fprintf(f, (p ? "      " : "  A = "));
     for (q=0; q<=tpv->a_order; q++)
     if (p+q <= tpv->a_order)
     //fprintf(f,"a%d%d=%le\n", p,q,tpv->a[p][q]);
     fprintf(f,"%12.5g", tpv->a[p][q]);
     fprintf(f,"\n");
     }
     }
     if (tpv->b_order > 0) {
     int p, q;
     for (p=0; p<=tpv->b_order; p++) {
     fprintf(f, (p ? "      " : "  B = "));
     for (q=0; q<=tpv->b_order; q++)
     if (p+q <= tpv->a_order)
     fprintf(f,"%12.5g", tpv->b[p][q]);
     //if (p+q <= tpv->b_order && p+q > 0)
     //fprintf(f,"b%d%d=%le\n", p,q,tpv->b[p][q]);
     fprintf(f,"\n");
     }
     }
     det = tpv_det_cd(tpv);
     pixsc = 3600*sqrt(fabs(det));
     //fprintf(f,"  det(CD)=%g\n", det);
     fprintf(f,"  sqrt(det(CD))=%g [arcsec]\n", pixsc);
     //fprintf(f,"\n");
     */
}

void tpv_print(const tpv_t* tpv) {
    tpv_print_to(tpv, stderr);
}

double tpv_get_orientation(const tpv_t* tpv) {
    return tan_get_orientation(&(tpv->wcstan));
}
