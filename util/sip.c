
/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "os-features.h"
#include "sip.h"
#include "starutil.h"
#include "mathutil.h"

static anbool has_distortions(const sip_t* sip) {
    return (sip->a_order >= 0);
}

anbool sip_xyz2pixelxy(const sip_t* sip, double x, double y, double z, double *px, double *py) {
    double xyz[3];
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
    return sip_xyzarr2pixelxy(sip, xyz, px, py);
}

void sip_wrap_tan(const tan_t* tan, sip_t* sip) {
    memset(sip, 0, sizeof(sip_t));
    memcpy(&(sip->wcstan), tan, sizeof(tan_t));
}

void sip_get_crval(const sip_t* sip, double* ra, double* dec) {
    *ra = sip->wcstan.crval[0];
    *dec = sip->wcstan.crval[1];
}

double sip_imagew(sip_t* sip) {
    assert(sip);
    return sip->wcstan.imagew;
}
double sip_imageh(sip_t* sip) {
    assert(sip);
    return sip->wcstan.imageh;
}
sip_t* sip_create() {
    sip_t* sip = calloc(1, sizeof(sip_t));

    sip->wcstan.cd[0][0] = 1;
    sip->wcstan.cd[0][1] = 0;
    sip->wcstan.cd[1][0] = 0;
    sip->wcstan.cd[1][1] = 1;

    return sip;
}

void sip_free(sip_t* sip) {
    free(sip);
}

void sip_copy(sip_t* dest, const sip_t* src) {
    memcpy(dest, src, sizeof(sip_t));
}

static void sip_distortion(const sip_t* sip, double x, double y,
                           double* X, double* Y) {
    // Get pixel coordinates relative to reference pixel
    double u = x - sip->wcstan.crpix[0];
    double v = y - sip->wcstan.crpix[1];
    sip_calc_distortion(sip, u, v, X, Y);
    *X += sip->wcstan.crpix[0];
    *Y += sip->wcstan.crpix[1];
}

// Pixels to RA,Dec in degrees.
void sip_pixelxy2radec(const sip_t* sip, double px, double py,
                       double *ra, double *dec) {
    if (has_distortions(sip)) {
        double U, V;
        sip_distortion(sip, px, py, &U, &V);
        // Run a normal TAN conversion on the distorted pixel coords.
        tan_pixelxy2radec(&(sip->wcstan), U, V, ra, dec);
    } else
        // Run a normal TAN conversion
        tan_pixelxy2radec(&(sip->wcstan), px, py, ra, dec);
}

// Pixels to Intermediate World Coordinates in degrees.
void sip_pixelxy2iwc(const sip_t* sip, double px, double py,
                     double *iwcx, double* iwcy) {
    if (has_distortions(sip)) {
        double U, V;
        sip_distortion(sip, px, py, &U, &V);
        // Run a normal TAN conversion on the distorted pixel coords.
        tan_pixelxy2iwc(&(sip->wcstan), U, V, iwcx, iwcy);
    } else
        // Run a normal TAN conversion
        tan_pixelxy2iwc(&(sip->wcstan), px, py, iwcx, iwcy);
}

// Pixels to XYZ unit vector.
void sip_pixelxy2xyzarr(const sip_t* sip, double px, double py, double *xyz) {
    if (has_distortions(sip)) {
        double U, V;
        sip_distortion(sip, px, py, &U, &V);
        // Run a normal TAN conversion on the distorted pixel coords.
        tan_pixelxy2xyzarr(&(sip->wcstan), U, V, xyz);
    } else
        // Run a normal TAN conversion
        tan_pixelxy2xyzarr(&(sip->wcstan), px, py, xyz);
}

// Pixels to RA,Dec in degrees.
void tan_pixelxy2radec(const tan_t* tan, double px, double py, double *ra, double *dec) {
    double xyz[3];
    tan_pixelxy2xyzarr(tan, px, py, xyz);
    xyzarr2radecdeg(xyz, ra,dec);
}

void   tan_pixelxy2radecarr(const tan_t* wcs_tan, double px, double py, double *radec) {
    tan_pixelxy2radec(wcs_tan, px, py, radec+0, radec+1);
}

void tan_iwc2pixelxy(const tan_t* tan, double x, double y,
                     double *px, double* py) {
    double U,V;
    Unused int r;
    double cdi[2][2];

    // Invert CD
    r = invert_2by2_arr((const double*)tan->cd, (double*)cdi);
    assert(r == 0);

    // Linear pixel coordinates
    U = cdi[0][0]*x + cdi[0][1]*y;
    V = cdi[1][0]*x + cdi[1][1]*y;

    // Re-add crpix to get pixel coordinates
    *px = U + tan->crpix[0];
    *py = V + tan->crpix[1];
}

void tan_pixelxy2iwc(const tan_t* tan, double px, double py, double *iwcx, double* iwcy)
{
    // Get pixel coordinates relative to reference pixel
    double U = px - tan->crpix[0];
    double V = py - tan->crpix[1];

    // Get intermediate world coordinates
    double x = tan->cd[0][0] * U + tan->cd[0][1] * V;
    double y = tan->cd[1][0] * U + tan->cd[1][1] * V;

    if (iwcx)
        *iwcx = x;
    if (iwcy)
        *iwcy = y;
}

void tan_iwc2radec(const tan_t* tan, double x, double y, double *p_ra, double *p_dec) {
    double xyz[3];
    tan_iwc2xyzarr(tan, x, y, xyz);
    xyzarr2radecdeg(xyz, p_ra, p_dec);
}

void tan_iwc2xyzarr(const tan_t* tan, double x, double y, double *xyz)
{
    double rx, ry, rz;
    double ix,iy,norm;
    double jx,jy,jz;

    // Mysterious factor of -1 correcting for vector directions below.
    x = -deg2rad(x);
    y =  deg2rad(y);

    // Take r to be the threespace vector of crval
    radecdeg2xyz(tan->crval[0], tan->crval[1], &rx, &ry, &rz);
    //printf("rx=%lf ry=%lf rz=%lf\n",rx,ry,rz);

    // FIXME -- what about *near* the poles?
    if (rz == 1.0) {
        // North pole
        ix = -1.0;
        iy = 0.0;
    } else if (rz == -1.0) {
        // South pole
        ix = -1.0;
        iy = 0.0;
    } else {
        // Form i = r cross north pole (0,0,1)
        ix = ry;
        iy = -rx;
        // iz = 0
        norm = hypot(ix, iy);
        ix /= norm;
        iy /= norm;
        //printf("ix=%lf iy=%lf iz=0.0\n",ix,iy);
        //	printf("r.i = %lf\n",ix*rx+iy*ry);
    }

    // Form j = i cross r;   iz=0 so some terms drop out
    jx = iy * rz;
    jy =         - ix * rz;
    jz = ix * ry - iy * rx;
    // norm should already be 1, but normalize anyway
    normalize(&jx, &jy, &jz);
    //	printf("jx=%lf jy=%lf jz=%lf\n",jx,jy,jz);
    //	printf("r.j = %lf\n",jx*rx+jy*ry+jz*rz);
    //	printf("i.j = %lf\n",ix*jx+iy*jy);

    if (tan->sin) {
        assert((x*x + y*y) < 1.0);
        // Figure out what factor of r we have to add in to make the resulting length = 1
        double rfrac = sqrt(1.0 - (x*x + y*y));
        // Don't scale the projected x,y positions, just add in the right amount of r to
        // bring it onto the unit sphere
        xyz[0] = ix*x + jx*y + rx * rfrac;
        xyz[1] = iy*x + jy*y + ry * rfrac;
        xyz[2] =        jz*y + rz * rfrac; // iz = 0

    } else {
        // Form the point on the tangent plane relative to observation point,
        xyz[0] = ix*x + jx*y + rx;
        xyz[1] = iy*x + jy*y + ry;
        xyz[2] =        jz*y + rz; // iz = 0
        // and normalize back onto the unit sphere
        normalize_3(xyz);
    }
}

// Pixels to XYZ unit vector.
void tan_pixelxy2xyzarr(const tan_t* tan, double px, double py, double *xyz)
{
    double x,y;
    tan_pixelxy2iwc(tan, px, py, &x, &y);
    tan_iwc2xyzarr(tan, x, y, xyz);
}

void sip_iwc2pixelxy(const sip_t* sip, double u, double v,
                     double *px, double* py) {
    double x,y;
    tan_iwc2pixelxy(&(sip->wcstan), u, v, &x, &y);
    sip_pixel_undistortion(sip, x, y, px, py);
}

// SIP:  (RA,Dec) --> IWC --> x,y --> x',y'

// RA,Dec in degrees to Pixels.
anbool sip_radec2pixelxy(const sip_t* sip, double ra, double dec, double *px, double *py) {
    double x,y;
    if (!tan_radec2pixelxy(&(sip->wcstan), ra, dec, &x, &y))
        return FALSE;
    sip_pixel_undistortion(sip, x, y, px, py);
    return TRUE;
}

void sip_iwc2radec(const sip_t* sip, double x, double y, double *p_ra, double *p_dec) {
    tan_iwc2radec(&(sip->wcstan), x, y, p_ra, p_dec);
}


// RA,Dec in degrees to Pixels.
anbool sip_radec2pixelxy_check(const sip_t* sip, double ra, double dec, double *px, double *py) {
    double u, v;
    double U, V;
    double U2, V2;
    if (!tan_radec2pixelxy(&(sip->wcstan), ra, dec, px, py))
        return FALSE;
    if (!has_distortions(sip))
        return TRUE;

    // Subtract crpix, invert SIP distortion, add crpix.
    // Sanity check:
    if (sip->a_order != 0 && sip->ap_order == 0) {
        fprintf(stderr, "suspicious inversion; no inversion SIP coeffs "
                "yet there are forward SIP coeffs\n");
    }
    U = *px - sip->wcstan.crpix[0];
    V = *py - sip->wcstan.crpix[1];
    sip_calc_inv_distortion(sip, U, V, &u, &v);
    // Check that we're dealing with the right range of the polynomial by inverting it and
    // checking that we end up back in the right place.
    sip_calc_distortion(sip, u, v, &U2, &V2);
    if (fabs(U2 - U) + fabs(V2 - V) > 10.0)
        return FALSE;
    *px = u + sip->wcstan.crpix[0];
    *py = v + sip->wcstan.crpix[1];
    return TRUE;
}

anbool sip_xyzarr2pixelxy(const sip_t* sip, const double* xyz, double *px, double *py) {
    double ra, dec;
    xyzarr2radecdeg(xyz, &ra, &dec);
    return sip_radec2pixelxy(sip, ra, dec, px, py);
}


anbool sip_xyzarr2iwc(const sip_t* sip, const double* xyz,
                      double* iwcx, double* iwcy) {
    return tan_xyzarr2iwc(&(sip->wcstan), xyz, iwcx, iwcy);
}
anbool sip_radec2iwc(const sip_t* sip, double ra, double dec,
                     double* iwcx, double* iwcy) {
    return tan_radec2iwc(&(sip->wcstan), ra, dec, iwcx, iwcy);
}

anbool tan_xyzarr2iwc(const tan_t* tan, const double* xyz,
                      double* iwcx, double* iwcy) {
    double xyzcrval[3];

    // FIXME be robust near the poles
    // Calculate intermediate world coordinates (x,y) on the tangent plane
    radecdeg2xyzarr(tan->crval[0], tan->crval[1], xyzcrval);

    if (!star_coords(xyz, xyzcrval, !tan->sin, iwcx, iwcy))
        return FALSE;

    *iwcx = rad2deg(*iwcx);
    *iwcy = rad2deg(*iwcy);
    return TRUE;
}

anbool tan_radec2iwc(const tan_t* tan, double ra, double dec,
                     double* iwcx, double* iwcy) {
    double xyz[3];
    radecdeg2xyzarr(ra, dec, xyz);
    return tan_xyzarr2iwc(tan, xyz, iwcx, iwcy);
}

// xyz unit vector to Pixels.
anbool tan_xyzarr2pixelxy(const tan_t* tan, const double* xyzpt, double *px, double *py) {
    double iwx=0, iwy=0;
    if (!tan_xyzarr2iwc(tan, xyzpt, &iwx, &iwy)) {
        return FALSE;
    }
    tan_iwc2pixelxy(tan, iwx, iwy, px, py);
    return TRUE;
}

// RA,Dec in degrees to Pixels.
anbool tan_radec2pixelxy(const tan_t* tan, double a, double d, double *px, double *py)
{
    double xyzpt[3];
    radecdeg2xyzarr(a,d,xyzpt);
    return tan_xyzarr2pixelxy(tan, xyzpt, px, py);
}

void sip_calc_distortion(const sip_t* sip, double u, double v, double* U, double *V) {
    // Do SIP distortion (in relative pixel coordinates)
    // See the sip_t struct definition in header file for details
    int p, q;
    double fuv=0.;
    double guv=0.;

    // avoid using pow() function
    double powu[SIP_MAXORDER];
    double powv[SIP_MAXORDER];
    powu[0] = 1.0;
    powu[1] = u;
    powv[0] = 1.0;
    powv[1] = v;
    for (p=2; p <= MAX(sip->a_order, sip->b_order); p++) {
        powu[p] = powu[p-1] * u;
        powv[p] = powv[p-1] * v;
    }

    for (p=0; p<=sip->a_order; p++)
        for (q=0; q<=sip->a_order; q++)
            // We include all terms, even the constant and linear ones; the standard
            // isn't clear on whether these are allowed or not.
            if (p+q <= sip->a_order)
                fuv += sip->a[p][q] * powu[p] * powv[q];
    for (p=0; p<=sip->b_order; p++) 
        for (q=0; q<=sip->b_order; q++) 
            if (p+q <= sip->b_order)
                guv += sip->b[p][q] * powu[p] * powv[q];

    *U = u + fuv;
    *V = v + guv;
}

void sip_pixel_distortion(const sip_t* sip, double x, double y, double* X, double *Y) {
    sip_distortion(sip, x, y, X, Y);
}

void sip_pixel_undistortion(const sip_t* sip, double x, double y, double* X, double *Y) {
    if (!has_distortions(sip)) {
        *X = x;
        *Y = y;
        return;
    }
    // Sanity check:
    if (sip->a_order != 0 && sip->ap_order == 0) {
        fprintf(stderr, "suspicious inversion; no inverse SIP coeffs "
                "yet there are forward SIP coeffs\n");
    }

    // Get pixel coordinates relative to reference pixel
    double u = x - sip->wcstan.crpix[0];
    double v = y - sip->wcstan.crpix[1];
    sip_calc_inv_distortion(sip, u, v, X, Y);
    *X += sip->wcstan.crpix[0];
    *Y += sip->wcstan.crpix[1];
}

void sip_calc_inv_distortion(const sip_t* sip, double U, double V, double* u, double *v)
{
    int p, q;
    double fUV=0.;
    double gUV=0.;

    // avoid using pow() function
    double powu[SIP_MAXORDER];
    double powv[SIP_MAXORDER];
    powu[0] = 1.0;
    powu[1] = U;
    powv[0] = 1.0;
    powv[1] = V;
    for (p=2; p <= MAX(sip->ap_order, sip->bp_order); p++) {
        powu[p] = powu[p-1] * U;
        powv[p] = powv[p-1] * V;
    }

    for (p=0; p<=sip->ap_order; p++)
        for (q=0; q<=sip->ap_order; q++)
            if (p+q <= sip->ap_order)
                fUV += sip->ap[p][q] * powu[p] * powv[q];
    for (p=0; p<=sip->bp_order; p++) 
        for (q=0; q<=sip->bp_order; q++) 
            if (p+q <= sip->bp_order)
                gUV += sip->bp[p][q] * powu[p] * powv[q];

    *u = U + fUV;
    *v = V + gUV;
}

double tan_det_cd(const tan_t* tan) {
    return (tan->cd[0][0]*tan->cd[1][1] - tan->cd[0][1]*tan->cd[1][0]);
}

double sip_det_cd(const sip_t* sip) {
    return tan_det_cd(&(sip->wcstan));
}

double tan_pixel_scale(const tan_t* tan) {
    double scale = deg2arcsec(sqrt(fabs(tan_det_cd(tan))));
    return scale;
}

// returns pixel scale in arcseconds
double sip_pixel_scale(const sip_t* sip) {
    return tan_pixel_scale(&(sip->wcstan));
}

static void print_to(const tan_t* tan, FILE* f, char* type) {
    fprintf(f,"%s Structure:\n", type);
    fprintf(f,"  crval=(%g, %g)\n", tan->crval[0], tan->crval[1]);
    fprintf(f,"  crpix=(%g, %g)\n", tan->crpix[0], tan->crpix[1]);
    fprintf(f,"  CD = ( %12.5g   %12.5g )\n", tan->cd[0][0], tan->cd[0][1]);
    fprintf(f,"       ( %12.5g   %12.5g )\n", tan->cd[1][0], tan->cd[1][1]);
    fprintf(f,"  image size = (%g x %g)\n", tan->imagew, tan->imageh);
}

void tan_print_to(const tan_t* tan, FILE* f) {
    if (tan->sin) {
        print_to(tan, f, "SIN");
    } else {
        print_to(tan, f, "TAN");
    }
}

void tan_print(const tan_t* tan) {
    tan_print_to(tan, stderr);
}

void sip_print_to(const sip_t* sip, FILE* f) {
    double det,pixsc;

    if (sip->wcstan.sin) {
        print_to(&(sip->wcstan), f, "SIN-SIP");
    } else {
        print_to(&(sip->wcstan), f, "TAN-SIP");
    }

    fprintf(f, "  SIP order: A=%i, B=%i, AP=%i, BP=%i\n",
            sip->a_order, sip->b_order, sip->ap_order, sip->bp_order);

    if (sip->a_order > 0) {
        int p, q;
        for (p=0; p<=sip->a_order; p++) {
            fprintf(f, (p ? "      " : "  A = "));
            for (q=0; q<=sip->a_order; q++)
                if (p+q <= sip->a_order)
                    //fprintf(f,"a%d%d=%le\n", p,q,sip->a[p][q]);
                    fprintf(f,"%12.5g", sip->a[p][q]);
            fprintf(f,"\n");
        }
    }
    if (sip->b_order > 0) {
        int p, q;
        for (p=0; p<=sip->b_order; p++) {
            fprintf(f, (p ? "      " : "  B = "));
            for (q=0; q<=sip->b_order; q++)
                if (p+q <= sip->a_order)
                    fprintf(f,"%12.5g", sip->b[p][q]);
            //if (p+q <= sip->b_order && p+q > 0)
            //fprintf(f,"b%d%d=%le\n", p,q,sip->b[p][q]);
            fprintf(f,"\n");
        }
    }

    if (sip->ap_order > 0) {
        int p, q;
        for (p=0; p<=sip->ap_order; p++) {
            fprintf(f, (p ? "      " : "  AP = "));
            for (q=0; q<=sip->ap_order; q++)
                if (p+q <= sip->ap_order)
                    fprintf(f,"%12.5g", sip->ap[p][q]);
            fprintf(f,"\n");
        }
    }
    if (sip->bp_order > 0) {
        int p, q;
        for (p=0; p<=sip->bp_order; p++) {
            fprintf(f, (p ? "      " : "  BP = "));
            for (q=0; q<=sip->bp_order; q++)
                if (p+q <= sip->bp_order)
                    fprintf(f,"%12.5g", sip->bp[p][q]);
            fprintf(f,"\n");
        }
    }


    det = sip_det_cd(sip);
    pixsc = 3600*sqrt(fabs(det));
    //fprintf(f,"  det(CD)=%g\n", det);
    fprintf(f,"  sqrt(det(CD))=%g [arcsec]\n", pixsc);
    //fprintf(f,"\n");
}

void sip_print(const sip_t* sip) {
    sip_print_to(sip, stderr);
}

double sip_get_orientation(const sip_t* sip) {
    return tan_get_orientation(&(sip->wcstan));
}

double tan_get_orientation(const tan_t* tan) {
    double T, A, orient;
    double det, parity;
    det = tan_det_cd(tan);
    parity = (det >= 0 ? 1.0 : -1.0);
    T = parity * tan->cd[0][0] + tan->cd[1][1];
    A = parity * tan->cd[1][0] - tan->cd[0][1];
    orient = -rad2deg(atan2(A, T));
    return orient;
}

