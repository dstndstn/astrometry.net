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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "sip.h"
#include "starutil.h"
#include "mathutil.h"

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
					   double *ra, double *dec)
{
	double U, V;
    sip_distortion(sip, px, py, &U, &V);
	// Run a normal TAN conversion on the distorted pixel coords.
	tan_pixelxy2radec(&(sip->wcstan), U, V, ra, dec);
}

// Pixels to XYZ unit vector.
void sip_pixelxy2xyzarr(const sip_t* sip, double px, double py, double *xyz) {
    /*
     double ra, dec;
     sip_pixelxy2radec(sip, px, py, &ra, &dec);
     radecdeg2xyzarr(ra, dec, xyz);
     */
	double U, V;
    sip_distortion(sip, px, py, &U, &V);
	// Run a normal TAN conversion on the distorted pixel coords.
	tan_pixelxy2xyzarr(&(sip->wcstan), U, V, xyz);
}

// Pixels to RA,Dec in degrees.
void tan_pixelxy2radec(const tan_t* tan, double px, double py, double *ra, double *dec)
{
	double xyz[3];
	tan_pixelxy2xyzarr(tan, px, py, xyz);
	xyzarr2radecdeg(xyz, ra,dec);
}

// Pixels to XYZ unit vector.
void tan_pixelxy2xyzarr(const tan_t* tan, double px, double py, double *xyz)
{
	double rx, ry, rz;
	double ix,iy,norm;
	double jx,jy,jz;
	double wx,wy,wz;

	// Get pixel coordinates relative to reference pixel
	double U = px - tan->crpix[0];
	double V = py - tan->crpix[1];

	// Get intermediate world coordinates
	double x = tan->cd[0][0] * U + tan->cd[0][1] * V;
	double y = tan->cd[1][0] * U + tan->cd[1][1] * V;
	//	printf("x=%lf y=%lf\n",x,y);

	// Mysterious! Who knows, but negating these coordinates makes WCStools match with SIP. 
	x = -deg2rad(x);
	y = -deg2rad(y);

	// Take r to be the threespace vector of crval
	radecdeg2xyz(tan->crval[0], tan->crval[1], &rx, &ry, &rz);
	//	printf("rx=%lf ry=%lf rz=%lf\n",rx,ry,rz);

	// Form i = r cross north pole, which is in direction of z
	ix = ry;
	iy = -rx;
	//     iz = 0 because the the north pole is at (0,0,1)
	norm = hypot(ix, iy);
	ix /= norm;
	iy /= norm;
	//	printf("ix=%lf iy=%lf iz=0.0\n",ix,iy);
	//	printf("r.i = %lf\n",ix*rx+iy*ry);

	// Form j = r cross i, which is in the direction of y
	jx =  rx*rz;
	jy =  rz*ry;
	jz = -rx*rx - ry*ry;
	// norm should already be 1, but normalize anyway
	normalize(&jx, &jy, &jz);
	//	printf("jx=%lf jy=%lf jz=%lf\n",jx,jy,jz);
	//	printf("r.j = %lf\n",jx*rx+jy*ry+jz*rz);
	//	printf("i.j = %lf\n",ix*jx+iy*jy);

	// Form the point on the tangent plane relative to observation point,
	// and normalize back onto the unit sphere
	wx = ix*x + jx*y + rx;
	wy = iy*x + jy*y + ry;
	wz =        jz*y + rz; // iz = 0
	normalize(&wx, &wy, &wz);
	//	printf("wx=%lf wy=%lf wz=%lf\n",wx,wy,wz);

	xyz[0] = wx;
	xyz[1] = wy;
	xyz[2] = wz;
}

// RA,Dec in degrees to Pixels.
bool sip_radec2pixelxy(const sip_t* sip, double ra, double dec, double *px, double *py)
{
	double u, v;
	double U, V;

	if (!tan_radec2pixelxy(&(sip->wcstan), ra, dec, px, py))
		return FALSE;

	// Subtract crpix, invert SIP distortion, add crpix.

	// Sanity check:
	if (sip->a_order != 0 && sip->ap_order == 0) {
		fprintf(stderr, "suspicious inversion; no inversion SIP coeffs "
				"yet there are forward SIP coeffs\n");
	}

	U = *px - sip->wcstan.crpix[0];
	V = *py - sip->wcstan.crpix[1];

	sip_calc_inv_distortion(sip, U, V, &u, &v);

	*px = u + sip->wcstan.crpix[0];
	*py = v + sip->wcstan.crpix[1];
	return TRUE;
}

// RA,Dec in degrees to Pixels.
bool sip_radec2pixelxy_check(const sip_t* sip, double ra, double dec, double *px, double *py) {
	double u, v;
	double U, V;
	double U2, V2;
	if (!tan_radec2pixelxy(&(sip->wcstan), ra, dec, px, py))
		return FALSE;
	// Subtract crpix, invert SIP distortion, add crpix.
	// Sanity check:
	if (sip->a_order != 0 && sip->ap_order == 0) {
		fprintf(stderr, "suspicious inversion; no inversion SIP coeffs "
				"yet there are forward SIP coeffs\n");
	}
	U = *px - sip->wcstan.crpix[0];
	V = *py - sip->wcstan.crpix[1];
	sip_calc_inv_distortion(sip, U, V, &u, &v);
    // Check that we're dealing with the right range of the polynomial be inverting it and
    // checking that we end up back in the right place.
    sip_calc_distortion(sip, u, v, &U2, &V2);
    if (fabs(U2 - U) + fabs(V2 - V) > 10.0)
        return FALSE;
	*px = u + sip->wcstan.crpix[0];
	*py = v + sip->wcstan.crpix[1];
	return TRUE;
}

bool sip_xyzarr2pixelxy(const sip_t* sip, const double* xyz, double *px, double *py) {
	double ra, dec;
	xyzarr2radecdeg(xyz, &ra, &dec);
	return sip_radec2pixelxy(sip, ra, dec, px, py);
}

// xyz unit vector to Pixels.
bool tan_xyzarr2pixelxy(const tan_t* tan, const double* xyzpt, double *px, double *py) {
	double x=0, y=0;
	double U,V;
	double xyzcrval[3];
	double cdi[2][2];
	int r;

	// Invert CD
    r = invert_2by2_arr((const double*)tan->cd, (double*)cdi);
	assert(r == 0);

	/*
	printf(":: %lf\n",  (tan->cd[0][0]*cdi[0][0] + tan->cd[0][1]*cdi[1][0] ));
	printf(":: %lf\n",  (tan->cd[0][0]*cdi[0][1] + tan->cd[0][1]*cdi[1][1] ));
	printf(":: %lf\n",  (tan->cd[1][0]*cdi[0][0] + tan->cd[1][1]*cdi[1][0] ));
	printf(":: %lf\n",  (tan->cd[1][0]*cdi[0][1] + tan->cd[1][1]*cdi[1][1] ));
	*/

	//assert( (tan->cd[0][0]*cdi[0][0] + tan->cd[0][1]*cdi[1][0] ) == 1.0);
	//assert( (tan->cd[0][0]*cdi[0][1] + tan->cd[0][1]*cdi[1][1] ) == 0.0);
	//assert( (tan->cd[1][0]*cdi[0][0] + tan->cd[1][1]*cdi[1][0] ) == 0.0);
	//assert( (tan->cd[1][0]*cdi[0][1] + tan->cd[1][1]*cdi[1][1] ) == 1.0);

	// FIXME be robust near the poles
	// Calculate intermediate world coordinates (x,y) on the tangent plane
	radecdeg2xyzarr(tan->crval[0],tan->crval[1],xyzcrval);
	if (!star_coords(xyzpt, xyzcrval, &y, &x))
		return FALSE;

	// Switch intermediate world coordinates into degrees
	x = rad2deg(x);
	y = rad2deg(y);

	// Linear pixel coordinates
	U = cdi[0][0]*x + cdi[0][1]*y;
	V = cdi[1][0]*x + cdi[1][1]*y;

	// Re-add crpix to get pixel coordinates
	*px = U + tan->crpix[0];
	*py = V + tan->crpix[1];
	return TRUE;
}

// RA,Dec in degrees to Pixels.
bool tan_radec2pixelxy(const tan_t* tan, double a, double d, double *px, double *py)
{
	double xyzpt[3];
	radecdeg2xyzarr(a,d,xyzpt);
	return tan_xyzarr2pixelxy(tan, xyzpt, px, py);
}

void sip_calc_distortion(const sip_t* sip, double u, double v, double* U, double *V)
{
	// Do SIP distortion (in relative pixel coordinates)
	// See the sip_t struct definition in header file for details
	int p, q;
	double fuv=0.;
	double guv=0.;
	for (p=0; p<=sip->a_order; p++)
		for (q=0; q<=sip->a_order; q++)
			if ((p+q > 0) && (p+q <= sip->a_order))
				fuv += sip->a[p][q]*pow(u,p)*pow(v,q);
	for (p=0; p<=sip->b_order; p++) 
		for (q=0; q<=sip->b_order; q++) 
			if ((p+q > 0) && (p+q <= sip->b_order))
				guv += sip->b[p][q]*pow(u,p)*pow(v,q);
	*U = u + fuv;
	*V = v + guv;
}

void sip_calc_inv_distortion(const sip_t* sip, double U, double V, double* u, double *v)
{
	int p, q;
	double fUV=0.;
	double gUV=0.;
	for (p=0; p<=sip->ap_order; p++)
		for (q=0; q<=sip->ap_order; q++)
			fUV += sip->ap[p][q]*pow(U,p)*pow(V,q);
	for (p=0; p<=sip->bp_order; p++) 
		for (q=0; q<=sip->bp_order; q++) 
			gUV += sip->bp[p][q]*pow(U,p)*pow(V,q);

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

// returns pixel scale in arcseconds (NOT arcsec^2)
double sip_pixel_scale(const sip_t* sip) {
	return tan_pixel_scale(&(sip->wcstan));
}

void sip_print_to(const sip_t* sip, FILE* f) {
   double det,pixsc;

	fprintf(f,"SIP Structure:\n");
	fprintf(f,"  crval=(%g, %g)\n", sip->wcstan.crval[0], sip->wcstan.crval[1]);
	fprintf(f,"  crpix=(%g, %g)\n", sip->wcstan.crpix[0], sip->wcstan.crpix[1]);
	fprintf(f,"  CD = ( %12.5g   %12.5g )\n", sip->wcstan.cd[0][0], sip->wcstan.cd[0][1]);
	fprintf(f,"       ( %12.5g   %12.5g )\n", sip->wcstan.cd[1][0], sip->wcstan.cd[1][1]);

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

	det = sip_det_cd(sip);
	pixsc = 3600*sqrt(fabs(det));
	fprintf(f,"  det(CD)=%g\n", det);
	fprintf(f,"  sqrt(det(CD))=%g [arcsec]\n", pixsc);
	//fprintf(f,"\n");
}

void sip_print(const sip_t* sip) {
	sip_print_to(sip, stderr);
}
