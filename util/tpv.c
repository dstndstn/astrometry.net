/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Keir Mierle, David W. Hogg, Sam Roweis and Dustin Lang.
  Copyright 2012 Dustin Lang.

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
#include <string.h>
#include <assert.h>
#include <sys/param.h>

#include "tpv.h"
#include "starutil.h"
#include "mathutil.h"

static anbool has_distortions(const tpv_t* tpv) {
	return (tpv->a_order >= 0);
}

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
	if (has_distortions(tpv)) {
		double U, V;
		tpv_distortion(tpv, px, py, &U, &V);
		// Run a normal TAN conversion on the distorted pixel coords.
		tan_pixelxy2radec(&(tpv->wcstan), U, V, ra, dec);
	} else
		// Run a normal TAN conversion
		tan_pixelxy2radec(&(tpv->wcstan), px, py, ra, dec);
}

// Pixels to Intermediate World Coordinates in degrees.
void tpv_pixelxy2iwc(const tpv_t* tpv, double px, double py,
					 double *iwcx, double* iwcy) {
	if (has_distortions(tpv)) {
		double U, V;
		tpv_distortion(tpv, px, py, &U, &V);
		// Run a normal TAN conversion on the distorted pixel coords.
		tan_pixelxy2iwc(&(tpv->wcstan), U, V, iwcx, iwcy);
	} else
		// Run a normal TAN conversion
		tan_pixelxy2iwc(&(tpv->wcstan), px, py, iwcx, iwcy);
}

// Pixels to XYZ unit vector.
void tpv_pixelxy2xyzarr(const tpv_t* tpv, double px, double py, double *xyz) {
	if (has_distortions(tpv)) {
		double U, V;
		tpv_distortion(tpv, px, py, &U, &V);
		// Run a normal TAN conversion on the distorted pixel coords.
		tan_pixelxy2xyzarr(&(tpv->wcstan), U, V, xyz);
	} else
		// Run a normal TAN conversion
		tan_pixelxy2xyzarr(&(tpv->wcstan), px, py, xyz);
}

void tpv_iwc2pixelxy(const tpv_t* tpv, double u, double v,
					 double *px, double* py) {
    double x,y;
    tan_iwc2pixelxy(&(tpv->wcstan), u, v, &x, &y);
    tpv_pixel_undistortion(tpv, x, y, px, py);
}

// TPV:  (RA,Dec) --> IWC --> x,y --> x',y'

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


// RA,Dec in degrees to Pixels.
anbool tpv_radec2pixelxy_check(const tpv_t* tpv, double ra, double dec, double *px, double *py) {
	double u, v;
	double U, V;
	double U2, V2;
	if (!tan_radec2pixelxy(&(tpv->wcstan), ra, dec, px, py))
		return FALSE;
	if (!has_distortions(tpv))
		return TRUE;

	// Subtract crpix, invert TPV distortion, add crpix.
	// Sanity check:
	if (tpv->a_order != 0 && tpv->ap_order == 0) {
		fprintf(stderr, "suspicious inversion; no inversion TPV coeffs "
				"yet there are forward TPV coeffs\n");
	}
	U = *px - tpv->wcstan.crpix[0];
	V = *py - tpv->wcstan.crpix[1];
	tpv_calc_inv_distortion(tpv, U, V, &u, &v);
    // Check that we're dealing with the right range of the polynomial by inverting it and
    // checking that we end up back in the right place.
    tpv_calc_distortion(tpv, u, v, &U2, &V2);
    if (fabs(U2 - U) + fabs(V2 - V) > 10.0)
        return FALSE;
	*px = u + tpv->wcstan.crpix[0];
	*py = v + tpv->wcstan.crpix[1];
	return TRUE;
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
	if (!has_distortions(tpv)) {
        *X = x;
        *Y = y;
        return;
    }
	// Sanity check:
	if (tpv->a_order != 0 && tpv->ap_order == 0) {
		fprintf(stderr, "suspicious inversion; no inverse TPV coeffs "
				"yet there are forward TPV coeffs\n");
	}

	// Get pixel coordinates relative to reference pixel
	double u = x - tpv->wcstan.crpix[0];
	double v = y - tpv->wcstan.crpix[1];
	tpv_calc_inv_distortion(tpv, u, v, X, Y);
	*X += tpv->wcstan.crpix[0];
	*Y += tpv->wcstan.crpix[1];
}

void tpv_calc_inv_distortion(const tpv_t* tpv, double U, double V, double* u, double *v)
{
	int p, q;
	double fUV=0.;
	double gUV=0.;

    // avoid using pow() function
    double powu[TPV_MAXORDER];
    double powv[TPV_MAXORDER];
    powu[0] = 1.0;
    powu[1] = U;
    powv[0] = 1.0;
    powv[1] = V;
	for (p=2; p <= MAX(tpv->ap_order, tpv->bp_order); p++) {
        powu[p] = powu[p-1] * U;
        powv[p] = powv[p-1] * V;
    }

	for (p=0; p<=tpv->ap_order; p++)
		for (q=0; q<=tpv->ap_order; q++)
			//fUV += tpv->ap[p][q] * pow(U,p) * pow(V,q);
			fUV += tpv->ap[p][q] * powu[p] * powv[q];
	for (p=0; p<=tpv->bp_order; p++) 
		for (q=0; q<=tpv->bp_order; q++) 
			//gUV += tpv->bp[p][q] * pow(U,p) * pow(V,q);
            gUV += tpv->bp[p][q] * powu[p] * powv[q];

	*u = U + fUV;
	*v = V + gUV;
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

   if (tpv->wcstan.sin) {
	   print_to(&(tpv->wcstan), f, "SIN-TPV");
   } else {
	   print_to(&(tpv->wcstan), f, "TAN-TPV");
   }

   fprintf(f, "  TPV order: A=%i, B=%i, AP=%i, BP=%i\n",
		   tpv->a_order, tpv->b_order, tpv->ap_order, tpv->bp_order);

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

	if (tpv->ap_order > 0) {
		int p, q;
		for (p=0; p<=tpv->ap_order; p++) {
			fprintf(f, (p ? "      " : "  AP = "));
			for (q=0; q<=tpv->ap_order; q++)
				if (p+q <= tpv->ap_order)
					fprintf(f,"%12.5g", tpv->ap[p][q]);
			fprintf(f,"\n");
		}
	}
	if (tpv->bp_order > 0) {
		int p, q;
		for (p=0; p<=tpv->bp_order; p++) {
			fprintf(f, (p ? "      " : "  BP = "));
			for (q=0; q<=tpv->bp_order; q++)
				if (p+q <= tpv->bp_order)
					fprintf(f,"%12.5g", tpv->bp[p][q]);
			fprintf(f,"\n");
		}
	}


	det = tpv_det_cd(tpv);
	pixsc = 3600*sqrt(fabs(det));
	//fprintf(f,"  det(CD)=%g\n", det);
	fprintf(f,"  sqrt(det(CD))=%g [arcsec]\n", pixsc);
	//fprintf(f,"\n");
}

void tpv_print(const tpv_t* tpv) {
	tpv_print_to(tpv, stderr);
}

double tpv_get_orientation(const tpv_t* tpv) {
    return tan_get_orientation(&(tpv->wcstan));
}
