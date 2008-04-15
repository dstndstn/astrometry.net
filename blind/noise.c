/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

/*
  Utilities for noise simulations.
*/

#include <math.h>
#include <assert.h>

#include "noise.h"
#include "mathutil.h"
#include "starutil.h"

void sample_in_circle(const double* center, double radius,
                      double* point) {
    int i;
    for (i=0; i<2; i++) {
        point[i] = gaussian_sample(center[i], radius);
    }
}

void sample_star_in_circle(double* center, double ABangle,
						   double* point) {
	double va[3], vb[3];
	double noisemag, noiseangle;
	int i;

	tan_vectors(center, va, vb);
	noisemag = sqrt(arcsec2distsq(60.0 * ABangle) * uniform_sample(0.0, 1.0));
	noiseangle = uniform_sample(0.0, 2.0*M_PI);
	for (i=0; i<3; i++)
		point[i] = center[i] +
			cos(noiseangle) * noisemag * va[i] +
			sin(noiseangle) * noisemag * vb[i];
	normalize_3(point);
}

void sample_star_in_ring(double* center,
						 double minAngle,
						 double maxAngle,
						 double* point) {
	double va[3], vb[3];
	double noisemag, noiseangle;
	int i;
	double d2min, d2max;

	tan_vectors(center, va, vb);
	d2min = arcsec2distsq(60.0 * minAngle);
	d2max = arcsec2distsq(60.0 * maxAngle);
	noisemag = sqrt(uniform_sample(d2min, d2max));
	noiseangle = uniform_sample(0.0, 2.0*M_PI);
	for (i=0; i<3; i++)
		point[i] = center[i] +
			cos(noiseangle) * noisemag * va[i] +
			sin(noiseangle) * noisemag * vb[i];
	normalize_3(point);
}

void add_star_noise(const double* real, double noisestd, double* noisy) {
	double va[3], vb[3];
	//double mag, angle;
    double mag1, mag2;
	int i;

	tan_vectors(real, va, vb);
    /*
     // magnitude of noise
     mag = gaussian_sample(0.0, noisestd);
     // direction
     angle = uniform_sample(0.0, 2.0*M_PI);
     // magnitude in the two tangential directions:
     mag1 = mag * sin(angle);
     mag2 = mag * cos(angle);
     */
    mag1 = gaussian_sample(0.0, noisestd);
    mag2 = gaussian_sample(0.0, noisestd);
	for (i=0; i<3; i++)
		noisy[i] = real[i] + mag1 * va[i] + mag2 * vb[i];
	normalize_3(noisy);
}

void add_field_noise(const double* real, double noisestd, double* noisy) {
    int i;
    for (i=0; i<2; i++) {
        noisy[i] = gaussian_sample(real[i], noisestd);
    }
}

void compute_star_code(const double* xyz, int dimquads, double* code) {
	double midAB[3];
	double Ax, Ay;
	double Bx, By;
	double scale, invscale;
	double ABx, ABy;
	double costheta, sintheta;
	double Dx, Dy;
	double ADx, ADy;
    bool ok = TRUE;
    int i;
    const double* A = xyz;
    const double* B = xyz + 3;

	star_midpoint(midAB, A, B);
	ok &= star_coords(A, midAB, &Ax, &Ay);
	ok &= star_coords(B, midAB, &Bx, &By);
	ABx = Bx - Ax;
	ABy = By - Ay;
	scale = (ABx * ABx) + (ABy * ABy);
	invscale = 1.0 / scale;
	costheta = (ABy + ABx) * invscale;
	sintheta = (ABy - ABx) * invscale;
    for (i=2; i<dimquads; i++) {
        ok &= star_coords(xyz + 3*i, midAB, &Dx, &Dy);
        assert(ok);
        ADx = Dx - Ax;
        ADy = Dy - Ay;
        code[(i-2)*2 + 0] =  ADx * costheta + ADy * sintheta;
        code[(i-2)*2 + 1] = -ADx * sintheta + ADy * costheta;
    }
}

void compute_field_code(const double* pix, int dimquads,
						double* code, double* p_scale) {
	double Ax, Ay, Bx, By, dx, dy, scale;
	double costheta, sintheta;
	double Cx, Cy, xxtmp;
    int i;

	Ax = pix[0];
	Ay = pix[1];
	Bx = pix[2];
	By = pix[3];
	dx = Bx - Ax;
	dy = By - Ay;
	scale = dx*dx + dy*dy;
	costheta = (dy + dx) / scale;
	sintheta = (dy - dx) / scale;

    for (i=2; i<dimquads; i++) {
        Cx = pix[(i*2) + 0];
        Cy = pix[(i*2) + 1];
        Cx -= Ax;
        Cy -= Ay;
        xxtmp = Cx;
        Cx =     Cx * costheta + Cy * sintheta;
        Cy = -xxtmp * sintheta + Cy * costheta;
        code[2*(i-2) + 0] = Cx;
        code[2*(i-2) + 1] = Cy;
    }
	if (p_scale)
		*p_scale = scale;
}


