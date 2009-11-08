/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

InlineDefine Const int dimquad2dimcode(int dimquad) {
    return 2 * (dimquad - 2);
}

InlineDefine Const double rad2deg(double x) {
    return x * DEG_PER_RAD;
}

InlineDefine Const double deg2rad(double x) {
    return x * RAD_PER_DEG;
}

InlineDefine Const double deg2arcmin(double x) {
    return x * ARCMIN_PER_DEG;
}

InlineDefine Const double arcmin2deg(double x) {
    return x * DEG_PER_ARCMIN;
}

InlineDefine Const double arcmin2arcsec(double x) {
    return x * ARCSEC_PER_ARCMIN;
}

InlineDefine Const double arcsec2arcmin(double x) {
    return x * ARCMIN_PER_ARCSEC;
}

InlineDefine Const double rad2arcmin(double x) {
    return x * ARCMIN_PER_RAD;
}

InlineDefine Const double rad2arcsec(double x) {
    return x * ARCSEC_PER_RAD;
}

InlineDefine Const double deg2arcsec(double x) {
    return x * ARCSEC_PER_DEG;
}

InlineDefine Const double arcmin2rad(double x) {
    return x * RAD_PER_ARCMIN;
}

InlineDefine Const double arcsec2rad(double x) {
    return x * RAD_PER_ARCSEC;
}

InlineDefine Const double arcsec2deg(double x) {
    return x * DEG_PER_ARCSEC;
}

InlineDefine Const double rad2distsq(double x) {
	// inverse of distsq2arc; cosine law.
	return 2.0 * (1.0 - cos(x));
}

InlineDefine Const double rad2dist(double x) {
	return sqrt(rad2distsq(x));
}

InlineDefine Const double arcsec2distsq(double x) {
	return rad2distsq(arcsec2rad(x));
}

InlineDefine Const Flatten double arcmin2dist(double x) {
	return rad2dist(arcmin2rad(x));
}

InlineDefine Const Flatten double arcmin2distsq(double arcmin) {
    return rad2distsq(arcmin2rad(arcmin));
}

InlineDefine Const double z2dec(double z) {
	return asin(z);
}

InlineDefine Const double xy2ra(double x, double y) {
	double a = atan2(y, x);
	if (a < 0)
		a += 2.0 * M_PI;
	return a;
}

InlineDefine Flatten void xyz2radec(double x, double y, double z, double *ra, double *dec) {
	*ra = xy2ra(x, y);
	*dec = z2dec(z);
}

InlineDefine Flatten void xyzarr2radec(const double* xyz, double *ra, double *dec) {
	xyz2radec(xyz[0], xyz[1], xyz[2], ra, dec);
}

InlineDefine Flatten void xyzarr2radecdeg(const double* xyz, double *ra, double *dec) {
	xyzarr2radec(xyz, ra, dec);
    *ra  = rad2deg(*ra);
    *dec = rad2deg(*dec);
}

InlineDefine Flatten void xyzarr2radecdegarr(double* xyz, double *radec) {
	xyzarr2radecdeg(xyz, radec, radec+1);
}

InlineDefine void radec2xyzarr(double ra, double dec, double* xyz) {
	double cosdec = cos(dec);
	xyz[0] = cosdec * cos(ra);
	xyz[1] = cosdec * sin(ra);
	xyz[2] = sin(dec);
}

InlineDefine void radec2xyz(double ra, double dec,
							double* x, double* y, double* z) {
	double cosdec = cos(dec);
	*x = cosdec * cos(ra);
	*y = cosdec * sin(ra);
	*z = sin(dec);
}

InlineDefine void radecdeg2xyz(double ra, double dec,
							   double* x, double* y, double* z) {
	radec2xyz(deg2rad(ra), deg2rad(dec), x, y, z);
}

InlineDefine void radecdeg2xyzarr(double ra, double dec, double* xyz) {
	radec2xyzarr(deg2rad(ra),deg2rad(dec), xyz);
}

WarnUnusedResult InlineDefine bool star_coords(const double *s, const double *r, double *x, double *y) {
	double sdotr = s[0] * r[0] + s[1] * r[1] + s[2] * r[2];
	if (sdotr <= 0.0)
		return FALSE;
	if (unlikely(r[2] == 1.0)) {
		double inv_s2 = 1.0 / s[2];
		*x = s[0] * inv_s2;
		*y = s[1] * inv_s2;
	} else if (unlikely(r[2] == -1.0)) {
		double inv_s2 = 1.0 / s[2];
		*x =  s[0] * inv_s2;
		*y = -s[1] * inv_s2;
	} else {
		double etax, etay, etaz, xix, xiy, xiz, eta_norm;
		double inv_en, inv_sdotr;
		// eta is a vector perpendicular to r pointing in the direction of increasing RA
 		etax = -r[1];
		etay =  r[0];
		etaz = 0.0;
		eta_norm = hypot(etax, etay); //sqrt(etax * etax + etay * etay);
		inv_en = 1.0 / eta_norm;
		etax *= inv_en;
		etay *= inv_en;
		// xi =  r cross eta, a vector pointing northwards, in direction of increasing DEC
		xix = -r[2] * etay;
		xiy =  r[2] * etax;
		xiz =  r[0] * etay - r[1] * etax;
		inv_sdotr = 1.0 / sdotr;
		*x = (s[0] * xix + s[1] * xiy + s[2] * xiz) * inv_sdotr;
		*y = (s[0] * etax + s[1] * etay) * inv_sdotr;
	}
	return TRUE;
}
