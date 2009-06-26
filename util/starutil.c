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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/param.h>
#include <regex.h>

#include "mathutil.h"
#include "starutil.h"
#include "keywords.h"
#include "errors.h"

#define POGSON 2.51188643150958
#define LOGP   0.92103403719762

#define InlineDefine InlineDefineC
#include "starutil.inc"
#undef InlineDefine

void radecrange2xyzrange(double ralo, double declo, double rahi, double dechi,
						 double* minxyz, double* maxxyz) {
	double minmult, maxmult;
	double uxlo, uxhi, uylo, uyhi;
	// Dec only affects z, and is monotonic (z = sin(dec))
	minxyz[2] = radec2z(0, declo);
	maxxyz[2] = radec2z(0, dechi);

	// min,max of cos(dec).  cos(dec) is concave down.
	minmult = MIN(cos(deg2rad(declo)), cos(deg2rad(dechi)));
	maxmult = MAX(cos(deg2rad(declo)), cos(deg2rad(dechi)));
	if (declo < 0 && dechi > 0)
		maxmult = 1.0;
	// unscaled x (ie, cos(ra))
	uxlo = MIN(cos(deg2rad(ralo)), cos(deg2rad(rahi)));
	if (ralo < 180 && rahi > 180)
		uxlo = -1.0;
	uxhi = MAX(cos(deg2rad(ralo)), cos(deg2rad(rahi)));
	minxyz[0] = MIN(uxlo * minmult, uxlo * maxmult);
	maxxyz[0] = MAX(uxhi * minmult, uxhi * maxmult);
	// unscaled y (ie, sin(ra))
	uylo = MIN(sin(deg2rad(ralo)), sin(deg2rad(rahi)));
	if (ralo < 270 && rahi > 270)
		uylo = -1.0;
	uyhi = MAX(sin(deg2rad(ralo)), sin(deg2rad(rahi)));
	if (ralo < 90 && rahi > 90)
		uyhi = -1.0;
	minxyz[1] = MIN(uylo * minmult, uylo * maxmult);
	maxxyz[1] = MAX(uyhi * minmult, uyhi * maxmult);
}


static int parse_hms_string(const char* str,
                            int* sign, int* term1, int* term2, double* term3) {
    bool matched;
    regmatch_t matches[6];
    int nmatches = 6;
    regex_t re;
    regmatch_t* m;
    const char* s;

    const char* restr = 
        "^([+-])?([[:digit:]]{2}):"
        "([[:digit:]]{2}):"
        "([[:digit:]]*(\\.[[:digit:]]*)?)$";
    if (regcomp(&re, restr, REG_EXTENDED)) {
        ERROR("Failed to compile H:M:S regex \"%s\"", restr);
        return -1;
    }
    matched = (regexec(&re, str, nmatches, matches, 0) == 0);
    regfree(&re);
    if (!matched)
        return 1;

    // sign
    m = matches + 1;
    s = str + m->rm_so;
    if (m->rm_so == -1 || s[0] == '+')
        *sign = 1;
    else
        *sign = -1;
    // hrs / deg
    m = matches + 2;
    s = str + m->rm_so;
    if (s[0] == '0')
        s++;
    *term1 = atoi(s);
    // Min
    m = matches + 3;
    s = str + m->rm_so;
    if (s[0] == '0')
        s++;
    *term2 = atoi(s);
    // Sec
    m = matches + 4;
    s = str + m->rm_so;
    *term3 = atof(s);
    return 0;
}

double atora(const char* str) {
    char* eptr;
    double ra;
    int sgn, hr, min;
    double sec;
    int rtn;

    rtn = parse_hms_string(str, &sgn, &hr, &min, &sec);
    if (rtn == -1) {
        ERROR("Failed to run regex");
        return HUGE_VAL;
    }
    if (rtn == 0)
        return sgn * hms2ra(hr, min, sec);

    ra = strtod(str, &eptr);
    if (eptr == str)
        // no conversion
        return HUGE_VAL;
    return ra;
}

double atodec(const char* str) {
    char* eptr;
    double dec;
    int sgn, deg, min;
    double sec;
    int rtn;

    rtn = parse_hms_string(str, &sgn, &deg, &min, &sec);
    if (rtn == -1) {
        ERROR("Failed to run regex");
        return HUGE_VAL;
    }
    if (rtn == 0)
        return dms2dec(sgn, deg, min, sec);

    dec = strtod(str, &eptr);
    if (eptr == str)
        // no conversion
        return HUGE_VAL;
    return dec;
}

double mag2flux(double mag) {
	return pow(POGSON, -mag);
}

double flux2mag(double flux) {
	return -log(flux) * LOGP;
}

inline void xyz2radec(double x, double y, double z, double *ra, double *dec) {
	*ra = xy2ra(x, y);
	*dec = z2dec(z);
}

inline void xyzarr2radec(const double* xyz, double *ra, double *dec) {
	xyz2radec(xyz[0], xyz[1], xyz[2], ra, dec);
}

inline void xyzarr2radecarr(const double* xyz, double *radec) {
	xyz2radec(xyz[0], xyz[1], xyz[2], radec+0, radec+1);
}

inline void radecdegarr2xyzarr(double* radec, double* xyz) {
    radecdeg2xyzarr(radec[0], radec[1], xyz);
}

inline void xyzarr2radecdeg(const double* xyz, double *ra, double *dec) {
	xyzarr2radec(xyz, ra, dec);
    *ra  = rad2deg(*ra);
    *dec = rad2deg(*dec);
}

inline void xyzarr2radecdegarr(double* xyz, double *radec) {
	xyzarr2radecdeg(xyz, radec, radec+1);
}

inline void radec2xyz(double ra, double dec,
					  double* x, double* y, double* z) {
	double cosdec = cos(dec);
	*x = cosdec * cos(ra);
	*y = cosdec * sin(ra);
	*z = sin(dec);
}

double distsq_between_radecdeg(double ra1, double dec1,
                               double ra2, double dec2) {
    double xyz1[3];
    double xyz2[3];
    radecdeg2xyzarr(ra1, dec1, xyz1);
    radecdeg2xyzarr(ra2, dec2, xyz2);
    return distsq(xyz1, xyz2, 3);
}

double arcsec_between_radecdeg(double ra1, double dec1,
                               double ra2, double dec2) {
    return distsq2arcsec(distsq_between_radecdeg(ra1, dec1, ra2, dec2));
}

inline void radecdeg2xyz(double ra, double dec,
						 double* x, double* y, double* z) {
	radec2xyz(deg2rad(ra), deg2rad(dec), x, y, z);
}

inline void radec2xyzarr(double ra, double dec, double* xyz) {
	xyz[0] = cos(dec) * cos(ra);
	xyz[1] = cos(dec) * sin(ra);
	xyz[2] = sin(dec);
}
inline void radecdeg2xyzarr(double ra, double dec, double* xyz) {
	radec2xyzarr(deg2rad(ra),deg2rad(dec), xyz);
}

// xyz stored as xyzxyzxyz.
inline void radec2xyzarrmany(double *ra, double *dec, double* xyz, int n) {
	int i;
	for (i=0; i<n; i++) {
		radec2xyzarr(ra[i], dec[i], xyz+3*i);
	}
}

inline void radecdeg2xyzarrmany(double *ra, double *dec, double* xyz, int n) {
	int i;
	for (i=0; i<n; i++) {
		radec2xyzarr(deg2rad(ra[i]), deg2rad(dec[i]), xyz+3*i);
	}
}

inline void project_equal_area(double x, double y, double z, double* projx, double* projy) {
	double Xp = x*sqrt(1./(1. + z));
	double Yp = y*sqrt(1./(1. + z));
	Xp = 0.5 * (1.0 + Xp);
	Yp = 0.5 * (1.0 + Yp);
	assert(Xp >= 0.0 && Xp <= 1.0);
	assert(Yp >= 0.0 && Yp <= 1.0);
	*projx = Xp;
	*projy = Yp;
}

inline void project_hammer_aitoff_x(double x, double y, double z, double* projx, double* projy) {
	double theta = atan(x/z);
	double r = sqrt(x*x+z*z);
	double zp, xp;
	/* Hammer-Aitoff projection with x-z plane compressed to purely +z side
	 * of xz plane */
	if (z < 0) {
		if (x < 0) {
			theta = theta - M_PI;
		} else {
			theta = M_PI + theta;
		}
	}
	theta /= 2.0;
	zp = r*cos(theta);
	xp = r*sin(theta);
	assert(zp >= -0.01);
	project_equal_area(xp, y, zp, projx, projy);
}

/* makes a star object located uniformly at random within the limits given
   on the sphere */
void make_rand_star(double* star, double ramin, double ramax,
					double decmin, double decmax)
{
	double decval, raval;
	if (ramin < 0.0)
		ramin = 0.0;
	if (ramax > (2*M_PI))
		ramax = 2 * M_PI;
	if (decmin < -M_PI / 2.0)
		decmin = -M_PI / 2.0;
	if (decmax > M_PI / 2.0)
		decmax = M_PI / 2.0;

	decval = asin(uniform_sample(sin(decmin), sin(decmax)));
	raval = uniform_sample(ramin, ramax);
	star[0] = radec2x(raval, decval);
	star[1] = radec2y(raval, decval);
	star[2] = radec2z(raval, decval);
}

// arc in radians
Const inline double arc2distsq(double arcInRadians) {
	// inverse of distsq2arc; cosine law.
	return 2.0 * (1.0 - cos(arcInRadians));
}

Const inline double rad2distsq(double arcInRadians) {
	return arc2distsq(arcInRadians);
}

Const inline double rad2dist(double arcInRadians) {
	return sqrt(rad2distsq(arcInRadians));
}

Const inline double arcsec2distsq(double arcInArcSec) {
   return arc2distsq(arcsec2rad(arcInArcSec));
}

Const inline double arcsec2dist(double arcInArcSec) {
   return sqrt(arcsec2distsq(arcInArcSec));
}

// Degrees to distance on the unit sphere.
Const inline double deg2dist(double arcInDegrees) {
    return arcsec2dist(deg2arcsec(arcInDegrees));
}

Const inline double arcmin2dist(double arcmin) {
    return rad2dist(arcmin2rad(arcmin));
}

Const inline double arcmin2distsq(double arcmin) {
    return arc2distsq(arcmin2rad(arcmin));
}

Const inline double dist2deg(double dist) {
    return arcsec2deg(dist2arcsec(dist));
}

Const inline double distsq2arcsec(double dist2) {
	return rad2arcsec(distsq2arc(dist2));
}

Const inline double dist2arcsec(double dist) {
	return distsq2arcsec(dist*dist);
}

Const inline double distsq2arc(double dist2) {
	// cosine law: c^2 = a^2 + b^2 - 2 a b cos C
	// c^2 is dist2.  We want C.
	// a = b = 1
	// c^2 = 1 + 1 - 2 cos C
	// dist2 = 2( 1 - cos C )
	// 1 - (dist2 / 2) = cos C
	// C = acos(1 - dist2 / 2)
	return acos(1.0 - dist2 / 2.0);
}

Const inline double distsq2rad(double dist2) {
	return distsq2arc(dist2);
}

Const inline double dist2rad(double dist) {
	return distsq2arc(dist*dist);
}


// RA in degrees to Mercator X coordinate [0, 1).
inline double ra2mercx(double ra) {
    double mx = ra / 360.0;
    if (mx < 0.0 || mx > 1.0) {
        mx = fmod(mx, 1.0);
        if (mx < 0.0)
            mx += 1.0;
    }
    return mx;
}

// Dec in degrees to Mercator Y coordinate [0, 1).
inline double dec2mercy(double dec) {
	return 0.5 + (asinh(tan(deg2rad(dec))) / (2.0 * M_PI));
}

double hms2ra(int h, int m, double s) {
    return 15.0 * (h + ((m + (s / 60.0)) / 60.0));
}

double dms2dec(int sgn, int d, int m, double s) {
    return sgn * (d + ((m + (s / 60.0)) / 60.0));
}

// RA in degrees to H:M:S
inline void ra2hms(double ra, int* h, int* m, double* s) {
    double rem;
    ra = fmod(ra, 360.0);
    if (ra < 0.0)
        ra += 360.0;
    rem = ra / 15.0;
    *h = floor(rem);
    // remaining (fractional) hours
    rem -= *h;
    // -> minutes
    rem *= 60.0;
    *m = floor(rem);
    // remaining (fractional) minutes
    rem -= *m;
    // -> seconds
    rem *= 60.0;
    *s = rem;
}

// Dec in degrees to D:M:S
inline void dec2dms(double dec, int* d, int* m, double* s) {
    double rem;
    double flr;
    int sign;
    sign = (dec >= 0.0) ? 1 : -1;
    dec *= sign;
    flr = floor(dec);
    *d = sign * flr;
    // remaining degrees:
    rem = dec - flr;
    // -> minutes
    rem *= 60.0;
    *m = floor(rem);
    // remaining (fractional) minutes
    rem -= *m;
    // -> seconds
    rem *= 60.0;
    *s = rem;
}

/* computes the 2D coordinates (x,y) (in units of a celestial sphere of radius 1)
   that star s would have in a TANGENTIAL PROJECTION defined by (centred at) star r.
   s and r are both given in xyz coordinates, the parameters are pointers to arrays of size3
	WARNING -- this code assumes s and r are UNIT vectors (ie normalized to have length 1)
	the resulting x direction is increasing DEC, the resulting y direction is increasing RA
	which might not be the normal convention
*/
inline bool star_coords(const double *s, const double *r, double *x, double *y)
{
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

inline void star_midpoint(double* mid, const double* A, const double* B) {
	double len;
	double invlen;
	// we don't divide by 2 because we immediately renormalize it...
	mid[0] = A[0] + B[0];
	mid[1] = A[1] + B[1];
	mid[2] = A[2] + B[2];
	//len = sqrt(square(mid[0]) + square(mid[1]) + square(mid[2]));
	len = sqrt(mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]);
	invlen = 1.0 / len;
	mid[0] *= invlen;
	mid[1] *= invlen;
	mid[2] *= invlen;
}
