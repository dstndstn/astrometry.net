/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <regex.h>

#include "os-features.h"
#include "keywords.h"
#include "mathutil.h"
#include "starutil.h"
#include "errors.h"

#define POGSON 2.51188643150958
#define LOGP   0.92103403719762

#define InlineDefine InlineDefineC
#include "starutil.inc"
#undef InlineDefine

void radec_derivatives(double ra, double dec, double* dra, double* ddec) {
    double cosd = cos(deg2rad(dec));
    double cosra = cos(deg2rad(ra));
    double sinra = sin(deg2rad(ra));
    if (dra) {
        dra[0] = cosd * -sinra;
        dra[1] = cosd *  cosra;
        dra[2] = 0.0;
        normalize_3(dra);
    }
    if (ddec) {
        double sind = sin(deg2rad(dec));
        ddec[0] = -sind * cosra;
        ddec[1] = -sind * sinra;
        ddec[2] =  cosd;
        normalize_3(ddec);
    }
}

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
    anbool matched;
    regmatch_t matches[6];
    int nmatches = 6;
    regex_t re;
    regmatch_t* m;
    const char* s;

    const char* restr = 
        "^([+-])?([[:digit:]]{1,2}):"
        "([[:digit:]]{1,2}):"
        "([[:digit:]]*(\\.[[:digit:]]*)?)$";

    if (!str)
        return 1;

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

    if (!str) {
        //ERROR("Null string to atora()");
        return LARGE_VAL;
    }
    rtn = parse_hms_string(str, &sgn, &hr, &min, &sec);
    if (rtn == -1) {
        ERROR("Failed to run regex");
        return LARGE_VAL;
    }
    if (rtn == 0)
        return sgn * hms2ra(hr, min, sec);

    ra = strtod(str, &eptr);
    if (eptr == str)
        // no conversion
        return LARGE_VAL;
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
        return LARGE_VAL;
    }
    if (rtn == 0)
        return dms2dec(sgn, deg, min, sec);

    dec = strtod(str, &eptr);
    if (eptr == str)
        // no conversion
        return LARGE_VAL;
    return dec;
}

double mag2flux(double mag) {
    return pow(POGSON, -mag);
}

double flux2mag(double flux) {
    return -log(flux) * LOGP;
}

inline void xyzarr2radecarr(const double* xyz, double *radec) {
    xyz2radec(xyz[0], xyz[1], xyz[2], radec+0, radec+1);
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

double deg_between_radecdeg(double ra1, double dec1, double ra2, double dec2) {
    return arcsec2deg(arcsec_between_radecdeg(ra1, dec1, ra2, dec2));
}

void project_equal_area(double x, double y, double z, double* projx, double* projy) {
    double Xp = x*sqrt(1./(1. + z));
    double Yp = y*sqrt(1./(1. + z));
    Xp = 0.5 * (1.0 + Xp);
    Yp = 0.5 * (1.0 + Yp);
    assert(Xp >= 0.0 && Xp <= 1.0);
    assert(Yp >= 0.0 && Yp <= 1.0);
    *projx = Xp;
    *projy = Yp;
}

void project_hammer_aitoff_x(double x, double y, double z, double* projx, double* projy) {
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
void dec2dms(double dec, int* sign, int* d, int* m, double* s) {
    double rem;
    double flr;
    *sign = (dec >= 0.0) ? 1 : -1;
    dec *= (*sign);
    flr = floor(dec);
    *d = flr;
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

void ra2hmsstring(double ra, char* str) {
    int h, m;
    double s;
    int ss;
    int ds;
    ra2hms(ra, &h, &m, &s);

    // round to display to 3 decimal places
    ss = (int)floor(s);
    ds = (int)round((s - ss) * 1000.0);
    if (ds >= 1000) {
        ss++;
        ds -= 1000;
    }
    if (ss >= 60) {
        ss -= 60;
        m += 1;
    }
    if (m >= 60) {
        m -= 60;
        h += 1;
    }
    sprintf(str, "%02i:%02i:%02i.%03i", h, m, ss, ds);
}

void dec2dmsstring(double dec, char* str) {
    int sign, d, m;
    double s;
    int ss, ds;
    dec2dms(dec, &sign, &d, &m, &s);
    ss = (int)floor(s);
    ds = (int)round((s - ss) * 1000.0);
    if (ds >= 1000) {
        ss++;
        ds -= 1000;
    }
    if (ss >= 60) {
        ss -= 60;
        m += 1;
    }
    if (m >= 60) {
        m -= 60;
        d += 1;
    }
    sprintf(str, "%c%02i:%02i:%02i.%03i", (sign==1 ? '+':'-'), d, m, ss, ds);
}

