/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang.

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
#include <sys/param.h>
#include <stdlib.h>
#include <assert.h>

#include "sip-utils.h"
#include "starutil.h"
#include "mathutil.h"

bool sip_pixel_is_inside_image(const sip_t* wcs, double x, double y) {
	return (x >= 1 && x <= wcs->wcstan.imagew && y >= 1 && y <= wcs->wcstan.imageh);
}

bool sip_is_inside_image(const sip_t* wcs, double ra, double dec) {
	double x,y;
	if (!sip_radec2pixelxy(wcs, ra, dec, &x, &y))
		return FALSE;
	return sip_pixel_is_inside_image(wcs, x, y);
}

int* sip_filter_stars_in_field(const sip_t* sip, const tan_t* tan,
							   const double* xyz, const double* radec,
							   int N, double** p_xy, int* inds, int* p_Ngood) {
	int i, Ngood;
	int W, H;
	double* xy = NULL;
	bool allocd = FALSE;
	
	assert(sip || tan);
	assert(xyz || radec);
	assert(p_Ngood);

	Ngood = 0;
	if (!inds) {
		inds = malloc(N * sizeof(int));
		allocd = TRUE;
	}

	if (p_xy)
		xy = malloc(N * 2 * sizeof(double));

	if (sip) {
		W = sip->wcstan.imagew;
		H = sip->wcstan.imageh;
	} else {
		W = tan->imagew;
		H = tan->imageh;
	}

	for (i=0; i<N; i++) {
		double x, y;
		if (xyz) {
			if (sip) {
				if (!sip_xyzarr2pixelxy(sip, xyz + i*3, &x, &y))
					continue;
			} else {
				if (!tan_xyzarr2pixelxy(tan, xyz + i*3, &x, &y))
					continue;
			}
		} else {
			if (sip) {
				if (!sip_radec2pixelxy(sip, radec[i*2], radec[i*2+1], &x, &y))
					continue;
			} else {
				if (!tan_radec2pixelxy(tan, radec[i*2], radec[i*2+1], &x, &y))
					continue;
			}
		}
		// FIXME -- check half- and one-pixel FITS issues.
		if ((x < 0) || (y < 0) || (x >= W) || (y >= H))
			continue;

		inds[Ngood] = i;
		if (xy) {
			xy[Ngood * 2 + 0] = x;
			xy[Ngood * 2 + 1] = y;
		}
		Ngood++;
	}

	if (allocd)
		inds = realloc(inds, Ngood * sizeof(int));

	if (xy)
		xy = realloc(xy, Ngood * 2 * sizeof(double));
	if (p_xy)
		*p_xy = xy;

	*p_Ngood = Ngood;
	
	return inds;
}

void sip_get_radec_center(const sip_t* wcs,
                          double* p_ra, double* p_dec) {
    double px = (wcs->wcstan.imagew + 1.0) / 2.0;
    double py = (wcs->wcstan.imageh + 1.0) / 2.0;
	sip_pixelxy2radec(wcs, px, py, p_ra, p_dec);
}

void sip_get_radec_center_hms(const sip_t* wcs,
                              int* rah, int* ram, double* ras,
                              int* decd, int* decm, double* decs) {
    double ra, dec;
    sip_get_radec_center(wcs, &ra, &dec);
    ra2hms(ra, rah, ram, ras);
    dec2dms(dec, decd, decm, decs);
}

void sip_get_radec_center_hms_string(const sip_t* wcs,
                                     char* rastr, char* decstr) {
    int rah, ram, decd, decm;
    double ras, decs;
    sip_get_radec_center_hms(wcs, &rah, &ram, &ras, &decd, &decm, &decs);
    sprintf(rastr, "%02i:%02i:%02.3g", rah, ram, ras);
    sprintf(decstr, "%+02i:%02i:%02.3g", decd, decm, decs);
}

void sip_get_field_size(const sip_t* wcs,
                        double* pw, double* ph,
                        char** units) {
    double minx = 0.5;
    double maxx = (wcs->wcstan.imagew + 0.5);
    double midx = (minx + maxx) / 2.0;
    double miny = 0.5;
    double maxy = (wcs->wcstan.imageh + 0.5);
    double midy = (miny + maxy) / 2.0;
    double ra1, dec1, ra2, dec2, ra3, dec3;
    double w, h;

    // measure width through the middle
	sip_pixelxy2radec(wcs, minx, midy, &ra1, &dec1);
	sip_pixelxy2radec(wcs, midx, midy, &ra2, &dec2);
	sip_pixelxy2radec(wcs, maxx, midy, &ra3, &dec3);
    w = arcsec_between_radecdeg(ra1, dec1, ra2, dec2) +
        arcsec_between_radecdeg(ra2, dec2, ra3, dec3);
    // measure height through the middle
	sip_pixelxy2radec(wcs, midx, miny, &ra1, &dec1);
	sip_pixelxy2radec(wcs, midx, midy, &ra2, &dec2);
	sip_pixelxy2radec(wcs, midx, maxy, &ra3, &dec3);
    h = arcsec_between_radecdeg(ra1, dec1, ra2, dec2) +
        arcsec_between_radecdeg(ra2, dec2, ra3, dec3);

    if (MIN(w, h) < 60.0) {
        *units = "arcseconds";
        *pw = w;
        *ph = h;
    } else if (MIN(w, h) < 3600.0) {
        *units = "arcminutes";
        *pw = w / 60.0;
        *ph = h / 60.0;
    } else {
        *units = "degrees";
        *pw = w / 3600.0;
        *ph = h / 3600.0;
    }
}

void sip_walk_image_boundary(const sip_t* wcs, double stepsize,
							 void (*callback)(const sip_t* wcs, double x, double y, double ra, double dec, void* token),
							 void* token) {
    int i, side;
    // Walk the perimeter of the image in steps of stepsize pixels
    double W = wcs->wcstan.imagew;
    double H = wcs->wcstan.imageh;
    {
		double Xmin = 0.5;
		double Xmax = W + 0.5;
		double Ymin = 0.5;
		double Ymax = H + 0.5;
        double offsetx[] = { Xmin, Xmax, Xmax, Xmin };
        double offsety[] = { Ymin, Ymin, Ymax, Ymax };
        double stepx[] = { +stepsize, 0, -stepsize, 0 };
        double stepy[] = { 0, +stepsize, 0, -stepsize };
        int Nsteps[] = { ceil(W/stepsize), ceil(H/stepsize), ceil(W/stepsize), ceil(H/stepsize) };

        for (side=0; side<4; side++) {
            for (i=0; i<Nsteps[side]; i++) {
                double ra, dec;
                double x, y;
                x = MIN(Xmax, MAX(Xmin, offsetx[side] + i * stepx[side]));
                y = MIN(Ymax, MAX(Ymin, offsety[side] + i * stepy[side]));
                sip_pixelxy2radec(wcs, x, y, &ra, &dec);
				callback(wcs, x, y, ra, dec, token);
            }
        }
    }
}

struct radecbounds {
	double rac, decc;
    double ramin, ramax, decmin, decmax;
};

static void radec_bounds_callback(const sip_t* wcs, double x, double y, double ra, double dec, void* token) {
	struct radecbounds* b = token;
	b->decmin = MIN(b->decmin, dec);
	b->decmax = MAX(b->decmax, dec);
	if (ra - b->rac > 180)
		// wrap-around: racenter < 180, ra has gone < 0 but been wrapped around to > 180
		ra -= 360;
	if (b->rac - ra > 180)
		// wrap-around: racenter > 180, ra has gone > 360 but wrapped around to > 0.
		ra += 360;

	b->ramin = MIN(b->ramin, ra);
	b->ramax = MAX(b->ramax, ra);
}

void sip_get_radec_bounds(const sip_t* wcs, int stepsize,
                          double* pramin, double* pramax,
                          double* pdecmin, double* pdecmax) {
	struct radecbounds b;

	sip_get_radec_center(wcs, &(b.rac), &(b.decc));
	b.ramin  = b.ramax = b.rac;
	b.decmin = b.decmax = b.decc;
	sip_walk_image_boundary(wcs, stepsize, radec_bounds_callback, &b);

	// Check for poles...
	// north pole
	if (sip_is_inside_image(wcs, 0, 90)) {
		b.ramin = 0;
		b.ramax = 360;
		b.decmax = 90;
	}
	if (sip_is_inside_image(wcs, 0, -90)) {
		b.ramin = 0;
		b.ramax = 360;
		b.decmin = -90;
	}

    if (pramin) *pramin = b.ramin;
    if (pramax) *pramax = b.ramax;
    if (pdecmin) *pdecmin = b.decmin;
    if (pdecmax) *pdecmax = b.decmax;
}

