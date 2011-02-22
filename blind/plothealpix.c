/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2010, 2011 Dustin Lang.

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
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "plothealpix.h"
#include "sip-utils.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "healpix-utils.h"
#include "healpix.h"

DEFINE_PLOTTER(healpix);

plothealpix_t* plot_healpix_get(plot_args_t* pargs) {
	return plotstuff_get_config(pargs, "healpix");
}

void* plot_healpix_init(plot_args_t* plotargs) {
	plothealpix_t* args = calloc(1, sizeof(plothealpix_t));
	args->nside = 1;
	args->stepsize = 50;
	return args;
}

int plot_healpix_plot(const char* command,
					  cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plothealpix_t* args = (plothealpix_t*)baton;
	double ra,dec,rad;
	il* hps;
	int i;
	double hpstep;
	int minx[12], maxx[12], miny[12], maxy[12];

	plotstuff_builtin_apply(cairo, pargs);

	if (plotstuff_get_radec_center_and_radius(pargs, &ra, &dec, &rad)) {
		ERROR("Failed to get RA,Dec center and radius");
		return -1;
	}
	hps = healpix_rangesearch_radec(ra, dec, rad, args->nside, NULL);
	logmsg("Found %i healpixes in range.\n", il_size(hps));
	hpstep = args->nside * args->stepsize * plotstuff_pixel_scale(pargs) / 60.0 / healpix_side_length_arcmin(args->nside);
	hpstep = MIN(1, hpstep);
	logmsg("Taking steps of %g in healpix space\n", hpstep);

	// For each of the 12 top-level healpixes, find the range of healpixes covered by this image.
	for (i=0; i<12; i++) {
		maxx[i] = maxy[i] = -1;
		minx[i] = miny[i] = args->nside+1;
	}
	for (i=0; i<il_size(hps); i++) {
		int hp = il_get(hps, i);
		int hpx, hpy;
		int bighp;
		healpix_decompose_xy(hp, &bighp, &hpx, &hpy, args->nside);
		logverb("  hp %i: bighp %i, x,y (%i,%i)\n", i, bighp, hpx, hpy);
		minx[bighp] = MIN(minx[bighp], hpx);
		maxx[bighp] = MAX(maxx[bighp], hpx);
		miny[bighp] = MIN(miny[bighp], hpy);
		maxy[bighp] = MAX(maxy[bighp], hpy);
	}
	il_free(hps);

	for (i=0; i<12; i++) {
		int hx,hy;
		int hp;
		double d, frac;
		double x,y;

		if (maxx[i] == -1)
			continue;
		logverb("Big healpix %i: x range [%i, %i], y range [%i, %i]\n",
			   i, minx[i], maxx[i], miny[i], maxy[i]);

		for (hy = miny[i]; hy <= maxy[i]; hy++) {
			logverb("  y=%i\n", hy);
			for (d=minx[i]; d<=maxx[i]; d+=hpstep) {
				hx = floor(d);
				frac = d - hx;
				hp = healpix_compose_xy(i, hx, hy, args->nside);
				healpix_to_radecdeg(hp, args->nside, frac, 0.0, &ra, &dec);
				if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y))
					continue;
				if (d == minx[i])
					cairo_move_to(pargs->cairo, x, y);
				else
					cairo_line_to(pargs->cairo, x, y);
			}
			cairo_stroke(pargs->cairo);
		}
		for (hx = minx[i]; hx <= maxx[i]; hx++) {
			for (d=miny[i]; d<=maxy[i]; d+=hpstep) {
				hy = floor(d);
				frac = d - hy;
				hp = healpix_compose_xy(i, hx, hy, args->nside);
				healpix_to_radecdeg(hp, args->nside, 0.0, frac, &ra, &dec);
				if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y))
					continue;
				if (d == miny[i])
					cairo_move_to(pargs->cairo, x, y);
				else
					cairo_line_to(pargs->cairo, x, y);
			}
			cairo_stroke(pargs->cairo);
		}
	}
	return 0;
}

int plot_healpix_command(const char* cmd, const char* cmdargs,
						 plot_args_t* pargs, void* baton) {
	plothealpix_t* args = (plothealpix_t*)baton;
	if (streq(cmd, "healpix_nside")) {
		args->nside = atoi(cmdargs);
	} else if (streq(cmd, "healpix_stepsize")) {
		args->stepsize = atoi(cmdargs);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_healpix_free(plot_args_t* plotargs, void* baton) {
	plothealpix_t* args = (plothealpix_t*)baton;
	free(args);
}

