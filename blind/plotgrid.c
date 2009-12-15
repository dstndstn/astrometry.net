/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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

#include "plotgrid.h"
#include "sip-utils.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

const plotter_t plotter_grid = {
	.name = "grid",
	.init = plot_grid_init,
	.command = plot_grid_command,
	.doplot = plot_grid_plot,
	.free = plot_grid_free
};

void* plot_grid_init(plot_args_t* plotargs) {
	plotgrid_t* args = calloc(1, sizeof(plotgrid_t));
	args->dolabel = TRUE;
	return args;
}

int plot_grid_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotgrid_t* args = (plotgrid_t*)baton;
	double ramin,ramax,decmin,decmax;
	double ra,dec;

	if (!pargs->wcs) {
		ERROR("No WCS was set -- can't plot grid lines");
		return -1;
	}
	// Find image bounds in RA,Dec...
	sip_get_radec_bounds(pargs->wcs, 50, &ramin, &ramax, &decmin, &decmax);
	if (args->rastep == 0 || args->decstep == 0) {
		// FIXME -- choose defaults
		ERROR("Need grid_rastep, grid_decstep");
		return -1;
	}
	logverb("Image bounds: RA %g, %g, Dec %g, %g\n",
			ramin, ramax, decmin, decmax);
	for (ra = args->rastep * floor(ramin / args->rastep);
		 ra <= args->rastep * ceil(ramax / args->rastep);
		 ra += args->rastep) {
		plot_line_constant_ra(pargs, ra, decmin, decmax);
		cairo_stroke(pargs->cairo);
	}
	for (dec = args->decstep * floor(decmin / args->decstep);
		 dec <= args->decstep * ceil(decmax / args->decstep);
		 dec += args->decstep) {
		plot_line_constant_dec(pargs, dec, ramin, ramax);
		cairo_stroke(pargs->cairo);
	}

	logmsg("Dolabel: %i\n", (int)args->dolabel);
	if (args->dolabel) {
		double cra, cdec;
		if (args->ralabelstep == 0 || args->declabelstep == 0) {
			// FIXME -- choose defaults
			ERROR("Need grid_ralabelstep, grid_declabelstep");
			return -1;
		}
		sip_get_radec_center(pargs->wcs, &cra, &cdec);
		assert(cra >= ramin && cra <= ramax);
		assert(cdec >= decmin && cdec <= decmax);
		for (ra = args->ralabelstep * floor(ramin / args->ralabelstep);
			 ra <= args->ralabelstep * ceil(ramax / args->ralabelstep);
			 ra += args->ralabelstep) {
			// where does this line leave the image?
			// cdec is inside
			// decmin is probably outside; 1.5 * decmin - 0.5 * cdec is definitely outside?
			//double out = MAX(-90, 1.5 * decmin - 0.5 * cdec);
			double out = MIN(90, 1.5 * decmax - 0.5 * cdec);
			double in = cdec;
			char label[32];
			double x,y;
			bool ok;
			int i, N;
			assert(!sip_is_inside_image(pargs->wcs, ra, out));
			//assert(!sip_is_inside_image(pargs->wcs, ra, in));
			i=0;
			N = 10;
			while (!sip_is_inside_image(pargs->wcs, ra, in)) {
				if (i == N)
					break;
				in = decmin + (double)i/(double)(N-1) * (decmax-decmin);
			}
			if (!sip_is_inside_image(pargs->wcs, ra, in))
				continue;
			while (fabs(out - in) > 1e-6) {
				// hahaha
				double half;
				bool isin;
				half = (out + in) / 2.0;
				isin = sip_is_inside_image(pargs->wcs, ra, half);
				if (isin)
					in = half;
				else
					out = half;
			}
			printf("in=%g, out=%g\n", in, out);
			snprintf(label, sizeof(label), "%.1f", ra);
			logmsg("Label \"%s\" at (%g,%g)\n", label, ra, in);
			ok = sip_radec2pixelxy(pargs->wcs, ra, in, &x, &y);
			cairo_move_to(pargs->cairo, x, y);
			cairo_show_text(pargs->cairo, label);
			cairo_stroke(pargs->cairo);
		}
		
	}


	return 0;
}

int plot_grid_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotgrid_t* args = (plotgrid_t*)baton;
	if (streq(cmd, "grid_rastep")) {
		args->rastep = atof(cmdargs);
	} else if (streq(cmd, "grid_decstep")) {
		args->decstep = atof(cmdargs);
	} else if (streq(cmd, "grid_ralabelstep")) {
		args->ralabelstep = atof(cmdargs);
	} else if (streq(cmd, "grid_declabelstep")) {
		args->declabelstep = atof(cmdargs);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_grid_free(plot_args_t* plotargs, void* baton) {
	plotgrid_t* args = (plotgrid_t*)baton;
	free(args);
}

