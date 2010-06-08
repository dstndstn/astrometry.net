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

#include "plotoutline.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "sip-utils.h"
#include "sip_qfits.h"

const plotter_t plotter_outline = {
	.name = "outline",
	.init = plot_outline_init,
	.command = plot_outline_command,
	.doplot = plot_outline_plot,
	.free = plot_outline_free
};

plotoutline_t* plot_outline_get(plot_args_t* pargs) {
	return plotstuff_get_config(pargs, "outline");
}

void* plot_outline_init(plot_args_t* plotargs) {
	plotoutline_t* args = calloc(1, sizeof(plotoutline_t));
	args->stepsize = 10;
	return args;
}

struct walk_token {
	cairo_t* cairo;
	bool first;
	plot_args_t* pargs;
};

static void walk_callback(const anwcs_t* wcs, double ix, double iy, double ra, double dec, void* token) {
	struct walk_token* walk = token;
	bool ok;
	double x, y;
	ok = plotstuff_radec2xy(walk->pargs, ra, dec, &x, &y);
	logverb("plotoutline: wcs x,y (%.0f,%.0f) -> RA,Dec (%.1g,%.1g) -> image x,y (%.1f, %.1f)\n",
			ix, iy, ra, dec, x, y);
	if (!ok)
		return;
	if (walk->first) {
		cairo_move_to(walk->cairo, x, y);
		walk->first = FALSE;
	} else
		cairo_line_to(walk->cairo, x, y);
}

int plot_outline_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	struct walk_token token;
	assert(args->stepsize > 0);
	assert(args->wcs);
	assert(pargs->wcs);

	plotstuff_builtin_apply(cairo, pargs);

	token.first = TRUE;
	token.cairo = cairo;
	token.pargs = pargs;
	anwcs_walk_image_boundary(args->wcs, args->stepsize, walk_callback, &token);
	cairo_close_path(cairo);
	if (args->fill)
		cairo_fill(cairo);
	else
		cairo_stroke(cairo);

	return 0;
}

int plot_outline_set_wcs_size(plotoutline_t* args, int W, int H) {
  if (!args->wcs) {
    ERROR("No WCS is currently set.");
    return -1;
  }
  anwcs_set_size(args->wcs, W, H);
  return 0;
}

int plot_outline_set_wcs_file(plotoutline_t* args, const char* filename, int ext) {
	anwcs_t* wcs = anwcs_open(filename, ext);
	if (!wcs) {
		ERROR("Failed to read WCS file \"%s\"", filename);
		return -1;
	}
	logverb("Read WCS file %s\n", filename);
	if (args->wcs)
		anwcs_free(args->wcs);
	args->wcs = wcs;
	//anwcs_print(args->wcs, stdout);
	return 0;
}

int plot_outline_set_wcs(plotoutline_t* args, sip_t* wcs) {
	if (args->wcs)
		anwcs_free(args->wcs);
	args->wcs = anwcs_new_sip(wcs);
	return 0;
}

int plot_outline_set_fill(plotoutline_t* args, bool fill) {
	args->fill = fill;
	return 0;
}

int plot_outline_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	if (streq(cmd, "outline_wcs")) {
		if (plot_outline_set_wcs_file(args, cmdargs, 0)) {
			return -1;
		}
	} else if (streq(cmd, "outline_fill")) {
		if (streq(cmdargs, "0"))
			args->fill = FALSE;
		else
			args->fill = TRUE;
	} else if (streq(cmd, "outline_step")) {
		args->stepsize = atof(cmdargs);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_outline_free(plot_args_t* plotargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	free(args);
}

