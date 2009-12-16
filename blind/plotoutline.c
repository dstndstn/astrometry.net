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

void* plot_outline_init(plot_args_t* plotargs) {
	plotoutline_t* args = calloc(1, sizeof(plotoutline_t));
	args->stepsize = 10;
	return args;
}

struct walk_token {
	cairo_t* cairo;
	bool first;
	sip_t* wcs;
};

static void walk_callback(const sip_t* wcs, double x, double y, double ra, double dec, void* token) {
	struct walk_token* walk = token;
	bool ok;
	ok = sip_radec2pixelxy(walk->wcs, ra, dec, &x, &y);
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

	token.first = TRUE;
	token.cairo = cairo;
	token.wcs = pargs->wcs;
	sip_walk_image_boundary(args->wcs, args->stepsize, walk_callback, &token);
	cairo_stroke(cairo);

	return 0;
}

int plot_outline_set_wcs_file(plotoutline_t* args, const char* cmdargs) {
	free(args->wcs);
	args->wcs = sip_read_tan_or_sip_header_file_ext(cmdargs, 0, NULL, FALSE);
	if (!args->wcs) {
		ERROR("Failed to read WCS file \"%s\"", cmdargs);
		return -1;
	}
	logverb("Read WCS file %s\n", cmdargs);
	return 0;
}

int plot_outline_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	if (streq(cmd, "outline_wcs")) {
		if (plot_outline_set_wcs_file(args, cmdargs)) {
			return -1;
		}
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

