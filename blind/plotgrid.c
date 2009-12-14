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
		// FIXME -- default
		ERROR("Need grid_rastep, grid_decstep");
		return -1;
	}
	for (ra = args->rastep * floor(ramin / args->rastep);
		 ra <= args->rastep * ceil(ramax / args->rastep);
		 ra += args->rastep) {
		plot_radec_line(pargs, ra, decmin, ra, decmax);
	}
	for (dec = args->decstep * floor(decmin / args->decstep);
		 dec <= args->decstep * ceil(decmax / args->decstep);
		 dec += args->decstep) {
		plot_radec_line(pargs, ramin, dec, ramax, dec);
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

