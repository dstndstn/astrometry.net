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

#include "plotfill.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

DEFINE_PLOTTER(fill);

void* plot_fill_init(plot_args_t* plotargs) {
	plotfill_t* args = calloc(1, sizeof(plotfill_t));
	return args;
}

int plot_fill_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotstuff_builtin_apply(cairo, pargs);
	cairo_paint(cairo);
	return 0;
}

int plot_fill_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	ERROR("Did not understand command \"%s\"", cmd);
	return -1;
}

void plot_fill_free(plot_args_t* plotargs, void* baton) {
	plotfill_t* args = (plotfill_t*)baton;
	free(args);
}

