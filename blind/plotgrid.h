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
#ifndef PLOTGRID_H
#define PLOTGRID_H

#include "plotstuff.h"

struct plotgrid_args {
	bool dolabel;

	double rastep;
	double decstep;

	double ralabelstep;
	double declabelstep;
};
typedef struct plotgrid_args plotgrid_t;

plotgrid_t* plot_grid_get(plot_args_t* pargs);

void* plot_grid_init(plot_args_t* args);

int plot_grid_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_grid_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_grid_free(plot_args_t* args, void* baton);

extern const plotter_t plotter_grid;

#endif
