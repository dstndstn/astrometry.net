/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2011 Dustin Lang.

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
	anbool dolabel;

	double rastep;
	double decstep;

	double ralabelstep;
	double declabelstep;

	int ralabeldir;
	int declabeldir;

	// Range of values to plot; default (0) is to choose sensible limits
	double ralo;
	double rahi;
	double declo;
	double dechi;

	// these strings are owned by the plotgrid object (will be free()d)
	char* raformat;
	char* decformat;
};
typedef struct plotgrid_args plotgrid_t;

plotgrid_t* plot_grid_get(plot_args_t* pargs);

void* plot_grid_init(plot_args_t* args);

int plot_grid_set_formats(plotgrid_t* grid, const char* raformat, const char* decformat);

int plot_grid_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_grid_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_grid_free(plot_args_t* args, void* baton);

#define DIRECTION_DEFAULT 0
#define DIRECTION_POS     1
#define DIRECTION_NEG     2
#define DIRECTION_POSNEG  3
#define DIRECTION_NEGPOS  4

void plot_grid_add_label(plot_args_t* pargs, double ra, double dec,
						 double lval, const char* format);

int plot_grid_find_ra_label_location(plot_args_t* pargs, double ra, double cdec, double decmin, double decmax, int dirn, double* pdec);
int plot_grid_find_dec_label_location(plot_args_t* pargs, double dec, double cra, double ramin, double ramax, int dirn, double* pra);

DECLARE_PLOTTER(grid);

#endif
