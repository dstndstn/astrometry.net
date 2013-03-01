/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2010 Dustin Lang.

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
#ifndef PLOTOUTLINE_H
#define PLOTOUTLINE_H

#include "plotstuff.h"
#include "anwcs.h"

struct plotoutline_args {
	anwcs_t* wcs;
	double stepsize;
	anbool fill;
};
typedef struct plotoutline_args plotoutline_t;

plotoutline_t* plot_outline_get(plot_args_t* pargs);

void* plot_outline_init(plot_args_t* args);

int plot_outline_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_outline_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_outline_free(plot_args_t* args, void* baton);

int plot_outline_set_wcs_file(plotoutline_t* args, const char* filename, int ext);

int plot_outline_set_wcs_size(plotoutline_t* args, int W, int H);

int plot_outline_set_wcs(plotoutline_t* args, sip_t* wcs);

int plot_outline_set_fill(plotoutline_t* args, anbool fill);

//extern const plotter_t plotter_outline;
DECLARE_PLOTTER(outline);

#endif
