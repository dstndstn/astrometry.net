/*
  This file is part of the Astrometry.net suite.
  Copyright 2010, 2011 Dustin Lang.

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
#ifndef PLOTHEALPIX_H
#define PLOTHEALPIX_H

#include "plotstuff.h"

struct plothealpix_args {
	int nside;
	int stepsize;
};
typedef struct plothealpix_args plothealpix_t;

plothealpix_t* plot_healpix_get(plot_args_t* pargs);

void* plot_healpix_init(plot_args_t* args);

int plot_healpix_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_healpix_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_healpix_free(plot_args_t* args, void* baton);

DECLARE_PLOTTER(healpix);

#endif
