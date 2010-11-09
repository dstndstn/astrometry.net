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
#ifndef PLOTFILL_H
#define PLOTFILL_H

#include "plotstuff.h"

struct plotfill_args {
};
typedef struct plotfill_args plotfill_t;

void* plot_fill_init(plot_args_t* args);

int plot_fill_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_fill_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_fill_free(plot_args_t* args, void* baton);

//extern const plotter_t plotter_fill;
//plotter_t* plot_fill_new();
DECLARE_PLOTTER(fill);

#endif
