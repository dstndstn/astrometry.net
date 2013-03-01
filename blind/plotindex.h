/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

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
#ifndef PLOTINDEX_H
#define PLOTINDEX_H

#include "plotstuff.h"
#include "bl.h"
#include "index.h"

struct plotindex_args {
	pl* indexes;
	pl* qidxes;
	anbool stars;
	anbool quads;
	anbool fill;
};
typedef struct plotindex_args plotindex_t;

// immediate: do cairo_move_to and cairo_line_to; no cairo_stroke/fill.
void plot_quad_xy(cairo_t* cairo, double* quadxy, int dimquads);
// immediate: plot the given quad number.  Requires the plot wcs to be set.
void plot_index_plotquad(cairo_t* cairo, plot_args_t* pargs, plotindex_t* args, index_t* index, int quadnum, int DQ);


plotindex_t* plot_index_get(plot_args_t* pargs);

int plot_index_add_file(plotindex_t* args, const char* fn);

int plot_index_add_qidx_file(plotindex_t* args, const char* fn);

void* plot_index_init(plot_args_t* args);

int plot_index_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_index_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_index_free(plot_args_t* args, void* baton);

//extern const plotter_t plotter_index;
DECLARE_PLOTTER(index);

#endif
