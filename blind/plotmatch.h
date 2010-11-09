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
#ifndef PLOTMATCH_H
#define PLOTMATCH_H

#include "plotstuff.h"
#include "bl.h"
#include "matchobj.h"

struct plotmatch_args {
	// of MatchObj's
	bl* matches;
};
typedef struct plotmatch_args plotmatch_t;

plotmatch_t* plot_match_get(plot_args_t* pargs);

int plot_match_add_match(plotmatch_t* args, const MatchObj* mo);
int plot_match_set_filename(plotmatch_t* args, const char* filename);

void* plot_match_init(plot_args_t* args);

int plot_match_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_match_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_match_free(plot_args_t* args, void* baton);

//extern const plotter_t plotter_match;
DECLARE_PLOTTER(match);

#endif
