/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef PLOTMATCH_H
#define PLOTMATCH_H

#include "astrometry/plotstuff.h"
#include "astrometry/bl.h"
#include "astrometry/matchobj.h"

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
