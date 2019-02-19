/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef PLOTFILL_H
#define PLOTFILL_H

#include "astrometry/plotstuff.h"

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
