/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/
#ifndef PLOTXXX_H
#define PLOTXXX_H

#include "astrometry/plotstuff.h"

struct plotxxx_args {
};
typedef struct plotxxx_args plotxxx_t;

plotxxx_t* plot_xxx_get(plot_args_t* pargs);

void* plot_xxx_init(plot_args_t* args);

int plot_xxx_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_xxx_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_xxx_free(plot_args_t* args, void* baton);

extern const plotter_t plotter_xxx;

#endif
