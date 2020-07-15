/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/
#ifndef PLOTCOADD_H
#define PLOTCOADD_H

#include "astrometry/plotstuff.h"

struct plotcoadd_args {
};
typedef struct plotcoadd_args plotcoadd_t;

plotcoadd_t* plot_coadd_get(plot_args_t* pargs);

void* plot_coadd_init(plot_args_t* args);

int plot_coadd_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_coadd_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_coadd_free(plot_args_t* args, void* baton);

#endif
