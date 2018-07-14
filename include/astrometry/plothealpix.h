/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
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
