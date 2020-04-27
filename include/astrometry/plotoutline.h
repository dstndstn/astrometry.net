/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef PLOTOUTLINE_H
#define PLOTOUTLINE_H

#include "astrometry/plotstuff.h"
#include "astrometry/anwcs.h"

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

int plot_outline_set_wcs(plotoutline_t* args, const sip_t* wcs);

int plot_outline_set_tan_wcs(plotoutline_t* args, const tan_t* wcs);

int plot_outline_set_fill(plotoutline_t* args, anbool fill);

//extern const plotter_t plotter_outline;
DECLARE_PLOTTER(outline);

#endif
