/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <string.h>
#include <math.h>

#include "plotfill.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

DEFINE_PLOTTER(fill);

void* plot_fill_init(plot_args_t* plotargs) {
    plotfill_t* args = calloc(1, sizeof(plotfill_t));
    return args;
}

int plot_fill_plot(const char* command,
                   cairo_t* cairo, plot_args_t* pargs, void* baton) {
    plotstuff_builtin_apply(cairo, pargs);
    cairo_paint(cairo);
    return 0;
}

int plot_fill_command(const char* cmd, const char* cmdargs,
                      plot_args_t* pargs, void* baton) {
    ERROR("Did not understand command \"%s\"", cmd);
    return -1;
}

void plot_fill_free(plot_args_t* plotargs, void* baton) {
    plotfill_t* args = (plotfill_t*)baton;
    free(args);
}

