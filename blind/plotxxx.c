/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <string.h>
#include <math.h>

#include "plotxxx.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

DEFINE_PLOTTER(xxx);

plotxxx_t* plot_xxx_get(plot_args_t* pargs) {
    return plotstuff_get_config(pargs, "xxx");
}

void* plot_xxx_init(plot_args_t* plotargs) {
    plotxxx_t* args = calloc(1, sizeof(plotxxx_t));
    return args;
}

int plot_xxx_plot(const char* command,
                  cairo_t* cairo, plot_args_t* pargs, void* baton) {
    plotxxx_t* args = (plotxxx_t*)baton;
    return 0;
}

int plot_xxx_command(const char* cmd, const char* cmdargs,
                     plot_args_t* pargs, void* baton) {
    plotxxx_t* args = (plotxxx_t*)baton;
    if (streq(cmd, "xxx_file")) {
        //plot_image_set_filename(args, cmdargs);
    } else {
        ERROR("Did not understand command \"%s\"", cmd);
        return -1;
    }
    return 0;
}

void plot_xxx_free(plot_args_t* plotargs, void* baton) {
    plotxxx_t* args = (plotxxx_t*)baton;
    free(args);
}

