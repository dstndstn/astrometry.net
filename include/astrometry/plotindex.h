/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef PLOTINDEX_H
#define PLOTINDEX_H

#include "astrometry/plotstuff.h"
#include "astrometry/bl.h"
#include "astrometry/index.h"

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
