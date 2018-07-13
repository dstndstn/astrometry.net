/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef PLOTANNOTATIONS_H
#define PLOTANNOTATIONS_H

#include "astrometry/plotstuff.h"

struct annotation_args {
    anbool NGC;

    anbool constellations;
    anbool constellation_lines;
    anbool constellation_markers;
    anbool constellation_labels;
    anbool constellation_labels_long;
    // don't *exactly* connect the stars, leave a gap of this many pixels.
    // probably want this = plotstuff marker size
    float  constellation_lines_offset;

    anbool constellation_pastel;

    anbool bright;
    anbool bright_labels;

    anbool bright_pastel;

    anbool HD;
    anbool HD_labels;
    float ngc_fraction;
    bl* targets;
    char* hd_catalog;
};
typedef struct annotation_args plotann_t;

void* plot_annotations_init(plot_args_t* args);

plotann_t* plot_annotations_get(plot_args_t* pargs);

int plot_annotations_command(const char* command, const char* cmdargs,
                             plot_args_t* args, void* baton);

int plot_annotations_plot(const char* command, 
                          cairo_t* cr, plot_args_t* args, void* baton);

void plot_annotations_free(plot_args_t* args, void* baton);

int plot_annotations_set_hd_catalog(plotann_t* ann, const char* hdfn);

int plot_annotations_add_named_target(plotann_t* ann, const char* target);

void plot_annotations_add_target(plotann_t* ann, double ra, double dec,
                                 const char* name);

void plot_annotations_clear_targets(plotann_t* ann);

DECLARE_PLOTTER(annotations);


#endif
