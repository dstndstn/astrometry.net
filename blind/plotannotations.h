#ifndef PLOTANNOTATIONS_H
#define PLOTANNOTATIONS_H

#include "plotstuff.h"

struct annotation_args {
	anbool NGC;

	anbool constellations;
	anbool constellation_lines;
	anbool constellation_markers;
	anbool constellation_labels;
	anbool constellation_labels_long;

	anbool bright;
	anbool HD;
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


DECLARE_PLOTTER(annotations);


#endif
