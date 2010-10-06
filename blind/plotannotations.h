#ifndef PLOTANNOTATIONS_H
#define PLOTANNOTATIONS_H

#include "plotstuff.h"

extern const plotter_t plotter_annotations;

struct annotation_args {
	bool NGC;
	bool constellations;
	bool bright;
	bool HD;
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

#endif
