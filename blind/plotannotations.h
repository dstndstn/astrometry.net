#ifndef PLOTANNOTATIONS_H
#define PLOTANNOTATIONS_H

#include "plotstuff.h"

extern const plotter_t plotter_annotations;

void* plot_annotations_init(plot_args_t* args);

int plot_annotations_command(const char* command, const char* cmdargs,
							 plot_args_t* args, void* baton);

int plot_annotations_plot(const char* command, 
						  cairo_t* cr, plot_args_t* args, void* baton);

void plot_annotations_free(plot_args_t* args, void* baton);

#endif
