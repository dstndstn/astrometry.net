#ifndef PLOTXY_H
#define PLOTXY_H

#include "plotstuff.h"

struct plotxy_args {
	char* fn;
	int ext;
	char* xcol;
	char* ycol;
	double xoff, yoff;
	int firstobj;
	int nobjs;
	double scale;
	double bglw;
	float bgr, bgg, bgb, bga;
};
typedef struct plotxy_args plotxy_t;

// Called prior to cairo surface initialization.
void* plot_xy_init(plot_args_t* args);

// Called post cairo surface initialization.
int plot_xy_init2(plot_args_t* args, void* baton);

int plot_xy_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_xy_plot(const char* command, cairo_t* cairo,
				 plot_args_t* plotargs, void* baton);

void plot_xy_free(plot_args_t* args, void* baton);

extern const plotter_t plotter_xy;

#endif
