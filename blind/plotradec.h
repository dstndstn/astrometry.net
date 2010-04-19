#ifndef PLOTRADEC_H
#define PLOTRADEC_H

#include "plotstuff.h"

struct plotradec_args {
	char* fn;
	int ext;
	char* racol;
	char* deccol;
	int firstobj;
	int nobjs;

	// coordinates added with radec_val <x> <y>
	dl* radecvals;
};
typedef struct plotradec_args plotradec_t;

// Called prior to cairo surface initialization.
void* plot_radec_init(plot_args_t* args);

// Called post cairo surface initialization.
int plot_radec_init2(plot_args_t* args, void* baton);

void plot_radec_set_racol(plotradec_t* args, const char* col);
void plot_radec_set_deccol(plotradec_t* args, const char* col);
void plot_radec_set_filename(plotradec_t* args, const char* fn);

int plot_radec_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_radec_plot(const char* command, cairo_t* cairo,
				 plot_args_t* plotargs, void* baton);

void plot_radec_free(plot_args_t* args, void* baton);

void plot_radec_vals(plotradec_t* args, double ra, double dec);

extern const plotter_t plotter_radec;

#endif
