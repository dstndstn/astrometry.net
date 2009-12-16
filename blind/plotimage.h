#ifndef PLOTIMAGE_H
#define PLOTIMAGE_H

#include "plotstuff.h"

struct plotimage_args {
	char* fn;
	int format; // PLOTSTUFF_FORMAT_*
	//char* format;
	// FIXME -- alpha?

	sip_t* wcs;
	double gridsize;

	unsigned char* img;
	int W;
	int H;
};
typedef struct plotimage_args plotimage_t;

void* plot_image_init(plot_args_t* args);

int plot_image_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_image_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_image_free(plot_args_t* args, void* baton);

int plot_image_set_filename(plotimage_t* args, const char* fn);

int plot_image_setsize(plot_args_t* pargs, plotimage_t* args);

void plot_image_rgba_data(cairo_t* cairo, unsigned char* img, int W, int H);

extern const plotter_t plotter_image;

#endif
