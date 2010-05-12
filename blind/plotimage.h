#ifndef PLOTIMAGE_H
#define PLOTIMAGE_H

#include "plotstuff.h"
#include "anwcs.h"

struct plotimage_args {
	char* fn;
	int format; // PLOTSTUFF_FORMAT_*

	double alpha;

	anwcs_t* wcs;
	double gridsize;

	// For FITS images: values that will be linearly transformed to 0,255.
	double image_low;
	double image_high;
	// image value that will be made transparent.
	double image_null;
	// FITS extension
	int fitsext;
	// FITS image plane
	int fitsplane;

	unsigned char* img;
	int W;
	int H;
};
typedef struct plotimage_args plotimage_t;

plotimage_t* plot_image_get(plot_args_t* pargs);

int plot_image_set_wcs(plotimage_t* args, const char* filename, int ext);

void* plot_image_init(plot_args_t* args);

int plot_image_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_image_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_image_free(plot_args_t* args, void* baton);

int plot_image_set_filename(plotimage_t* args, const char* fn);

int plot_image_setsize(plot_args_t* pargs, plotimage_t* args);

void plot_image_rgba_data(cairo_t* cairo, unsigned char* img, int W, int H,
						  double alpha);

extern const plotter_t plotter_image;

#endif
