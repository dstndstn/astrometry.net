#ifndef PLOTIMAGE_H
#define PLOTIMAGE_H

#include "plotstuff.h"

void* plot_image_init(plot_args_t* args);

int plot_image_command(const char* command, const char* cmdargs, cairo_t* cr,
					plot_args_t* args, void* baton);

int plot_image_plot(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_image_free(plot_args_t* args, void* baton);

void plot_image_rgba_data(cairo_t* cairo, unsigned char* img, int W, int H);

#endif
