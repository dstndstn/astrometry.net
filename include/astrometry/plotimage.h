/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef PLOTIMAGE_H
#define PLOTIMAGE_H

#include "astrometry/plotstuff.h"
#include "astrometry/anwcs.h"

struct plotimage_args {
    char* fn;
    int format; // PLOTSTUFF_FORMAT_*

    // Use slow but correct resampling?
    // default is to use faster but approximate Cairo rendering.
    anbool resample;

    int downsample;

    // 
    double arcsinh;

    double rgbscale[3];

    double alpha;

    anwcs_t* wcs;
    double gridsize;

    // Only used when WCS is *not* set: scales the image by the given factors.
    //double scalex;
    //double scaley;

    // For FITS images: values that will be linearly transformed to 0,255.
    double image_low;
    double image_high;
    // image value that will be made transparent.
    double image_null;

    // image values outside this range will be made transparent
    // (if non-zero)
    double image_valid_low;
    double image_valid_high;

    int n_invalid_low;
    int n_invalid_high;
    int n_invalid_null;

    // FITS extension
    int fitsext;
    // FITS image plane
    int fitsplane;

    anbool auto_scale;

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

int plot_image_getsize(plotimage_t* args, int* W, int* H);

int plot_image_set_filename(plotimage_t* args, const char* fn);

int plot_image_setsize(plot_args_t* pargs, plotimage_t* args);

// 'percentile' must be in [0,1]
// results are placed in 'rgbout'
// don't rename 'rgbout' -- that name is used in plotstuff.i by SWIG
int plot_image_get_percentile(plot_args_t* pargs, plotimage_t* args,
                              double percentile,
                              unsigned char* rgbout);

void plot_image_add_to_pixels(plotimage_t* args, int rgb[3]);

unsigned char* plot_image_scale_float(plotimage_t* args, const float* fimg);

void plot_image_rgba_data(cairo_t* cairo, plotimage_t* args);

// After setting filename, actually open and read the image file.
int plot_image_read(const plot_args_t* pargs, plotimage_t* args);

void plot_image_make_color_transparent(plotimage_t* args, unsigned char r, unsigned char g, unsigned char b);

DECLARE_PLOTTER(image);

#endif
