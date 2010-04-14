#ifndef PLOTSTUFF_H
#define PLOTSTUFF_H

#include <stdio.h>
#include <cairo.h>

#include "keywords.h"
#include "sip.h"
#include "bl.h"
#include "anwcs.h"

#define PLOTSTUFF_FORMAT_JPG 1
#define PLOTSTUFF_FORMAT_PNG 2
#define PLOTSTUFF_FORMAT_PPM 3
#define PLOTSTUFF_FORMAT_PDF 4
// Save the image as RGBA image "pargs->outimage"
#define PLOTSTUFF_FORMAT_MEMIMG 5
#define PLOTSTUFF_FORMAT_FITS 6

struct plot_args {
    char* outfn;
	FILE* fout;
	int outformat;

	unsigned char* outimage;

	cairo_t* cairo;
	cairo_surface_t* target;

	//bl* plotters;

	cairo_operator_t op;

	//sip_t* wcs;
	anwcs_t* wcs;

	int W, H;
	float rgba[4];
	float lw;
	int marker;
	float markersize;

	float fontsize;

	// step size in pixels for drawing curved lines in RA,Dec
	float linestep;
};
typedef struct plot_args plot_args_t;

typedef void* (*plot_func_init_t)(plot_args_t* args);
typedef int   (*plot_func_init2_t)(plot_args_t* args, void* baton);
typedef int   (*plot_func_command_t)(const char* command, const char* cmdargs, plot_args_t* args, void* baton);
typedef int   (*plot_func_plot_t)(const char* command, cairo_t* cr, plot_args_t* args, void* baton);
typedef void  (*plot_func_free_t)(plot_args_t* args, void* baton);

struct plotter {
	// don't change the order of these fields!
	char* name;
	plot_func_init_t init;
	plot_func_init2_t init2;
	plot_func_command_t command;
	plot_func_plot_t doplot;
	plot_func_free_t free;
	void* baton;
};
typedef struct plotter plotter_t;

// return PLOTSTUFF_FORMAT_*, or -1 on error
int parse_image_format(const char* fmt);

int parse_color(const char* color, float* r, float* g, float* b, float* a);
int parse_color_rgba(const char* color, float* rgba);
int cairo_set_color(cairo_t* cairo, const char* color);
void cairo_set_rgba(cairo_t* cairo, const float* rgba);

int plotstuff_init(plot_args_t* plotargs);
int plotstuff_read_and_run_command(plot_args_t* pargs, FILE* f);
int plotstuff_run_command(plot_args_t* pargs, const char* cmd);

void* plotstuff_get_config(plot_args_t* pargs, const char* name);

int plotstuff_set_color(plot_args_t* pargs, const char* name);

int plotstuff_set_marker(plot_args_t* pargs, const char* name);

int plotstuff_set_markersize(plot_args_t* pargs, double ms);

int plotstuff_set_size(plot_args_t* pargs, int W, int H);

int
ATTRIB_FORMAT(printf,2,3)
plotstuff_run_commandf(plot_args_t* pargs, const char* fmt, ...);

int plotstuff_output(plot_args_t* pargs);
void plotstuff_free(plot_args_t* pargs);

/// WCS-related stuff:

// in arcsec/pixel
double plotstuff_pixel_scale(plot_args_t* pargs);

// RA,Dec in degrees
// x,y in pixels (cairo coordinates)
// Returns TRUE on success.
bool plotstuff_radec2xy(plot_args_t* pargs, double ra, double dec,
						double* x, double* y);

int plotstuff_get_radec_center_and_radius(plot_args_t* pargs, double* ra, double* dec, double* radius);

int plot_line_constant_ra(plot_args_t* pargs, double ra, double dec1, double dec2);
int plot_line_constant_dec(plot_args_t* pargs, double dec, double ra1, double ra2);





int plotstuff_append_doubles(const char* str, dl* lst);

#endif
