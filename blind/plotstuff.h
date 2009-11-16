#ifndef PLOTSTUFF_H
#define PLOTSTUFF_H

#include <stdio.h>
#include <cairo.h>

#include "keywords.h"

#define PLOTSTUFF_FORMAT_JPG 0
#define PLOTSTUFF_FORMAT_PNG 1
#define PLOTSTUFF_FORMAT_PPM 2
#define PLOTSTUFF_FORMAT_PDF 3

struct plot_args {
    char* outfn;
	FILE* fout;
	int outformat;
	cairo_t* cairo;
	cairo_surface_t* target;

	int W, H;
	float r, g, b, a;
	float lw;
	int marker;
	float markersize;
};
typedef struct plot_args plot_args_t;

typedef void* (*plot_func_init_t)(plot_args_t* args);
typedef int   (*plot_func_command_t)(const char* command, cairo_t* cr, plot_args_t* args, void* baton);
typedef void  (*plot_func_free_t)(plot_args_t* args, void* baton);

int parse_color(const char* color, float* r, float* g, float* b, float* a);

int plotstuff_init(plot_args_t* plotargs);
int plotstuff_read_and_run_command(plot_args_t* pargs, FILE* f);
int plotstuff_run_command(plot_args_t* pargs, const char* cmd);

int
ATTRIB_FORMAT(printf,2,3)
plotstuff_run_commandf(plot_args_t* pargs, const char* fmt, ...);

int plotstuff_output(plot_args_t* pargs);
void plotstuff_free(plot_args_t* pargs);

#endif
