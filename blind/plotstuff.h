#ifndef PLOTSTUFF_H
#define PLOTSTUFF_H

#include <cairo.h>

struct plot_args {
	//cairo_t* cairo;

	float r, g, b, a;
	float lw;
	int marker;
	float markersize;
};
typedef struct plot_args plot_args_t;

typedef void* (*plot_func_init_t)(plot_args_t* args);
typedef int   (*plot_func_command_t)(const char* command, cairo_t* cr, plot_args_t* args, void* baton);
typedef void  (*plot_func_free_t)(plot_args_t* args, void* baton);

#endif
