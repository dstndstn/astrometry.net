#ifndef PLOTXY_H
#define PLOTXY_H

#include "plotstuff.h"

void* plot_xy_init(plot_args_t* args);

int plot_xy_command(const char* command, cairo_t* cr,
					plot_args_t* args, void* baton);

void plot_xy_free(plot_args_t* args, void* baton);

#endif
