#ifndef RENDER_QUADS_H
#define RENDER_QUADS_H

#include "tilerender.h"

int render_quads(cairo_t* cairo, render_args_t* args);

void quad_radec_to_xy(render_args_t* args, const double* radecs,
					  double* xys, int dimquad);

#endif
