#ifndef MERCRENDER_H
#define MERCRENDER_H

#include "tilerender.h"
#include "merctree.h"

float* mercrender_file(char* fn, render_args_t* args, int symbol);

void mercrender(merctree* m, render_args_t* args, float* fluximg, int symbol);

#define RENDERSYMBOL_psf 0
#define RENDERSYMBOL_x   1
#define RENDERSYMBOL_o   2
#define RENDERSYMBOL_dot 3

int add_star(double xp, double yp, double rflux, double bflux, double nflux,
		float* fluximg, int rendersymbol, render_args_t* args);

#endif
