/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef STARXY_H
#define STARXY_H

#include "astrometry/bl.h"
#include "astrometry/an-bool.h"

struct starxy_t {
    double* x;
    double* y;
    double* flux;
    double* background;
    int N;

    double xlo, xhi, ylo, yhi;
};
typedef struct starxy_t starxy_t;

starxy_t* starxy_new(int N, anbool flux, anbool back);

void starxy_compute_range(starxy_t* xy);

double starxy_getx(const starxy_t* f, int i);
double starxy_gety(const starxy_t* f, int i);

double starxy_get_x(const starxy_t* f, int i);
double starxy_get_y(const starxy_t* f, int i);
double starxy_get_flux(const starxy_t* f, int i);

void starxy_get(const starxy_t* f, int i, double* xy);

void starxy_setx(starxy_t* f, int i, double x);
void starxy_sety(starxy_t* f, int i, double y);

void starxy_set_x(starxy_t* f, int i, double x);
void starxy_set_y(starxy_t* f, int i, double y);
void starxy_set_flux(starxy_t* f, int i, double y);

// Copies just the first N entries into a new starxy_t object.
starxy_t* starxy_subset(starxy_t*, int N);

// Copies from the given arrays into the starxy_t.
void starxy_set_x_array(starxy_t* s, const double* x);
void starxy_set_y_array(starxy_t* s, const double* y);
void starxy_set_flux_array(starxy_t* s, const double* f);
void starxy_set_bg_array(starxy_t* s, const double* f);

// interleaved x,y
void starxy_set_xy_array(starxy_t* s, const double* xy);

void starxy_sort_by_flux(starxy_t* f);

void starxy_set(starxy_t* f, int i, double x, double y);

int starxy_n(const starxy_t* f);

double* starxy_copy_x(const starxy_t* xy);
double* starxy_copy_y(const starxy_t* xy);
double* starxy_copy_xy(const starxy_t* xy);

// Returns a flat array of [x0, y0, x1, y1, ...].
// If "arr" is NULL, allocates and returns a new array.
double* starxy_to_xy_array(starxy_t* xy, double* arr);

// Like starxy_to_xy_array, but also includes "flux" and "background" if they're set.
double* starxy_to_flat_array(starxy_t* xy, double* arr);

void starxy_alloc_data(starxy_t* f, int N, anbool flux, anbool back);

void starxy_from_dl(starxy_t* xy, dl* l, anbool flux, anbool back);

// Just free the data, not the field itself.
void starxy_free_data(starxy_t* f);

void starxy_free(starxy_t* f);

#endif
