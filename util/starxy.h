/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

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

#include "bl.h"
#include "an-bool.h"

struct starxy_t {
    double* x;
    double* y;
    double* flux;
    double* background;
    int N;

    double xlo, xhi, ylo, yhi;
};
typedef struct starxy_t starxy_t;

starxy_t* starxy_new(int N, bool flux, bool back);

void starxy_compute_range(starxy_t* xy);

double starxy_getx(const starxy_t* f, int i);
double starxy_gety(const starxy_t* f, int i);

void starxy_get(const starxy_t* f, int i, double* xy);

void starxy_setx(starxy_t* f, int i, double x);
void starxy_sety(starxy_t* f, int i, double y);

void starxy_set_x_array(starxy_t* s, const double* x);
void starxy_set_y_array(starxy_t* s, const double* y);
void starxy_set_flux_array(starxy_t* s, const double* f);

void starxy_sort_by_flux(starxy_t* f);

void starxy_set(starxy_t* f, int i, double x, double y);

int starxy_n(const starxy_t* f);

double* starxy_copy_x(const starxy_t* xy);
double* starxy_copy_y(const starxy_t* xy);
double* starxy_copy_xy(const starxy_t* xy);

double* starxy_to_flat_array(starxy_t* xy, double* arr);

void starxy_alloc_data(starxy_t* f, int N, bool flux, bool back);

void starxy_from_dl(starxy_t* xy, dl* l, bool flux, bool back);

// Just free the data, not the field itself.
void starxy_free_data(starxy_t* f);

void starxy_free(starxy_t* f);

#endif
