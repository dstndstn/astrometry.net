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
};
typedef struct starxy_t starxy_t;

double starxy_getx(starxy_t* f, int i);
double starxy_gety(starxy_t* f, int i);

void starxy_setx(starxy_t* f, int i, double x);
void starxy_sety(starxy_t* f, int i, double y);

void starxy_set(starxy_t* f, int i, double x, double y);

int starxy_n(starxy_t* f);

starxy_t* starxy_alloc(int N, bool flux, bool back);

void starxy_alloc_data(starxy_t* f, int N, bool flux, bool back);

void starxy_from_dl(starxy_t* xy, dl* l, bool flux, bool back);

double* starxy_to_flat_array(starxy_t* xy, double* arr);

// Just free the data, not the field itself.
void starxy_free_data(starxy_t* f);

void starxy_free(starxy_t* f);

#endif
