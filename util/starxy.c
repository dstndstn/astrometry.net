/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <assert.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "starxy.h"
#include "permutedsort.h"

void starxy_set_xy_array(starxy_t* s, const double* xy) {
	int i,N;
    N = starxy_n(s);
    for (i=0; i<N; i++) {
		s->x[i] = xy[2*i+0];
		s->y[i] = xy[2*i+1];
    }
}

starxy_t* starxy_subset(starxy_t* full, int N) {
	starxy_t* sub = starxy_new(N, full->flux ? TRUE:FALSE, full->background?TRUE:FALSE);
	if (!sub)
		return sub;
	starxy_set_x_array(sub, full->x);
	starxy_set_y_array(sub, full->y);
	if (full->flux)
		starxy_set_flux_array(sub, full->flux);
	if (full->background)
		starxy_set_bg_array(sub, full->background);
	return sub;
}

void starxy_compute_range(starxy_t* xy) {
    int i, N;
    xy->xlo = xy->ylo =  HUGE_VAL;
    xy->xhi = xy->yhi = -HUGE_VAL;
    N = starxy_n(xy);
    for (i=0; i<N; i++) {
        xy->xlo = MIN(xy->xlo, starxy_getx(xy, i));
        xy->xhi = MAX(xy->xhi, starxy_getx(xy, i));
        xy->ylo = MIN(xy->ylo, starxy_gety(xy, i));
        xy->yhi = MAX(xy->yhi, starxy_gety(xy, i));
    }
}

double starxy_getx(const starxy_t* f, int i) {
  assert(f);
  assert(i < f->N);
  assert(i >= 0);
  assert(f->x);
  return f->x[i];
}

double starxy_gety(const starxy_t* f, int i) {
  assert(f);
  assert(i < f->N);
  assert(i >= 0);
  assert(f->y);
  return f->y[i];
}

double starxy_get_flux(const starxy_t* f, int i) {
  assert(f);
  assert(i >= 0);
  assert(i < f->N);
  assert(f->flux);
  return f->flux[i];
}

double starxy_get_x(const starxy_t* f, int i) {
  return starxy_getx(f, i);
}
double starxy_get_y(const starxy_t* f, int i) {
  return starxy_gety(f, i);
}

void starxy_get(const starxy_t* f, int i, double* xy) {
    xy[0] = starxy_getx(f, i);
    xy[1] = starxy_gety(f, i);
}

void starxy_setx(starxy_t* f, int i, double val) {
  assert(f);
  assert(i >= 0);
  assert(i < f->N);
  assert(f->x);
  f->x[i] = val;
}

void starxy_sety(starxy_t* f, int i, double val) {
  assert(f);
  assert(i >= 0);
  assert(i < f->N);
  assert(f->y);
  f->y[i] = val;
}

void starxy_set_flux(starxy_t* f, int i, double val) {
  assert(f);
  assert(i >= 0);
  assert(i < f->N);
  assert(f->flux);
  f->flux[i] = val;
}

void starxy_set_x(starxy_t* f, int i, double x) {
  starxy_setx(f, i, x);
}
void starxy_set_y(starxy_t* f, int i, double y) {
  starxy_sety(f, i, y);
}

void starxy_set(starxy_t* f, int i, double x, double y) {
    assert(i < f->N);
    f->x[i] = x;
    f->y[i] = y;
}

int starxy_n(const starxy_t* f) {
    return f->N;
}

void starxy_free_data(starxy_t* f) {
    if (!f) return;
    free(f->x);
    free(f->y);
    free(f->flux);
    free(f->background);
}

void starxy_free(starxy_t* f) {
    starxy_free_data(f);
    free(f);
}

double* starxy_copy_x(const starxy_t* xy) {
    double* res = malloc(sizeof(double) * starxy_n(xy));
    memcpy(res, xy->x, sizeof(double) * starxy_n(xy));
    return res;
}

double* starxy_copy_y(const starxy_t* xy) {
    double* res = malloc(sizeof(double) * starxy_n(xy));
    memcpy(res, xy->y, sizeof(double) * starxy_n(xy));
    return res;
}

double* starxy_copy_xy(const starxy_t* xy) {
    int i, N;
    double* res;
    N = starxy_n(xy);
    res = malloc(sizeof(double) * 2 * N);
    for (i=0; i<N; i++) {
        res[2*i + 0] = starxy_getx(xy, i);
        res[2*i + 1] = starxy_gety(xy, i);
    }
    return res;
}

void starxy_sort_by_flux(starxy_t* s) {
	int* perm;
	perm = permuted_sort(s->flux, sizeof(double), compare_doubles_desc,
						 NULL, s->N);
	permutation_apply(perm, s->N, s->x, s->x, sizeof(double));
	permutation_apply(perm, s->N, s->y, s->y, sizeof(double));
	if (s->flux)
		permutation_apply(perm, s->N, s->flux, s->flux, sizeof(double));
	if (s->background)
		permutation_apply(perm, s->N, s->background, s->background, sizeof(double));
	free(perm);
}

void starxy_set_x_array(starxy_t* s, const double* x) {
	memcpy(s->x, x, s->N * sizeof(double));
}
void starxy_set_y_array(starxy_t* s, const double* y) {
	memcpy(s->y, y, s->N * sizeof(double));
}
void starxy_set_flux_array(starxy_t* s, const double* f) {
	memcpy(s->flux, f, s->N * sizeof(double));
}

void starxy_set_bg_array(starxy_t* s, const double* f) {
	memcpy(s->background, f, s->N * sizeof(double));
}

starxy_t* starxy_new(int N, anbool flux, anbool back) {
    starxy_t* xy = calloc(1, sizeof(starxy_t));
    starxy_alloc_data(xy, N, flux, back);
    return xy;
}

void starxy_alloc_data(starxy_t* f, int N, anbool flux, anbool back) {
    f->x = malloc(N * sizeof(double));
    f->y = malloc(N * sizeof(double));
    if (flux)
        f->flux = malloc(N * sizeof(double));
    else
        f->flux = NULL;
    if (back)
        f->background = malloc(N * sizeof(double));
    else
        f->background = NULL;
    f->N = N;
}

double* starxy_to_flat_array(starxy_t* xy, double* arr) {
    int nr = 2;
    int i, ind;
    if (xy->flux)
        nr++;
    if (xy->background)
        nr++;

    if (!arr)
        arr = malloc(nr * starxy_n(xy) * sizeof(double));

    ind = 0;
    for (i=0; i<xy->N; i++) {
        arr[ind] = xy->x[i];
        ind++;
        arr[ind] = xy->y[i];
        ind++;
        if (xy->flux) {
            arr[ind] = xy->flux[i];
            ind++;
        }
        if (xy->background) {
            arr[ind] = xy->background[i];
            ind++;
        }
    }
    return arr;
}

double* starxy_to_xy_array(starxy_t* xy, double* arr) {
    int i;
    if (!arr)
        arr = malloc(2 * starxy_n(xy) * sizeof(double));
    for (i=0; i<starxy_n(xy); i++) {
        arr[2*i]   = xy->x[i];
        arr[2*i+1] = xy->y[i];
    }
    return arr;
}

void starxy_from_dl(starxy_t* xy, dl* l, anbool flux, anbool back) {
    int i;
    int nr = 2;
    int ind;
    if (flux)
        nr++;
    if (back)
        nr++;

    starxy_alloc_data(xy, dl_size(l)/nr, flux, back);
    ind = 0;
    for (i=0; i<xy->N; i++) {
        xy->x[i] = dl_get(l, ind);
        ind++;
        xy->y[i] = dl_get(l, ind);
        ind++;
        if (flux) {
            xy->flux[i] = dl_get(l, ind);
            ind++;
        }
        if (back) {
            xy->background[i] = dl_get(l, ind);
            ind++;
        }
    }
}

