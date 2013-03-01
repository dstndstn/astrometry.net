/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

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

// Just what the world needs, another...
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <stdio.h>

#include "bl.h"
#include "an-bool.h"

// Stored in row-major form.
struct sparsematrix {
	int R;
	int C;
	bl* rows;
};
typedef struct sparsematrix sparsematrix_t;

sparsematrix_t* sparsematrix_new(int R, int C);

void sparsematrix_free(sparsematrix_t* sp);

// make each row sum to 1.
void sparsematrix_normalize_rows(sparsematrix_t* sp);

void sparsematrix_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, anbool addto);

void sparsematrix_transpose_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, anbool addto);

void sparsematrix_set(sparsematrix_t* sp, int r, int c, double val);

int sparsematrix_count_elements_in_row(const sparsematrix_t* sp, int row);

int sparsematrix_count_elements(const sparsematrix_t* sp);

double sparsematrix_max(const sparsematrix_t* sp);

double sparsematrix_argmax(const sparsematrix_t* sp, int* pr, int* pc);

void sparsematrix_subset_rows(sparsematrix_t* sp, const int* rows, int NR);

void sparsematrix_print_row(const sparsematrix_t* sp, int row, FILE* fid);

double sparsematrix_sum_row(const sparsematrix_t* sp, int r);

void sparsematrix_scale_row(const sparsematrix_t* sp, int r, double scale);

#endif
