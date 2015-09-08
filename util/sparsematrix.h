/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

// Just what the world needs, another...
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <stdio.h>

#include "astrometry/bl.h"
#include "astrometry/an-bool.h"

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
