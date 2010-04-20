// Just what the world needs, another...
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

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

void sparsematrix_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, bool addto);

void sparsematrix_transpose_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, bool addto);

void sparsematrix_set(sparsematrix_t* sp, int r, int c, double val);




#endif
