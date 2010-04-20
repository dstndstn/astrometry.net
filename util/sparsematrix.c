
#include "sparsematrix.h"

struct entry {
	int c;
	double val;
};
typedef struct entry entry_t;

int compare_entries(const void* v1, const void* v2) {
	const entry_t* e1 = v1;
	const entry_t* e2 = v2;
	if (e1->c < e2->c)
		return -1;
	if (e1->c == e2->c)
		return 0;
	return 1;
}

sparsematrix_t* sparsematrix_new(int R, int C) {
	int i;
	sparsematrix_t* sp = calloc(1, sizeof(sparsematrix_t));
	sp->R = R;
	sp->C = C;
	sp->rows = calloc(R, sizeof(bl));
	for (i=0; i<R; i++)
		bl_init(sp->rows + i, 16, sizeof(entry_t));
	return sp;
}

void sparsematrix_free(sparsematrix_t* sp) {
	int i;
	if (!sp) return;
	for (i=0; i<sp->R; i++)
		bl_remove_all(sp->rows + i);
	free(sp->rows);
	free(sp);
}

void sparsematrix_set(sparsematrix_t* sp, int r, int c, double val) {
	entry_t e;
	e.c = c;
	e.val = val;
	bl_insert_sorted(sp->rows + r, &e, compare_entries);
}

// make each row sum to 1.
void sparsematrix_normalize_rows(sparsematrix_t* sp) {
	int i;
	for (i=0; i<sp->R; i++) {
		int j, N;
		double sum;
		bl* row = sp->rows + i;
		entry_t* e;
		sum = 0;
		N = bl_size(row);
		for (j=0; j<N; j++) {
			e = bl_access(row, j);
			sum += e->val;
		}
		for (j=0; j<N; j++) {
			e = bl_access(row, j);
			e->val /= sum;
		}
	}
}

void sparsematrix_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, bool addto) {
	int i;
	for (i=0; i<sp->R; i++) {
		int j, N;
		double sum;
		bl* row = sp->rows + i;
		entry_t* e;
		sum = 0;
		N = bl_size(row);
		for (j=0; j<N; j++) {
			e = bl_access(row, j);
			sum += e->val * vec[e->c];
		}
		if (addto)
			out[i] += sum;
		else
			out[i] = sum;
	}
}

void sparsematrix_transpose_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, bool addto) {
	int i;
	if (!addto)
		for (i=0; i<sp->C; i++)
			out[i] = 0;

	for (i=0; i<sp->R; i++) {
		int j, N;
		bl* row = sp->rows + i;
		entry_t* e;
		N = bl_size(row);
		for (j=0; j<N; j++) {
			e = bl_access(row, j);
			out[e->c] += e->val * vec[i];
		}
	}
}

