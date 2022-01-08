/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "sparsematrix.h"
#include "mathutil.h"

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

#define FOR_EACH(sp, X)                                 \
    do {                                                \
        int r;                                          \
        for (r=0; r<sp->R; r++) {                       \
            bl* row = sp->rows + r;                     \
            int ci;                                     \
            for (ci=0; ci<bl_size(row); ci++) {		\
                entry_t* e = bl_access(row, ci);	\
                X;                                      \
            }                                           \
        }                                               \
    } while (0)

#define FOR_EACH_IN_ROW(sp, r, X)                                       \
    do {                                                                \
        bl* row = sp->rows + r;                                         \
        int ci;                                                         \
        for (ci=0; ci<bl_size(row); ci++) {				\
                                           entry_t* e = bl_access(row, ci); \
                                           X;                           \
                                           }                            \
	} while (0)

/*
 #define FOR_EACH_DECL()							\
 int r;										\
 int ci;										\
 int c;										\
 bl* row;									\
 entry_t* e;
 #define FOR_EACH(sp, r, ci, c, row, e)									\
 for (row=sp->rows, ci=0, r=0, e=bl_access(row,ci); r<sp->R;			\
 r = (ci == bl_size(row) ? r+1 : r),							\
 ci = (ci == bl_size(row) ? 0 : ci+1	),					\
 row = sp->rows + r,										\
 e = (r == sp->R ? NULL : bl_access(row, ci)))
 //ci = (ci == bl_size(row) ? e=bl_access((ci == bl_size(row) ? sp->rows + (++r) : rows), (ci == bl_size(row) ? 0 : ci)), ci++)
 */

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
        if (N == 0)
            continue;
        for (j=0; j<N; j++) {
            e = bl_access(row, j);
            sum += e->val;
        }
        printf("row sum of %i: %g\n", i, sum);
        for (j=0; j<N; j++) {
            e = bl_access(row, j);
            e->val /= sum;
        }
    }
}

void sparsematrix_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, anbool addto) {
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

void sparsematrix_transpose_mult_vec(const sparsematrix_t* sp, const double* vec, double* out, anbool addto) {
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

int sparsematrix_count_elements_in_row(const sparsematrix_t* sp, int row) {
    return bl_size(sp->rows + row);
}

int sparsematrix_count_elements(const sparsematrix_t* sp) {
    int i, N;
    N = 0;
    for (i=0; i<sp->R; i++)
        N += bl_size(sp->rows + i);
    return N;
}

void sparsematrix_subset_rows(sparsematrix_t* sp, const int* rows, int NR) {
    bl* newrows;
    int i;
    assert(NR <= sp->R);
    newrows = malloc(NR * sizeof(bl));
    for (i=0; i<NR; i++)
        newrows[i] = sp->rows[rows[i]];
    free(sp->rows);
    sp->rows = newrows;
    sp->R = NR;
}

double sparsematrix_max(const sparsematrix_t* sp) {
    double mx = -LARGE_VAL;
    FOR_EACH(sp, mx = MAX(mx, e->val));
    /*
     FOR_EACH_DECL();
     FOR_EACH(sp, r, ci, c, row, e) {
     mx = MAX(mx, e->val);
     }
     */
    return mx;
}

double sparsematrix_argmax(const sparsematrix_t* sp, int* pr, int* pc) {
    double mx = -LARGE_VAL;
    FOR_EACH(sp, if (e->val > mx) { mx = e->val; *pr = r; *pc = e->c; });
    return mx;
}

double sparsematrix_sum_row(const sparsematrix_t* sp, int r) {
    double sum = 0.0;
    FOR_EACH_IN_ROW(sp, r, sum += e->val);
    return sum;
}

void sparsematrix_scale_row(const sparsematrix_t* sp, int r, double scale) {
    FOR_EACH_IN_ROW(sp, r, e->val *= scale);
}

void sparsematrix_print_row(const sparsematrix_t* sp, int r, FILE* fid) {
    const bl* row = sp->rows + r;
    int i;
    for (i=0; i<bl_size(row); i++) {
        entry_t* e = bl_access_const(row, i);
        if (i)
            fprintf(fid, ", ");
        fprintf(fid, "[%i]=%g", e->c, e->val);
    }
    fprintf(fid, "\n");
}
