/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <assert.h>
#include <stdlib.h>

#include "eigen-math.h"

evector_t* evector_new(int N) {
    evector_t* v = malloc(sizeof(evector_t));
    assert(v);
    v->data = calloc(N, sizeof(double));
    assert(v->data);
    v->N = N;
    return v;
}

ematrix_t* ematrix_new(int R, int C) {
    ematrix_t* m = malloc(sizeof(ematrix_t));
    assert(m);
    m->data = calloc(R * C, sizeof(double));
    assert(m->data);
    m->rows = R;
    m->cols = C;
    return m;
}

void ematrix_set(ematrix_t* m, int r, int c, double v) {
    m->data[r * m->cols + c] = v;
}
double ematrix_get(const ematrix_t* m, int r, int c) {
    return m->data[r * m->cols + c];
}

void evector_set(evector_t* m, int i, double v) {
    m->data[i] = v;
}
double evector_get(const evector_t* m, int i) {
    return m->data[i];
}

