/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

static kdtree_t* build_tree(CuTest* tc, double* data, int N, int D,
                            int Nleaf, int treetype, int treeopts);
static double* random_points_d(int N, int D);

static double* random_points_d(int N, int D) {
    int i;
    double* data = malloc(N * D * sizeof(double));
    for (i=0; i<(N*D); i++) {
        data[i] = rand() / (double)RAND_MAX;
    }
    return data;
}

static kdtree_t* build_tree(CuTest* tc, double* data, int N, int D,
                            int Nleaf, int treetype, int treeopts) {
    kdtree_t* kd;
    kd = kdtree_build(NULL, data, N, D, Nleaf, treetype, treeopts);
    if (!kd)
        return NULL;
    CuAssertIntEquals(tc, kdtree_check(kd), 0);
    return kd;
}
