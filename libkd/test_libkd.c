/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <strings.h>

#include "os-features.h"
#include "errors.h"
#include "cutest.h"
#include "kdtree.h"
#include "mathutil.h"
#include "an-fls.h"

#include "test_libkd_common.c"

static int calculate_R(int leafid, int nlevels, int N) {
    int l;
    unsigned int mask, L;

    int nbottom = 1 << (nlevels - 1);
    

    mask = (1 << (nlevels-1));
    L = 0;
    // Compute the L index of the node one to the right of this node.
    int nextguy = leafid + 1;
    if (nextguy == nbottom)
        return N-1;
    for (l=0; l<(nlevels-1); l++) {
        mask /= 2;
        if (nextguy & mask) {
            L += N/2;
            N = (N+1)/2;
        } else {
            N = N/2;
        }
    }
    L--;
    return L;
}

int linearR(int leafid, int nbottom, int N) {
    int64_t res = leafid + 1;
    res *= N;
    res /= nbottom;
    return res - 1;
}

double linearRF(int leafid, int nbottom, int N) {
    double res = leafid + 1.0;
    res *= N;
    res /= (double)nbottom;
    return res - 1.0;
}

void tst_1(CuTest* ct) {
    kdtree_t* kd;
    double * data;
    int N = 1000;
    int Nleaf = 5;
    int D = 3;
    int i;

    data = random_points_d(N, D);
    kd = build_tree(ct, data, N, D, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);

    printf("kd->nbottom = %i, kd->nlevels = %i.\n", kd->nbottom, kd->nlevels);

    for (i=0; i<kd->nbottom; i++) {
        int R1 = kdtree_right(kd, kd->ninterior + i);
        int R2 = calculate_R(i, kd->nlevels, N);
        int R3 = linearR(i, kd->nbottom, N);
        double d3 = linearRF(i, kd->nbottom, N);

        printf("%i %i %i %g\n", R1, R2, R3, d3);
        printf("                               %s   %g\n",
               (R1 != R3) ? "***" : "   ",
               (double)R3 - d3);
        /*
         CuAssertIntEquals(ct, R1, R2);
         CuAssertIntEquals(ct, R1, R3);
         */
    }
    kdtree_free(kd);
}

static void compute_splitbits(int ndim, uint32_t* dimmask, uint32_t* dimbits, uint32_t* splitmask) {
    int D;
    int bits;
    uint32_t val;
    D = ndim;
    bits = 0;
    val = 1;
    while (val < D) {
        bits++;
        val *= 2;
    }
    *dimmask = val - 1;
    *dimbits = bits;
    *splitmask = ~(*dimmask);
}

void test_splitbits(CuTest* ct) {
    uint32_t dmask, dbits, smask;
    int dim;

    compute_splitbits(1, &dmask, &dbits, &smask);
    CuAssertIntEquals(ct, 0x0, dbits);
    CuAssertIntEquals(ct, 0x0, dmask);
    CuAssertIntEquals(ct,~0x0, smask);

    compute_splitbits(2, &dmask, &dbits, &smask);
    CuAssertIntEquals(ct, 0x1, dbits);
    CuAssertIntEquals(ct, 0x1, dmask);
    CuAssertIntEquals(ct,~0x1, smask);

    for (dim=3; dim<=4; dim++) {
        compute_splitbits(dim, &dmask, &dbits, &smask);
        CuAssertIntEquals(ct, 0x2, dbits);
        CuAssertIntEquals(ct, 0x3, dmask);
        CuAssertIntEquals(ct,~0x3, smask);
    }

    for (dim=5; dim<=8; dim++) {
        compute_splitbits(dim, &dmask, &dbits, &smask);
        CuAssertIntEquals(ct, 0x3, dbits);
        CuAssertIntEquals(ct, 0x7, dmask);
        CuAssertIntEquals(ct,~0x7, smask);
    }

    for (dim=9; dim<=16; dim++) {
        compute_splitbits(dim, &dmask, &dbits, &smask);
        CuAssertIntEquals(ct, 0x4, dbits);
        CuAssertIntEquals(ct, 0xF, dmask);
        CuAssertIntEquals(ct,~0xF, smask);
    }
}

void test_short_partition(CuTest* ct) {
    kdtree_t* kd;
    double* data;
    int N = 21;
    int Nleaf = 16;
    int D = 2;
    int i;
    double minval[D], maxval[D];
    uint16_t cdata[] = { 12669, 12669, 12669, 12669, 12669, 12669, 
                         12669, 12669, 12669, 12669, 12669, 12669,
                         13860, 13913, 14164, 14557, 15283, 17130,
                         17130, 17130, 17130 };


    minval[0] = -0.20710678118654757;
    minval[1] = -0.20710678118654757;
    maxval[0] =  1.2071067811865475;
    maxval[1] =  1.2071067811865475;

    data = calloc(N*D, sizeof(double));
    // convert from "cdata" to "data" space (I got the test data above
    // from the internal data representation of a problem tree).
    double scale = (maxval[0] - minval[0]) / (double)UINT16_MAX;
    for (i=0; i<N; i++) {
        data[2*i+0] = minval[0];
        data[2*i+1] = minval[0] + (double)cdata[i] * scale;
    }
    kd = kdtree_build_2(NULL, data, N, D, Nleaf, KDTT_DSS, KD_BUILD_SPLIT,
                        minval, maxval);
    CuAssertPtrNotNull(ct, kd);
    free(data);

    /*
     printf("kd:\n");
     kdtree_print(kd);
     */

    uint16_t* kdata = kd->data.s;
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, kdata[2*i+0], 0);
        CuAssertIntEquals(ct, kdata[2*i+1], cdata[i]);
    }
    CuAssertIntEquals(ct, 0, kdtree_check(kd));
    kdtree_free(kd);
}

void test_empty_node(CuTest* ct) {
    kdtree_t* kd;
    double* data;
    int N = 21;
    int Nleaf = 16;
    int D = 2;
    int i, ok;
    double minval[D], maxval[D];

    minval[0] = 0;
    minval[1] = 0;
    maxval[0] = 1;
    maxval[1] = 1;

    data = calloc(N*D, sizeof(double));
    // convert from "cdata" to "data" space
    double scale = (maxval[0] - minval[0]) / (double)UINT16_MAX;
    for (i=0; i<N; i++) {
        data[2*i+0] = minval[0];
        data[2*i+1] = minval[0] + 3 * scale;
    }

    kd = kdtree_build_2(NULL, data, N, D, Nleaf, KDTT_DSS, KD_BUILD_SPLIT,
                        minval, maxval);
    CuAssertPtrNotNull(ct, kd);
    free(data);

    uint16_t* kdata = kd->data.s;
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, kdata[2*i+0]);
        CuAssertIntEquals(ct, 3, kdata[2*i+1]);
    }

    ok = kdtree_check(kd);
    CuAssertIntEquals(ct, 0, ok);
    kdtree_free(kd);
}

static inline u8 node_level(int nodeid) {
    int val = (nodeid + 1) >> 1;
    u8 level = 0;
    while (val) {
        val = val >> 1;
        level++;
    }
    return level;
}

void test_2(CuTest* ct) {
    int N = 1024;
    int i;

    for (i=0; i<N; i++) {
        int L1 = node_level(i);
        int L3 = an_flsB(i+1);
        CuAssertIntEquals(ct, L1, L3);
    }
}

void test_nlevels(CuTest* ct) {
    // No nodes: no levels!
    CuAssertIntEquals(ct, 0, kdtree_nnodes_to_nlevels(0));
    // Single node.
    CuAssertIntEquals(ct, 1, kdtree_nnodes_to_nlevels(1));
    // sort of invalid input - incomplete level...
    CuAssertIntEquals(ct, 1, kdtree_nnodes_to_nlevels(2));
    CuAssertIntEquals(ct, 2, kdtree_nnodes_to_nlevels(3));
    CuAssertIntEquals(ct, 10, kdtree_nnodes_to_nlevels(1023));
}

static void run_test_nn(CuTest* tc, int treetype, int treeopts,
                        double eps) {
    int N = 1000;
    int Nleaf = 10;
    int D = 3;
    int Q = 10;
    kdtree_t* kd;
    double* origdata;
    double* treedata;
    double query[D];
    int i, q, d;

    srand(0);

    origdata = random_points_d(N, D);
    treedata = malloc(N * D * sizeof(double));
    memcpy(treedata, origdata, N*D*sizeof(double));

    kd = build_tree(tc, treedata, N, D, Nleaf, treetype, treeopts);

    CuAssert(tc, "kd", kd != NULL);

    if (treeopts & KD_BUILD_NO_LR)
        CuAssert(tc, "no lr", kd->lr == NULL);

    for (q=0; q<Q; q++) {
        int ind;
        double d2;
        double trued2;
        int trueind;
        for (d=0; d<D; d++)
            query[d] = rand() / (double)RAND_MAX;

        ind = kdtree_nearest_neighbour(kd, query, &d2);

        trued2 = LARGE_VAL;
        trueind = -1;
        for (i=0; i<N; i++) {
            double d2 = distsq(query, origdata + i*D, D);
            if (d2 < trued2) {
                trueind = i;
                trued2 = d2;
            }
        }

        /*
         printf("Naive : ind %i, dist %g.\n", trueind, sqrt(trued2));
         printf("Kdtree: ind %i, dist %g.\n", kd->perm[ind], sqrt(d2));
         */
        CuAssertIntEquals(tc, kd->perm[ind], trueind);

        if (fabs(sqrt(d2) - sqrt(trued2)) >= eps) {
            printf("Naive : %.12g\n", sqrt(trued2));
            printf("Kdtree: %.12g\n", sqrt(d2));
        }
        
        CuAssertDblEquals(tc, sqrt(d2), sqrt(trued2), eps);
    }

    kdtree_free(kd);
    free(treedata);
    free(origdata);
}

static void run_test_rs_ND(CuTest* tc, int treetype, int treeopts,
                           double eps, int N, int D) {
    int Nleaf = 10;
    int Q = 10;
    double rad2 = 0.01;
    double* origdata;
    double* treedata;
    kdtree_t* kd;
    double query[D];
    int i, q, d;

    srand(0);

    origdata = random_points_d(N, D);
    treedata = malloc(N * D * sizeof(double));
    memcpy(treedata, origdata, N*D*sizeof(double));

    kd = build_tree(tc, treedata, N, D, Nleaf, treetype, treeopts);
    CuAssert(tc, "kd", kd != NULL);

    for (q=0; q<Q; q++) {
        int ind;
        double d2;
        double trued2;
        int ntrue;
        kdtree_qres_t* res;

        for (d=0; d<D; d++)
            query[d] = rand() / (double)RAND_MAX;

        res = kdtree_rangesearch(kd, query, rad2);

        ntrue = 0;
        for (i=0; i<N; i++) {
            double d2 = distsq(query, origdata + i*D, D);
            if (d2 <= rad2) {
                ntrue++;
            }
        }

        CuAssertIntEquals(tc, res->nres, ntrue);

        for (i=0; i<res->nres; i++) {
            ind = res->inds[i];
            d2 = res->sdists[i];
            trued2 = distsq(query, origdata + ind*D, D);
        }

        CuAssert(tc, "res", res != NULL);
        for (i=0; i<res->nres; i++) {
            ind = res->inds[i];
            d2 = res->sdists[i];
            trued2 = distsq(query, origdata + ind*D, D);

            CuAssert(tc, "ind pos", ind >= 0);
            CuAssert(tc, "ind pos", ind < N);
            CuAssert(tc, "inrange", d2 <= rad2);
            CuAssert(tc, "inrange", trued2 <= rad2);
            CuAssert(tc, "d2pos", d2 >= 0.0);
            CuAssert(tc, "trued2pos", trued2 >= 0.0);
            CuAssertDblEquals(tc, sqrt(d2), sqrt(trued2), sqrt(eps));
        }
        /*
         printf("Naive : ind %i, dist %g.\n", trueind, sqrt(trued2));
         printf("Kdtree: ind %i, dist %g.\n", kd->perm[ind], sqrt(d2));
         */

        kdtree_free_query(res);
    }

    kdtree_free(kd);
    free(treedata);
    free(origdata);
}

static void run_test_rs(CuTest* tc, int treetype, int treeopts,
                        double eps) {
    int N = 1000;
    int D = 3;
    run_test_rs_ND(tc, treetype, treeopts, eps, N, D);
}

void test_rs_bb_duu(CuTest* tc) {
    run_test_rs(tc, KDTT_DUU, KD_BUILD_BBOX, 1e-9);
}

void test_rs_bb_ddd_small(CuTest* tc) {
    run_test_rs_ND(tc, KDTT_DOUBLE, KD_BUILD_BBOX, 1e-9, 10, 1);
}

void test_rs_bb_ddd(CuTest* tc) {
    run_test_rs(tc, KDTT_DOUBLE, KD_BUILD_BBOX, 1e-9);
}
void test_rs_split_ddd(CuTest* tc) {
    run_test_rs(tc, KDTT_DOUBLE, KD_BUILD_SPLIT, 1e-9);
}
void test_rs_both_ddd(CuTest* tc) {
    run_test_rs(tc, KDTT_DOUBLE, KD_BUILD_BBOX | KD_BUILD_SPLIT, 1e-9);
}

void test_rs_split_duu(CuTest* tc) {
    run_test_rs(tc, KDTT_DUU, KD_BUILD_SPLIT, 1e-9);
}

/**
 Sadly, does not work.

 void test_rs_split_ddu(CuTest* tc) {
 run_test_rs(tc, KDTT_DDU, KD_BUILD_SPLIT, 1e-9);
 }
 */

void test_rs_bb_dss(CuTest* tc) {
    run_test_rs(tc, KDTT_DSS, KD_BUILD_BBOX, 1e-5);
}
void test_rs_split_dss(CuTest* tc) {
    run_test_rs(tc, KDTT_DSS, KD_BUILD_SPLIT, 1e-5);
}




void test_nn_bb_ddd(CuTest* tc) {
    run_test_nn(tc, KDTT_DOUBLE, KD_BUILD_BBOX, 1e-9);
}

void test_nn_split_ddd(CuTest* tc) {
    run_test_nn(tc, KDTT_DOUBLE, KD_BUILD_SPLIT, 1e-9);
}

void test_nn_both_ddd(CuTest* tc) {
    run_test_nn(tc, KDTT_DOUBLE, KD_BUILD_SPLIT | KD_BUILD_BBOX, 1e-9);
}

void test_nn_split_duu(CuTest* tc) {
    run_test_nn(tc, KDTT_DUU, KD_BUILD_SPLIT, 1e-9);
}

void test_nn_bb_duu(CuTest* tc) {
    run_test_nn(tc, KDTT_DUU, KD_BUILD_BBOX, 1e-9);
}

void test_nn_split_dss(CuTest* tc) {
    run_test_nn(tc, KDTT_DSS, KD_BUILD_SPLIT, 1e-5);
}

void test_nn_split_dssB(CuTest* tc) {
    run_test_nn(tc, KDTT_DSS, KD_BUILD_SPLIT | KD_BUILD_NO_LR | KD_BUILD_SPLITDIM, 1e-5);
}

void test_nn_bb_dss(CuTest* tc) {
    run_test_nn(tc, KDTT_DSS, KD_BUILD_BBOX, 1e-5);
}

void test_nn_bb_dssB(CuTest* tc) {
    run_test_nn(tc, KDTT_DSS, KD_BUILD_BBOX | KD_BUILD_NO_LR, 1e-5);
}



void test_nn_split_ddd_linearlr(CuTest* tc) {
    run_test_nn(tc, KDTT_DOUBLE, KD_BUILD_SPLIT | KD_BUILD_NO_LR | KD_BUILD_LINEAR_LR, 1e-9);
}
void test_nn_split_duu_linearlr(CuTest* tc) {
    run_test_nn(tc, KDTT_DUU, KD_BUILD_SPLIT | KD_BUILD_SPLITDIM | KD_BUILD_NO_LR | KD_BUILD_LINEAR_LR, 1e-9);
}
void test_nn_split_dss_linearlr(CuTest* tc) {
    run_test_nn(tc, KDTT_DSS, KD_BUILD_SPLIT | KD_BUILD_SPLITDIM | KD_BUILD_NO_LR | KD_BUILD_LINEAR_LR, 1e-5);
}

void run_test_lr(CuTest* tc, int D, int Nleaf, int treetype, int treeopts) {
    int i;
    kdtree_t* kd;
    double* treedata;
    int32_t* lr;
    int N;
    for (N=100; N<=1000; N+=9) {
        treedata = random_points_d(N, D);
        kd = build_tree(tc, treedata, N, D, Nleaf, treetype, treeopts);
        CuAssert(tc, "kd", kd != NULL);

        lr = kd->lr;
        kd->lr = NULL;

        for (i=0; i<kd->nbottom; i++) {
            if (i)
                CuAssertIntEquals(tc, lr[i-1]+1, kdtree_left(kd, i + kd->ninterior));
            CuAssertIntEquals(tc, lr[i], kdtree_right(kd, i + kd->ninterior));
        }

        kd->lr = lr;
        kdtree_free(kd);
        free(treedata);
    }
}

void test_lr_ddd(CuTest* tc) {
    run_test_lr(tc, 3, 10, KDTT_DOUBLE, KD_BUILD_SPLIT);
}

void test_no_lr_with_ints(CuTest* tc) {
    double* data;
    kdtree_t* kd;
    int N = 1000;
    int D = 3;
    int Nleaf = 10;
    data = random_points_d(N, D);
    kd = build_tree(tc, data, N, D, Nleaf, KDTT_DSS, KD_BUILD_SPLIT | KD_BUILD_NO_LR);
    CuAssert(tc, "no kd", kd == NULL);
    free(data);

    errors_free();
}

