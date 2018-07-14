/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"

// This one fails!
/*
 void tst_nullspace(CuTest* tc) {
 double cov[4] = {-0.93390448957619598, 1.8004204750064117, 0, 0};
 double U[4], V[4], S[2];
 {
 double* pcov[] = { cov, cov+2 };
 double* pU[]   = { U,   U  +2 };
 double* pV[]   = { V,   V  +2 };
 double eps, tol;
 eps = 1e-30;
 tol = 1e-30;
 svd(2, 2, 1, 1, eps, tol, pcov, S, pU, pV);
 }
 CuAssertDblEquals(tc, 0, S[1], 1e-6);
 CuAssertDblEquals(tc, 2.02822373, S[0], 1e-6);
 }
 */

void test_nullspace_gsl(CuTest* tc) {
    int i, j;
    gsl_matrix* A;
    gsl_matrix* V;
    gsl_matrix* U;
    gsl_vector* S;
    gsl_matrix_view vcov;
    double cov[4] = {-0.93390448957619598, 1.8004204750064117, 0, 0};
    int M=2, N=2;

    A = gsl_matrix_alloc(M, N);
    S = gsl_vector_alloc(N);
    V = gsl_matrix_alloc(N, N);

    vcov = gsl_matrix_view_array(cov, M, N);
    gsl_matrix_memcpy(A, &(vcov.matrix));

    gsl_linalg_SV_decomp_jacobi(A, V, S);
    // the U result is written to A.
    U = A;

    printf("S = [");
    for (i=0; i<N; i++)
        printf(" %g", gsl_vector_get(S, i));
    printf(" ]\n");

    printf("U = [");
    for (j=0; j<M; j++) {
        if (j)
            printf("    [");
        for (i=0; i<N; i++)
            printf(" %g", gsl_matrix_get(U, j, i));
        printf(" ]\n");
    }

    printf("V = [");
    for (j=0; j<N; j++) {
        if (j)
            printf("    [");
        for (i=0; i<N; i++)
            printf(" %g", gsl_matrix_get(V, j, i));
        printf(" ]\n");
    }

    CuAssertDblEquals(tc, 0, gsl_vector_get(S, 1), 1e-6);
    CuAssertDblEquals(tc, 2.02822373, gsl_vector_get(S, 0), 1e-6);

    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);

}
