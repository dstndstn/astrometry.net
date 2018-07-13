/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <stdarg.h>

#include "os-features.h"
#include "gslutils.h"
#include "errors.h"

static void errhandler(const char * reason,
                       const char * file,
                       int line,
                       int gsl_errno) {
    ERROR("GSL error: \"%s\" in %s:%i (gsl errno %i = %s)",
          reason, file, line,
          gsl_errno,
          gsl_strerror(gsl_errno));
}

void gslutils_use_error_system() {
    gsl_set_error_handler(errhandler);
}

int gslutils_invert_3x3(const double* A, double* B) {
    gsl_matrix* LU;
    gsl_permutation *p;
    gsl_matrix_view mB;
    int rtn = 0;
    int signum;

    p = gsl_permutation_alloc(3);
    gsl_matrix_const_view mA = gsl_matrix_const_view_array(A, 3, 3);
    mB = gsl_matrix_view_array(B, 3, 3);
    LU = gsl_matrix_alloc(3, 3);

    gsl_matrix_memcpy(LU, &mA.matrix);
    if (gsl_linalg_LU_decomp(LU, p, &signum) ||
        gsl_linalg_LU_invert(LU, p, &mB.matrix)) {
        ERROR("gsl_linalg_LU_decomp() or _invert() failed.");
        rtn = -1;
    }

    gsl_permutation_free(p);
    gsl_matrix_free(LU);
    return rtn;
}

void gslutils_matrix_multiply(gsl_matrix* C,
                              const gsl_matrix* A, const gsl_matrix* B) {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}

int gslutils_solve_leastsquares_v(gsl_matrix* A, int NB, ...) {
    int i, res;
    gsl_vector**  B = malloc(NB * sizeof(gsl_vector*));
    // Whoa, three-star programming!
    gsl_vector*** X = malloc(NB * sizeof(gsl_vector**));
    gsl_vector*** R = malloc(NB * sizeof(gsl_vector**));

    gsl_vector** Xtmp = malloc(NB * sizeof(gsl_vector*));
    gsl_vector** Rtmp = malloc(NB * sizeof(gsl_vector*));

    va_list va;
    va_start(va, NB);
    for (i=0; i<NB; i++) {
        B[i] = va_arg(va, gsl_vector*);
        X[i] = va_arg(va, gsl_vector**);
        R[i] = va_arg(va, gsl_vector**);
    }
    va_end(va);

    res = gslutils_solve_leastsquares(A, B, Xtmp, Rtmp, NB);
    for (i=0; i<NB; i++) {
        if (X[i])
            *(X[i]) = Xtmp[i];
        else
            gsl_vector_free(Xtmp[i]);
        if (R[i])
            *(R[i]) = Rtmp[i];
        else
            gsl_vector_free(Rtmp[i]);
    }
    free(Xtmp);
    free(Rtmp);
    free(X);
    free(R);
    free(B);
    return res;
}

int gslutils_solve_leastsquares(gsl_matrix* A, gsl_vector** B,
                                gsl_vector** X, gsl_vector** resids,
                                int NB) {
    int i;
    gsl_vector *tau, *resid = NULL;
    Unused int ret;
    int M, N;

    M = A->size1;
    N = A->size2;

    for (i=0; i<NB; i++) {
        assert(B[i]);
        assert(B[i]->size == M);
    }

    tau = gsl_vector_alloc(MIN(M, N));
    assert(tau);

    ret = gsl_linalg_QR_decomp(A, tau);
    assert(ret == 0);
    // A,tau now contains a packed version of Q,R.

    for (i=0; i<NB; i++) {
        if (!resid) {
            resid = gsl_vector_alloc(M);
            assert(resid);
        }
        X[i] = gsl_vector_alloc(N);
        assert(X[i]);
        ret = gsl_linalg_QR_lssolve(A, tau, B[i], X[i], resid);
        assert(ret == 0);
        if (resids) {
            resids[i] = resid;
            resid = NULL;
        }
    }

    gsl_vector_free(tau);
    if (resid)
        gsl_vector_free(resid);

    return 0;
}

