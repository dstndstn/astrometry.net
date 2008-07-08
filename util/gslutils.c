/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <assert.h>
#include <sys/param.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_linalg.h>

#include "gslutils.h"

int gslutils_solve_leastsquares(gsl_matrix* A, gsl_vector** B,
                                gsl_vector** X, gsl_vector** resids,
                                int NB) {
    //double rmsB=0;
    int i;
    gsl_vector *tau, *resid = NULL;
    int ret;
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
    /*
     // RMS of (AX-B).
     for (j=0; j<M; j++) {
     rmsB += square(gsl_vector_get(resid1, j));
     rmsB += square(gsl_vector_get(resid2, j));
     }
     if (M > 0)
     rmsB = sqrt(rmsB / (double)(M*2));
     debug("gsl rms                = %g\n", rmsB);
     */

    gsl_vector_free(tau);
    if (resid)
        gsl_vector_free(resid);

    return 0;
}

