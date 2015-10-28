/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */
//#include <iostream>
#include <Eigen/Dense>
//using namespace std;
using namespace Eigen;

extern "C" {

#include "eigen-math.h"
#include "stdio.h"

#if 0
} // fool emacs indenter
#endif

int eigen_solve_least_squares(ematrix_t* A, evector_t** B,
                               evector_t** X, int NB) {
    int i;
    int r,c;

    Map<Matrix<double, Dynamic, Dynamic, RowMajor> >
        mA(A->data, A->rows, A->cols);

    /*
     printf("mA:\n");
     for (r=0; r<mA.rows(); r++) {
     printf("[");
     for (c=0; c<mA.cols(); c++) {
     printf("%s %8.3g", c ? "," : " ", mA(r,c));
     }
     printf("]\n");
     }
     */

    /*
     for (i=0; i<NB; i++) {
     printf("mB(%i):\n", i);
     Map<VectorXd> mB(B[i]->data, B[i]->N, RowMajor);
     printf("[");
     for (c=0; c<mB.size(); c++) {
     printf("%s %8.3g", c ? "," : " ", mB(c));
     }
     printf("]\n");
     }
     */

    JacobiSVD<MatrixXd> svd(mA, ComputeThinU | ComputeThinV);
    for (i=0; i<NB; i++) {
        Map<VectorXd> b(B[i]->data, B[i]->N, RowMajor);
        VectorXd x = svd.solve(b);
        // copy results back to C space...
        for (r=0; r<x.size(); r++)
            evector_set(X[i], r, x[r]);
    }

    return 0;
}

#if 0
{ // fool emacs indenter
#endif
}


