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

//int eigen_solve_least_squares(const ematrix_t* A, const evector_t** B,

int eigen_solve_least_squares(ematrix_t* A, evector_t** B,
                               evector_t** X, evector_t** resids,
                               int NB) {
    //
    //MatrixXd;
    //VectorXd;

    //Map<MatrixXd> mA(A->data, A->rows, A->cols, RowMajor);
    int i;

    Map<Matrix<double, Dynamic, Dynamic, RowMajor> >
        mA(A->data, A->rows, A->cols);

    //cout << "mA:" << mA;
    
    int r,c;

    printf("mA:\n");
    for (r=0; r<mA.rows(); r++) {
        printf("[");
        for (c=0; c<mA.cols(); c++) {
            printf("%s %8.3g", c ? "," : " ", mA(r,c));
        }
        printf("]\n");
    }

    for (i=0; i<NB; i++) {
        printf("mB(%i):\n", i);
        //Map<MatrixXd> mB(B[i]->data, B[i]->N, RowMajor);
        Map<VectorXd> mB(B[i]->data, B[i]->N, RowMajor);
        //cout << "mB:" << mB;
        printf("[");
        //for (c=0; c<mB.cols(); c++) {
        for (c=0; c<mB.size(); c++) {
            printf("%s %8.3g", c ? "," : " ", mB(c));
        }
        printf("]\n");
    }

    return 0;
}

#if 0
{ // fool emacs indenter
#endif
}


