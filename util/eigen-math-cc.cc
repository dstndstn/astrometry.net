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

    Map<Matrix<double, Dynamic, Dynamic, RowMajor> >
        mA(A->data, A->rows, A->cols);

    //Map<MatrixXd> mB(B->data, B->rows, B->cols, RowMajor);

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

    //cout << "mB:" << mB;

    return 0;
}

#if 0
{ // fool emacs indenter
#endif
}


