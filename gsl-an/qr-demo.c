#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>

int main() {
	int ret;
	int i, j;
	gsl_vector* tau;
	gsl_matrix *A;
	gsl_matrix *Q, *R, *RTR;
	gsl_matrix_view Rtop;
	int M = 4;
	int N = 3;

	/*
	  gsl_matrix A;
	  double data[9];
	  memset(&A, 0, sizeof(gsl_matrix));
	  A.size1 = 3;
	  A.size2 = 3;
	  A.tda = 3;
	  A.data = data;
	  gsl_matrix_set(&A, 0, 0, 34.0);
	  gsl_matrix_set(&A, 0, 1, 4.0);
	  gsl_matrix_set(&A, 0, 2, 14.0);
	  gsl_matrix_set(&A, 1, 0, 1.0);
	  gsl_matrix_set(&A, 1, 1, 8.0);
	  gsl_matrix_set(&A, 1, 2, 3.0);
	  gsl_matrix_set(&A, 2, 0, 7.0);
	  gsl_matrix_set(&A, 2, 1, 1.0);
	  gsl_matrix_set(&A, 2, 2, 8.0);
	*/

	A = gsl_matrix_alloc(M, N);

	for (i=0; i<M; i++)
		for (j=0; j<N; j++)
			gsl_matrix_set(A, i, j, (double)rand()/(double)RAND_MAX);

	for (i=0; i<A->size1; i++) {
		printf((i==0) ? "A = (" : "    (");
		for (j=0; j<A->size2; j++) {
			printf(" %12.5g ", gsl_matrix_get(A, i, j));
		}
		printf(")\n");
	}
	printf("\n");

	tau = gsl_vector_alloc(N);

	ret = gsl_linalg_QR_decomp(A, tau);

	Q = gsl_matrix_alloc(M, M);
	R = gsl_matrix_alloc(M, N);

	ret = gsl_linalg_QR_unpack(A, tau, Q, R);

	for (i=0; i<Q->size1; i++) {
		printf((i==0) ? "Q = (" : "    (");
		for (j=0; j<Q->size2; j++) {
			printf(" %12.5g ", gsl_matrix_get(Q, i, j));
		}
		printf(")\n");
	}
	printf("\n");

	for (i=0; i<R->size1; i++) {
		printf((i==0) ? "R = (" : "    (");
		for (j=0; j<R->size2; j++) {
			printf(" %12.5g ", gsl_matrix_get(R, i, j));
		}
		printf(")\n");
	}
	printf("\n");


	Rtop = gsl_matrix_submatrix(R, 0, 0, N, N);
	RTR = gsl_matrix_alloc(N, N);
	gsl_matrix_memcpy(RTR, &(Rtop.matrix));
	ret = gsl_blas_dtrmm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
						 1.0, RTR, RTR);
	//(Rtop.matrix), &(Rtop.matrix));

	for (i=0; i<RTR->size1; i++) {
		printf((i==0) ? "RTR = (" : "      (");
		for (j=0; j<RTR->size2; j++) {
			printf(" %12.5g ", gsl_matrix_get(RTR, i, j));
		}
		printf(")\n");
	}
	printf("\n");

	gsl_matrix_free(RTR);


	gsl_matrix_free(Q);
	gsl_matrix_free(R);
	gsl_vector_free(tau);

	gsl_matrix_free(A);

	return 0;
}
