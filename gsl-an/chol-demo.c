#include <stdio.h>
#include <string.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_double.h>

int main() {
	int ret;
	gsl_matrix A;
	double data[9];
	int i, j;

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

	for (i=0; i<A.size1; i++) {
		printf((i==0) ? "A = (" : "    (");
		for (j=0; j<A.size2; j++) {
			printf(" %12.5g ", gsl_matrix_get(&A, i, j));
		}
		printf(")\n");
	}
	printf("\n");

	ret = gsl_linalg_cholesky_decomp(&A);

	for (i=0; i<A.size1; i++) {
		printf((i==0) ? "L = (" : "    (");
		for (j=0; j<A.size2; j++) {
			printf(" %12.5g ", (j <= i) ? gsl_matrix_get(&A, i, j) : 0.0);
		}
		printf(")\n");
	}
	printf("\n");

	return 0;
}
