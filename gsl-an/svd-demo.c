#include <stdio.h>
#include <string.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_double.h>

int main() {
	int ret;
	gsl_matrix* A;
    gsl_matrix* V;
    gsl_vector* S;
    gsl_vector* work;
    int i, j;
    int M, N;

    M = N = 2;

    work = gsl_vector_alloc(N);
    S = gsl_vector_alloc(N);
    A = gsl_matrix_alloc(M, N);
    V = gsl_matrix_alloc(N, N);

    gsl_matrix_set(A, 0, 0, -0.93);
    gsl_matrix_set(A, 0, 1,  1.80);
    gsl_matrix_set(A, 1, 0,  0);
    gsl_matrix_set(A, 1, 1,  0);

    gsl_linalg_SV_decomp(A, V, S, work);

    printf("S = [");
    for (i=0; i<N; i++)
        printf(" %g", gsl_vector_get(S, i));
    printf(" ]\n");


	return 0;
}
