#include <stdio.h>
#include <string.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_double.h>

int main() {
	gsl_matrix* A;
	gsl_matrix* U;
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

    gsl_matrix_set(A, 0, 0, -0.93);
    gsl_matrix_set(A, 0, 1,  1.80);
    gsl_matrix_set(A, 1, 0,  0);
    gsl_matrix_set(A, 1, 1,  0);

    gsl_linalg_SV_decomp_jacobi(A, V, S);

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

	return 0;
}
