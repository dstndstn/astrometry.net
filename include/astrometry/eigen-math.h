/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
typedef struct {
    double* data;
    int cols;
    int rows;
} ematrix_t;

typedef struct {
    double* data;
    int N;
} evector_t;

ematrix_t* ematrix_new(int R, int C);

evector_t* evector_new(int N);

void ematrix_set(ematrix_t* m, int r, int c, double v);
double ematrix_get(const ematrix_t* m, int r, int c);

void evector_set(evector_t* m, int i, double v);
double evector_get(const evector_t* m, int i);


//int eigen_solve_least_squares(const ematrix_t* A, const evector_t** B,
int eigen_solve_least_squares(ematrix_t* A, evector_t** B,
                              evector_t** X, int NB);


