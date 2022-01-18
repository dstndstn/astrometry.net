/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef MATHUTIL_H
#define MATHUTIL_H

#include "astrometry/keywords.h"
#include "bl.h"

#define LARGE_VAL  1e30
#define LARGE_VALF 1e30f

int point_in_polygon(double x, double y, const dl* polygon);

/*
 Given a point "pt", computes two unit vectors that are tangent
 to the point and perpendicular to each other.
 */
void tan_vectors(const double* pt, double* vec1, double* vec2);

int invert_2by2(const double A[2][2], double Ainv[2][2]);

int invert_2by2_arr(const double* A, double* Ainv);

int is_power_of_two(unsigned int x);

void matrix_matrix_3(double* m1, double* m2, double* result);

void matrix_vector_3(double* m, double* v, double* result);

double dot_product_3(double* v1, double* v2);

double vector_length_3(double* v);

double vector_length_squared_3(double* v);

double inverse_3by3(double *matrix);

void image_to_xyz(double uu, double vv, double* s, double* transform);

void fit_transform(double* star, double* field, int N, double* trans);

double uniform_sample(double low, double high);

double gaussian_sample(double mean, double stddev);

// just drop partial blocks off the end.
#define EDGE_TRUNCATE 0
// just average the pixels in partial blocks.
#define EDGE_AVERAGE  1

int get_output_image_size(int W, int H, int blocksize, int edgehandling,
                          int* outw, int* outh);

#define SIGN(x) (((x) == 0) ? 0.0 : (((x) > 0) ? 1.0 : -1.0))

/**
 Average the image in "blocksize" x "blocksize" blocks, placing the
 output in the "output" image.  The output image will have size
 "*newW" by "*newH".  If you pass "output = NULL", memory will be
 allocated for the output image.  It is valid to pass in "output" =
 "image".  The output image is returned.
 */
float* average_image_f(const float* image, int W, int H,
                       int blocksize, int edgehandling,
                       int* newW, int* newH,
                       float* output);

float* average_weighted_image_f(const float* image, const float* weight,
                                int W, int H,
                                int blocksize, int edgehandling,
                                int* newW, int* newH,
                                float* output, float nilval);

Const InlineDeclare int imax(int a, int b);

Const InlineDeclare int imin(int a, int b);

InlineDeclare double distsq_exceeds(double* d1, double* d2, int D, double limit);

Const InlineDeclare double square(double d);

// note, this function works on angles in degrees; it wraps around
// at 360.
Const InlineDeclare int inrange(double ra, double ralow, double rahigh);

InlineDeclare double distsq(const double* d1, const double* d2, int D);

InlineDeclare void cross_product(double* v1, double* v2, double* cross);

InlineDeclare void normalize(double* x, double* y, double* z);

InlineDeclare void normalize_3(double* xyz);


#ifdef INCLUDE_INLINE_SOURCE
#define InlineDefine InlineDefineH
#include "astrometry/mathutil.inc"
#undef InlineDefine
#endif


#endif
