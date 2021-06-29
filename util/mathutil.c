/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include "mathutil.h"
#include "keywords.h"
#include "log.h"
#include "errors.h"

#define InlineDefine InlineDefineC
#include "mathutil.inc"
#undef InlineDefine

#include "bl.h"

/**
 Returns 1 if the given point is inside the given polygon
 (listed as x0,y0, x1,y1, etc).
 */
int point_in_polygon(double x, double y, const dl* polygon) {
    size_t i;
    size_t N = dl_size(polygon) / 2;
    int inside = 0;
    for (i=0; i<N; i++) {
        size_t j = (i - 1 + N) % N;
        double xi, xj, yi, yj;
        yi = dl_get_const(polygon, i*2+1);
        yj = dl_get_const(polygon, j*2+1);
        if (yi == yj)
            continue;
        xi = dl_get_const(polygon, i*2+0);
        xj = dl_get_const(polygon, j*2+0);
        if (
            ( ((yi <= y) && (y < yj)) ||
              ((yj <= y) && (y < yi)) ) &&
            (x < (xi + ((xj - xi) * (y - yi) / (yj - yi))))
            ) {
            // toggle
            inside = 1-inside;
        }
    }
    return inside;
}

int get_output_image_size(int W, int H, int S,
                          int edgehandling,
                          int* newW, int* newH) {
    int outw, outh;
    if (S < 2) {
        logerr("Need scale >= 2");
        return -1;
    }
    if (edgehandling == 0) {
        // truncate.
        outw = W / S;
        outh = H / S;
    } else if (edgehandling == 1) {
        // average
        outw = (W + S-1) / S;
        outh = (H + S-1) / S;
    } else {
        logerr("Unknown edge handling code %i", edgehandling);
        return -1;
    }
    if (newW)
        *newW = outw;
    if (newH)
        *newH = outh;
    return 0;
}

float* average_image_f(const float* image, int W, int H,
                       int S, int edgehandling,
                       int* newW, int* newH,
                       float* output) {
    return average_weighted_image_f(image, NULL, W, H, S, edgehandling,
                                    newW, newH, output, 0.0);
}

float* average_weighted_image_f(const float* image, const float* weight,
                                int W, int H, int S, int edgehandling,
                                int* newW, int* newH, float* output, float nilval) {
    int outw, outh;
    int i,j;
    if (get_output_image_size(W, H, S, edgehandling, &outw, &outh))
        return NULL;
    if (output == NULL) {
        output = malloc((size_t)outw * (size_t)outh * sizeof(float));
        if (!output) {
            SYSERROR("Failed to allocate %i x %i floats", outw, outh);
            return NULL;
        }
    }
    for (j=0; j<outh; j++) {
        for (i=0; i<outw; i++) {
            float sum = 0.0;
            float wsum = 0.0;
            int I, J;
            int ii;
            for (J=0; J<S; J++) {
                if (j*S + J >= H)
                    break;
                for (I=0; I<S; I++) {
                    if (i*S + I >= W)
                        break;
                    ii = (j*S + J)*W + (i*S + I);
                    if (weight) {
                        wsum += weight[ii];
                        sum += image[ii] * weight[ii];
                    } else {
                        wsum += 1.0;
                        sum += image[ii];
                    }
                }
            }
            if (wsum == 0.0)
                output[j * outw + i] = nilval;
            else
                output[j * outw + i] = sum / wsum;
        }
    }
    if (newW)
        *newW = outw;
    if (newH)
        *newH = outh;
    return output;
}



// "borrowed" from <linux/bitops.h> from linux-2.4
static unsigned int my_hweight32(unsigned int w) {
    unsigned int res = (w & 0x55555555) + ((w >> 1) & 0x55555555);
    res = (res & 0x33333333) + ((res >> 2) & 0x33333333);
    res = (res & 0x0F0F0F0F) + ((res >> 4) & 0x0F0F0F0F);
    res = (res & 0x00FF00FF) + ((res >> 8) & 0x00FF00FF);
    return (res & 0x0000FFFF) + ((res >> 16) & 0x0000FFFF);
}

int invert_2by2_arr(const double* A, double* Ainv) {
    double det;
    double inv_det;
    det = A[0] * A[3] - A[1] * A[2];
    if (det == 0.0)
        return -1;
    inv_det = 1.0 / det;
    Ainv[0] =  A[3] * inv_det;
    Ainv[1] = -A[1] * inv_det;
    Ainv[2] = -A[2] * inv_det;
    Ainv[3] =  A[0] * inv_det;
    return 0;
}

int invert_2by2(const double A[2][2], double Ainv[2][2]) {
    double det;
    double inv_det;
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (det == 0.0)
        return -1;
    inv_det = 1.0 / det;
    Ainv[0][0] =  A[1][1] * inv_det;
    Ainv[0][1] = -A[0][1] * inv_det;
    Ainv[1][0] = -A[1][0] * inv_det;
    Ainv[1][1] =  A[0][0] * inv_det;
    return 0;
}

void tan_vectors(const double* pt, double* vec1, double* vec2) {
    double etax, etay, etaz, xix, xiy, xiz, eta_norm;
    double inv_en;
    // eta is a vector perpendicular to pt
    etax = -pt[1];
    etay =  pt[0];
    etaz = 0.0;
    eta_norm = hypot(etax, etay); //sqrt(etax * etax + etay * etay);
    inv_en = 1.0 / eta_norm;
    etax *= inv_en;
    etay *= inv_en;

    vec1[0] = etax;
    vec1[1] = etay;
    vec1[2] = etaz;

    // xi =  pt cross eta
    xix = -pt[2] * etay;
    xiy =  pt[2] * etax;
    xiz =  pt[0] * etay - pt[1] * etax;
    // xi has unit length since pt and eta have unit length.

    vec2[0] = xix;
    vec2[1] = xiy;
    vec2[2] = xiz;
}

int is_power_of_two(unsigned int x) {
    return (my_hweight32(x) == 1);
}

double vector_length_3(double* v) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

double vector_length_squared_3(double* v) {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

double dot_product_3(double* va, double* vb) {
    return va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2];
}

void matrix_matrix_3(double* ma, double* mb, double* result) {
    result[0] = ma[0]*mb[0] + ma[1]*mb[3] + ma[2]*mb[6];
    result[3] = ma[3]*mb[0] + ma[4]*mb[3] + ma[5]*mb[6];
    result[6] = ma[6]*mb[0] + ma[7]*mb[3] + ma[8]*mb[6];

    result[1] = ma[0]*mb[1] + ma[1]*mb[4] + ma[2]*mb[7];
    result[4] = ma[3]*mb[1] + ma[4]*mb[4] + ma[5]*mb[7];
    result[7] = ma[6]*mb[1] + ma[7]*mb[4] + ma[8]*mb[7];

    result[2] = ma[0]*mb[2] + ma[1]*mb[5] + ma[2]*mb[8];
    result[5] = ma[3]*mb[2] + ma[4]*mb[5] + ma[5]*mb[8];
    result[8] = ma[6]*mb[2] + ma[7]*mb[5] + ma[8]*mb[8];
}

void matrix_vector_3(double* m, double* v, double* result) {
    result[0] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2];
    result[1] = m[3]*v[0] + m[4]*v[1] + m[5]*v[2];
    result[2] = m[6]*v[0] + m[7]*v[1] + m[8]*v[2];
}


#define GAUSSIAN_SAMPLE_INVALID -1e300

double gaussian_sample(double mean, double stddev) {
    // from http://www.taygeta.com/random/gaussian.html
    // Algorithm by Dr. Everett (Skip) Carter, Jr.
    static double y2 = GAUSSIAN_SAMPLE_INVALID;
    double x1, x2, w, y1;

    // this algorithm generates random samples in pairs; the INVALID
    // jibba-jabba stores the second value until the next time the
    // function is called.

    if (y2 != GAUSSIAN_SAMPLE_INVALID) {
        y1 = y2;
        y2 = GAUSSIAN_SAMPLE_INVALID;
        return mean + y1 * stddev;
    }
    do {
        x1 = uniform_sample(-1, 1);
        x2 = uniform_sample(-1, 1);
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

    w = sqrt( (-2.0 * log(w)) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    return mean + y1 * stddev;
}

double uniform_sample(double low, double high) {
    if (low == high) return low;
    return low + (high - low)*((double)rand() / (double)RAND_MAX);
}

/* computes IN PLACE the inverse of a 3x3 matrix stored as a 9-vector
 with the first ROW of the matrix in positions 0-2, the second ROW
 in positions 3-5, and the last ROW in positions 6-8. */
double inverse_3by3(double *matrix)
{
    double det;
    double a11, a12, a13, a21, a22, a23, a31, a32, a33;
    double b11, b12, b13, b21, b22, b23, b31, b32, b33;

    a11 = matrix[0];
    a12 = matrix[1];
    a13 = matrix[2];
    a21 = matrix[3];
    a22 = matrix[4];
    a23 = matrix[5];
    a31 = matrix[6];
    a32 = matrix[7];
    a33 = matrix[8];

    det = a11 * ( a22 * a33 - a23 * a32 ) +
        a12 * ( a23 * a31 - a21 * a33 ) +
        a13 * ( a21 * a32 - a22 * a31 );

    if (det == 0.0) {
        return det;
    }

    b11 = + ( a22 * a33 - a23 * a32 ) / det;
    b12 = - ( a12 * a33 - a13 * a32 ) / det;
    b13 = + ( a12 * a23 - a13 * a22 ) / det;

    b21 = - ( a21 * a33 - a23 * a31 ) / det;
    b22 = + ( a11 * a33 - a13 * a31 ) / det;
    b23 = - ( a11 * a23 - a13 * a21 ) / det;

    b31 = + ( a21 * a32 - a22 * a31 ) / det;
    b32 = - ( a11 * a32 - a12 * a31 ) / det;
    b33 = + ( a11 * a22 - a12 * a21 ) / det;

    matrix[0] = b11;
    matrix[1] = b12;
    matrix[2] = b13;
    matrix[3] = b21;
    matrix[4] = b22;
    matrix[5] = b23;
    matrix[6] = b31;
    matrix[7] = b32;
    matrix[8] = b33;

    //fprintf(stderr,"matrix determinant = %g\n",det);

    return (det);
}


void image_to_xyz(double uu, double vv, double* s, double* transform) {
    double x, y, z;
    double length;
    assert(s);
    assert(transform);

    x = uu * transform[0] +
        vv * transform[1] +
        transform[2];

    y = uu * transform[3] +
        vv * transform[4] +
        transform[5];

    z = uu * transform[6] +
        vv * transform[7] +
        transform[8];

    length = sqrt(x*x + y*y + z*z);

    x /= length;
    y /= length;
    z /= length;

    s[0] = x;
    s[1] = y;
    s[2] = z;
}

/*
 star = trans * (field; 1)

 star is 3 x N
 field is 2 x N and is supplemented by a row of ones.
 trans is 3 x 3

 "star" and "field" are stored in lexicographical order, so the element at
 (r, c) is stored at array[r + c*H], where H is the "height" of the
 matrix: 3 for "star", 2 for "field".

 "trans" is stored in the transposed order (to be compatible with
 image_to_xyz and the previous version of this function).

 S = T F
 S F' = T F F'
 S F' (F F')^-1 = T
 */
void fit_transform(double* star, double* field, int N, double* trans) {
    int r, c, k;
    double FFt[9];
    double det;
    double* R;
    double* F;

    // build F = (field; ones)
    F = malloc(3 * N * sizeof(double));
    for (c=0; c<N; c++) {
        // row 0
        F[0 + c*3] = field[0 + c*2];
        // row 1
        F[1 + c*3] = field[1 + c*2];
        // row 2
        F[2 + c*3] = 1.0;
    }

    // first compute  FFt  =   F   *   F'
    //               (3x3) = (3xN) * (Nx3)
    for (r=0; r<3; r++)
        for (c=0; c<3; c++) {
            // FFt(r,c) = sum_k F(r,k) * F'(k,c)
            //          = sum_k F(r,k) * F(c,k)
            // FFt[r+c*3] = sum_k F[r + k*3] * F[c + k*3].
            double acc = 0.0;
            for (k=0; k<N; k++)
                acc += F[r + k*3] * F[c + k*3];
            FFt[r + c*3] = acc;
        }

    // invert FFt in-place.
    det = inverse_3by3(FFt);

    if (det < 0)
        fprintf(stderr, "WARNING (fit_transform) -- determinant<0\n");

    if (det == 0.0) {
        fprintf(stderr, "ERROR (fit_transform) -- determinant zero\n");
        return;
    }

    R = malloc(N * 3 * sizeof(double));

    // compute   R   =   F'  *  FFt  = F' inv(F F')
    //         (Nx3) = (Nx3) * (3x3)
    for (r=0; r<N; r++)
        for (c=0; c<3; c++) {
            // R(r,c) = sum_k F'(r,k) * FFt(k,c)
            //        = sum_k F(k,r) * FFt(k,c)
            //        = sum_k F[k + r*3] * FFt[k + c*3]
            double acc = 0.0;
            for (k=0; k<3; k++)
                acc += F[k + r*3] * FFt[k + c*3];
            R[r + c*N] = acc;
        }

    // finally, compute   T   =   S   *   R   = S F' inv(F F')
    //                  (3x3) = (3xN) * (Nx3)
    for (r=0; r<3; r++)
        for (c=0; c<3; c++) {
            // T(r,c) = sum_k S(r,k) * R(k,c)
            //        = sum_k S[r + k*3] * R[k + c*N]
            double acc = 0.0;
            for (k=0; k<N; k++)
                acc += star[r + k*3] * R[k + c*N];
            // "trans" is stored in transposed order.
            trans[c + r*3] = acc;
        }

    free(F);
    free(R);
}

