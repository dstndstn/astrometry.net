/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef MATHUTIL_H
#define MATHUTIL_H

#include "starutil.h"
#include "keywords.h"

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



Const InlineDeclare int imax(int a, int b);

Const InlineDeclare int imin(int a, int b);

InlineDeclare double distsq_exceeds(double* d1, double* d2, int D, double limit);

Const InlineDeclare double square(double d);

// note, this is function works on angles in degrees; it wraps around
// at 360.
Const InlineDeclare int inrange(double ra, double ralow, double rahigh);

InlineDeclare double distsq(double* d1, double* d2, int D);

InlineDeclare void cross_product(double* v1, double* v2, double* cross);

InlineDeclare void normalize(double* x, double* y, double* z);

InlineDeclare void normalize_3(double* xyz);


#ifdef INCLUDE_INLINE_SOURCE
#include "mathutil.inc"
#endif


#endif
