/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef AN_GSL_UTILS_H
#define AN_GSL_UTILS_H

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

void gslutils_use_error_system();

/**
 Solves a least-squares matrix equation
 A X_i = B_i

 For NB pairs of X_i, B_i.

 NOTE: THIS DESTROYS A!

 A: MxN matrix
 B: array of NB x length-M vectors
 X: must be an array big enough to hold NB vectors.
   (they will be length-N).
 resids: if non-NULL, must be an array big enough to hold NB vectors
   (they will be length-M).

 The result vectors are freshly allocated and should be freed with gsl_vector_free().
 */
int gslutils_solve_leastsquares(gsl_matrix* A, gsl_vector** B,
                                gsl_vector** X, gsl_vector** resids,
                                int NB);

/**
 Same as above, but using varargs.  There must be exactly 3 * NB additional
 arguments, in the order:

 B0, &X0, &resid0, B1, &X1, &resid1, ...

 ie, the types must be repeating triples of:
 
 gsl_vector* b, gsl_vector** x, gsl_vector** resid

 */
int gslutils_solve_leastsquares_v(gsl_matrix* A, int NB, ...);

// C = A B
void gslutils_matrix_multiply(gsl_matrix* C, const gsl_matrix* A, const gsl_matrix* B);

int gslutils_invert_3x3(const double* A, double* B);

#endif
