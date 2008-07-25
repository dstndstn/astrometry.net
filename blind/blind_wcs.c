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
#include <math.h>
#include <assert.h>

#include "blind_wcs.h"
#include "mathutil.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "sip.h"
#include "sip_qfits.h"
#include "log.h"

int blind_wcs_compute(double* starxyz,
                      double* fieldxy,
                      int N,
                      // output:
                      tan_t* tan,
                      double* p_scale) {
	int i, j, k;
	double star_cm[3] = {0, 0, 0};
	double field_cm[2] = {0, 0};
	double cov[4] = {0, 0, 0, 0};
	double R[4] = {0, 0, 0, 0};
	double scale;
	// projected star coordinates
	double* p;
	// relative field coordinates
	double* f;
	double pcm[2] = {0, 0};

    gsl_matrix* A;
    gsl_matrix* U;
    gsl_matrix* V;
    gsl_vector* S;
    gsl_vector* work;
    gsl_matrix_view vcov;
    gsl_matrix_view vR;

	// -get field & star centers-of-mass of the matching quad.
	//  (this will become the tangent point)
	for (i=0; i<N; i++) {
		star_cm[0] += starxyz[i*3 + 0];
		star_cm[1] += starxyz[i*3 + 1];
		star_cm[2] += starxyz[i*3 + 2];
		field_cm[0] += fieldxy[i*2 + 0];
		field_cm[1] += fieldxy[i*2 + 1];
	}
	field_cm[0] /= (double)N;
	field_cm[1] /= (double)N;
	normalize_3(star_cm);

	// -allocate and fill "p" and "f" arrays. ("projected" and "field")
	p = malloc(N * 2 * sizeof(double));
	f = malloc(N * 2 * sizeof(double));
	j = 0;

	for (i=0; i<N; i++) {
		bool ok;
		// -project the stars around the quad center of mass
		ok = star_coords(starxyz + i*3, star_cm, p + 2*i, p + 2*i + 1);
		assert(ok);
		// -grab the corresponding field coords
		f[2*i+0] = fieldxy[2*i+0] - field_cm[0];
		f[2*i+1] = fieldxy[2*i+1] - field_cm[1];
	}

	// -compute the center of mass of the projected stars and subtract it out.
	//  This will be close to zero, but we need it to be exactly zero
	//  before we start rigid Procrustes
	for (i=0; i<N; i++) {
		pcm[0] += p[2*i + 0];
		pcm[1] += p[2*i + 1];
	}
	pcm[0] /= (double)N;
	pcm[1] /= (double)N;
	for (i=0; i<N; i++) {
		p[2*i + 0] -= pcm[0];
		p[2*i + 1] -= pcm[1];
	}

	// -compute the covariance between field positions and projected
	//  positions of the corresponding stars.
	for (i=0; i<N; i++)
		for (j=0; j<2; j++)
			for (k=0; k<2; k++)
				cov[j*2 + k] += p[i*2 + k] * f[i*2 + j];

	for (i=0; i<4; i++)
        assert(isfinite(cov[i]));

	// -run SVD
    V = gsl_matrix_alloc(2, 2);
    S = gsl_vector_alloc(2);
    work = gsl_vector_alloc(2);
    vcov = gsl_matrix_view_array(cov, 2, 2);
    vR   = gsl_matrix_view_array(R, 2, 2);
    A = &(vcov.matrix);
    // The Jacobi version doesn't always compute an orthonormal U if S has zeros.
    //gsl_linalg_SV_decomp_jacobi(A, V, S);
    gsl_linalg_SV_decomp(A, V, S, work);
    // the U result is written to A.
    U = A;
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(work);
    // R = V U'
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, V, U, 0.0, &(vR.matrix));

	for (i=0; i<4; i++)
        assert(isfinite(R[i]));

	// -compute scale: make the variances equal.
	{
		double pvar, fvar;
		pvar = fvar = 0.0;
		for (i=0; i<N; i++)
			for (j=0; j<2; j++) {
				pvar += square(p[i*2 + j]);
				fvar += square(f[i*2 + j]);
			}
		scale = sqrt(pvar / fvar);
	}

	// -compute WCS parameters.
	xyzarr2radecdegarr(star_cm, tan->crval);
	tan->crpix[0] = field_cm[0] + pcm[0];
	tan->crpix[1] = field_cm[1] + pcm[1];
	scale = rad2deg(scale);
	// The CD rows are reversed from R because star_coords and the Intermediate
	// World Coordinate System consider x and y to be exchanged.
	tan->cd[0][0] = R[2] * scale; // CD1_1
	tan->cd[0][1] = R[3] * scale; // CD1_2
	tan->cd[1][0] = R[0] * scale; // CD2_1
	tan->cd[1][1] = R[1] * scale; // CD2_2

    assert(isfinite(tan->cd[0][0]));
    assert(isfinite(tan->cd[0][1]));
    assert(isfinite(tan->cd[1][0]));
    assert(isfinite(tan->cd[1][1]));

	/*
	 * Did I add pcm in the right direction?
	 * I've never seen it make more of a difference than machine epsilon.
	  {
		double rmsA = 0.0;
		double rmsB = 0.0;
		double rmsC = 0.0;
		tan_t tan_tst;
		memcpy(&tan_tst, tan, sizeof(tan_t));

		tan_tst.crpix[0] = field_cm[0] + pcm[0];
		tan_tst.crpix[1] = field_cm[1] + pcm[1];

		for (i=0; i<N; i++) {
			double x, y;
			tan_xyzarr2pixelxy(&tan_tst, starxyz + i*3, &x, &y);
			rmsA += square(x - fieldxy[i*2 + 0]) + square(y - fieldxy[i*2 + 1]);
		}
		rmsA = sqrt(rmsA / (double)N);

		tan_tst.crpix[0] = field_cm[0];
		tan_tst.crpix[1] = field_cm[1];

		for (i=0; i<N; i++) {
			double x, y;
			tan_xyzarr2pixelxy(&tan_tst, starxyz + i*3, &x, &y);
			rmsB += square(x - fieldxy[i*2 + 0]) + square(y - fieldxy[i*2 + 1]);
		}
		rmsB = sqrt(rmsB / (double)N);

		tan_tst.crpix[0] = field_cm[0] - pcm[0];
		tan_tst.crpix[1] = field_cm[1] - pcm[1];

		for (i=0; i<N; i++) {
			double x, y;
			tan_xyzarr2pixelxy(&tan_tst, starxyz + i*3, &x, &y);
			rmsC += square(x - fieldxy[i*2 + 0]) + square(y - fieldxy[i*2 + 1]);
		}
		rmsC = sqrt(rmsC / (double)N);

		printf("rmsA = %.16g pix.\n", rmsA);
		printf("rmsB = %.16g pix.\n", rmsB);
		printf("rmsC = %.16g pix.\n", rmsC);
	}
	*/

	if (p_scale) *p_scale = scale;

	free(p);
	free(f);
    return 0;
}

