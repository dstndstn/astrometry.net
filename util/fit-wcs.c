/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2012 Dustin Lang.

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

#include "fit-wcs.h"
#include "starutil.h"
#include "mathutil.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "sip.h"
#include "sip_qfits.h"
#include "log.h"

static
int fit_tan_wcs_solve(const double* starxyz,
					const double* fieldxy,
					const double* weights,
					int N,
					const double* crpix,
					const tan_t* tanin,
					tan_t* tanout,
					double* p_scale) {
	int i, j, k;
	double field_cm[2] = {0, 0};
	double cov[4] = {0, 0, 0, 0};
	double R[4] = {0, 0, 0, 0};
	double scale;
	// projected star coordinates
	double* p;
	// relative field coordinates
	double* f;
	double pcm[2] = {0, 0};
	double w = 0;
	double totalw;

    gsl_matrix* A;
    gsl_matrix* U;
    gsl_matrix* V;
    gsl_vector* S;
    gsl_vector* work;
    gsl_matrix_view vcov;
    gsl_matrix_view vR;

	double crxyz[3];

	double star_cm[3] = {0, 0, 0};

	assert(((tanin != NULL) && (crpix != NULL)) || ((tanin == NULL) && (crpix == NULL)));

	if (tanin) {
		// default vals...
		memcpy(tanout, tanin, sizeof(tan_t));
	} else {
		memset(tanout, 0, sizeof(tan_t));
	}

	// -allocate and fill "p" and "f" arrays. ("projected" and "field")
	p = malloc(N * 2 * sizeof(double));
	f = malloc(N * 2 * sizeof(double));

	// -get field center-of-mass
	totalw = 0.0;
	for (i=0; i<N; i++) {
		w = (weights ? weights[i] : 1.0);
		field_cm[0] += w * fieldxy[i*2 + 0];
		field_cm[1] += w * fieldxy[i*2 + 1];
		totalw += w;
	}
	field_cm[0] /= totalw;
	field_cm[1] /= totalw;
	// Subtract it out.
	for (i=0; i<N; i++) {
		f[2*i+0] = fieldxy[2*i+0] - field_cm[0];
		f[2*i+1] = fieldxy[2*i+1] - field_cm[1];
	}

	if (tanin) {
		// Use original WCS to set the center of projection to the new crpix.
		tan_pixelxy2xyzarr(tanin, crpix[0], crpix[1], crxyz);
		for (i=0; i<N; i++) {
			Unused anbool ok;
			// -project the stars around crval
			ok = star_coords(starxyz + i*3, crxyz, TRUE, p + 2*i, p + 2*i + 1);
			assert(ok);
		}
	} else {
		// -get the star center-of-mass (this will become the tangent point CRVAL)
		for (i=0; i<N; i++) {
			w = (weights ? weights[i] : 1.0);
			star_cm[0] += w * starxyz[i*3 + 0];
			star_cm[1] += w * starxyz[i*3 + 1];
			star_cm[2] += w * starxyz[i*3 + 2];
		}
		normalize_3(star_cm);
		// -project the stars around their center of mass
		for (i=0; i<N; i++) {
			Unused anbool ok;
			ok = star_coords(starxyz + i*3, star_cm, TRUE, p + 2*i, p + 2*i + 1);
			assert(ok);
		}
	}

	// -compute the center of mass of the projected stars and subtract it out.
	for (i=0; i<N; i++) {
		w = (weights ? weights[i] : 1.0);
		pcm[0] += w * p[2*i + 0];
		pcm[1] += w * p[2*i + 1];
	}
	pcm[0] /= totalw;
	pcm[1] /= totalw;
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
    gsl_vector_free(S);
    gsl_vector_free(work);
    // R = V U'
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, V, U, 0.0, &(vR.matrix));
    gsl_matrix_free(V);

	for (i=0; i<4; i++)
        assert(isfinite(R[i]));

	// -compute scale: make the variances equal.
	{
		double pvar, fvar;
		pvar = fvar = 0.0;
		for (i=0; i<N; i++) {
			w = (weights ? weights[i] : 1.0);
			for (j=0; j<2; j++) {
				pvar += w * square(p[i*2 + j]);
				fvar += w * square(f[i*2 + j]);
			}
		}
		scale = sqrt(pvar / fvar);
	}

	// -compute WCS parameters.
	scale = rad2deg(scale);

	tanout->cd[0][0] = R[0] * scale; // CD1_1
	tanout->cd[0][1] = R[1] * scale; // CD1_2
	tanout->cd[1][0] = R[2] * scale; // CD2_1
	tanout->cd[1][1] = R[3] * scale; // CD2_2

    assert(isfinite(tanout->cd[0][0]));
    assert(isfinite(tanout->cd[0][1]));
    assert(isfinite(tanout->cd[1][0]));
    assert(isfinite(tanout->cd[1][1]));

	if (tanin) {
		// CRPIX is fixed.
		tanout->crpix[0] = crpix[0];
		tanout->crpix[1] = crpix[1];
		// Set CRVAL temporarily...
		tan_pixelxy2radec(tanin, crpix[0], crpix[1],
						  tanout->crval+0, tanout->crval+1);
		// Shift CRVAL so that the center of the quad is in the right place.
		{
			double ix,iy;
			double dx,dy;
			double dxyz[3];
			tan_pixelxy2iwc(tanout, field_cm[0], field_cm[1], &ix, &iy);
			dx = rad2deg(pcm[0]) - ix;
			dy = rad2deg(pcm[1]) - iy;
			tan_iwc2xyzarr(tanout, dx, dy, dxyz);
			xyzarr2radecdeg(dxyz, tanout->crval + 0, tanout->crval + 1);
		}
	} else {
		tanout->crpix[0] = field_cm[0];
		tanout->crpix[1] = field_cm[1];
		
		xyzarr2radecdegarr(star_cm, tanout->crval);

		// FIXME -- we ignore pcm.  It should get added back in (after
		// multiplication by CD in the appropriate units) to either crval or
		// crpix.  It's a very small correction probably of the same size
		// as the other approximations we're making.
	}

	if (p_scale) *p_scale = scale;
	free(p);
	free(f);
    return 0;
}




int fit_tan_wcs_move_tangent_point_weighted(const double* starxyz,
										  const double* fieldxy,
										  const double* weights,
										  int N,
										  const double* crpix,
										  const tan_t* tanin,
										  tan_t* tanout) {
	return fit_tan_wcs_solve(starxyz, fieldxy, weights, N, crpix, tanin, tanout,
						   NULL);
}

int fit_tan_wcs_move_tangent_point(const double* starxyz,
								 const double* fieldxy,
								 int N,
								 const double* crpix,
								 const tan_t* tanin,
								 tan_t* tanout) {
	return fit_tan_wcs_move_tangent_point_weighted(starxyz, fieldxy, NULL,
												 N, crpix, tanin, tanout);
}


int fit_tan_wcs_weighted(const double* starxyz,
					 const double* fieldxy,
					 const double* weights,
					 int N,
					 // output:
					 tan_t* tan,
					 double* p_scale) {
	return fit_tan_wcs_solve(starxyz, fieldxy, weights, N, NULL, NULL, tan, p_scale);
}

int fit_tan_wcs(const double* starxyz,
			const double* fieldxy,
			int N,
			// output:
			tan_t* tan,
			double* p_scale) {
	return fit_tan_wcs_weighted(starxyz, fieldxy, NULL, N,
									tan, p_scale);
}

