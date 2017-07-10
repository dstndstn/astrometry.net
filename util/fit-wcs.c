/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <assert.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"

#include "os-features.h"
#include "fit-wcs.h"
#include "starutil.h"
#include "mathutil.h"
#include "sip.h"
#include "sip_qfits.h"
#include "log.h"
#include "errors.h"
#include "gslutils.h"
#include "sip-utils.h"

int fit_sip_wcs_2(const double* starxyz,
                  const double* fieldxy,
                  const double* weights,
                  int M,
                  int sip_order,
                  int inv_order,
                  int W, int H,
                  int crpix_center,
                  double* crpix,
                  int doshift,
                  sip_t* sipout) {
    tan_t wcs;
    memset(&wcs, 0, sizeof(tan_t));
    // Start with a TAN
    if (fit_tan_wcs(starxyz, fieldxy, M, &wcs, NULL)) {
        ERROR("Failed to fit for TAN WCS");
        return -1;
    }

    if (crpix || crpix_center) {
        double cx,cy;
        double cr,cd;
        if (crpix) {
            cx = crpix[0];
            cy = crpix[1];
        } else {
            int i;
            if (W == 0) {
                for (i=0; i<M; i++) {
                    W = MAX(W, (int)ceil(fieldxy[2*i + 0]));
                }
            }
            if (H == 0) {
                for (i=0; i<M; i++) {
                    H = MAX(H, (int)ceil(fieldxy[2*i + 1]));
                }
            }
            cx = 1. + 0.5 * W;
            cy = 1. + 0.5 * H;
        }
        tan_pixelxy2radec(&wcs, cx, cy, &cr, &cd);
        wcs.crpix[0] = cx;
        wcs.crpix[1] = cy;
        wcs.crval[0] = cr;
        wcs.crval[1] = cd;
    }
    wcs.imagew = W;
    wcs.imageh = H;

    return fit_sip_wcs(starxyz, fieldxy, weights, M, &wcs,
                       sip_order, inv_order, doshift, sipout);
}

int fit_sip_wcs(const double* starxyz,
                const double* fieldxy,
                const double* weights,
                int M,
                const tan_t* tanin1,
                int sip_order,
                int inv_order,
                int doshift,
                sip_t* sipout) {
    int sip_coeffs;
    double xyzcrval[3];
    double cdinv[2][2];
    double sx, sy, sU, sV, su, sv;
    int N;
    int i, j, p, q, order;
    double totalweight;
    int rtn;
    gsl_matrix *mA;
    gsl_vector *b1, *b2, *x1, *x2;
    gsl_vector *r1=NULL, *r2=NULL;
    tan_t tanin2;
    int ngood;
    const tan_t* tanin = &tanin2;
    // We need at least the linear terms to compute CD.
    if (sip_order < 1)
        sip_order = 1;

    // convenience: allow the user to call like:
    //    fit_sip_wcs(... &(sipout.wcstan), ..., sipout);
    memcpy(&tanin2, tanin1, sizeof(tan_t));

    memset(sipout, 0, sizeof(sip_t));
    memcpy(&(sipout->wcstan), tanin, sizeof(tan_t));
    sipout->a_order  = sipout->b_order  = sip_order;
    sipout->ap_order = sipout->bp_order = inv_order;

    // The SIP coefficients form an (order x order) upper triangular
    // matrix missing the 0,0 element.
    sip_coeffs = (sip_order + 1) * (sip_order + 2) / 2;
    N = sip_coeffs;

    if (M < N) {
        ERROR("Too few correspondences for the SIP order specified (%i < %i)\n", M, N);
        return -1;
    }

    mA = gsl_matrix_alloc(M, N);
    b1 = gsl_vector_alloc(M);
    b2 = gsl_vector_alloc(M);
    assert(mA);
    assert(b1);
    assert(b2);

    /*
     *  We use a clever trick to estimate CD, A, and B terms in two
     *  seperated least squares fits, then finding A and B by multiplying
     *  the found parameters by CD inverse.
     * 
     *  Rearranging the SIP equations (see sip.h) we get the following
     *  matrix operation to compute x and y in world intermediate
     *  coordinates, which is convienently written in a way which allows
     *  least squares estimation of CD and terms related to A and B.
     * 
     *  First use the x's to find the first set of parametetrs
     * 
     *     +--------------------- Intermediate world coordinates in DEGREES
     *     |          +--------- Pixel coordinates u and v in PIXELS
     *     |          |     +--- Polynomial u,v terms in powers of PIXELS
     *     v          v     v
     *   ( x1 )   ( 1 u1 v1 p1 )   (sx              )
     *   ( x2 ) = ( 1 u2 v2 p2 ) * (cd11            ) :
     *   ( x3 )   ( 1 u3 v3 p3 )   (cd12            ) :
     *   ( ...)   (   ...    )     (cd11*A + cd12*B ) :
     * cd11 is a scalar, degrees per pixel
     * cd12 is a scalar, degrees per pixel
     * cd11*A and cs12*B are mixture of SIP terms (A,B) and CD matrix
     *   (cd11,cd12)
     * 
     *  Then find cd21 and cd22 with the y's
     * 
     *   ( y1 )   ( 1 u1 v1 p1 )   (sy              )
     *   ( y2 ) = ( 1 u2 v2 p2 ) * (cd21            ) :
     *   ( y3 )   ( 1 u3 v3 p3 )   (cd22            ) :
     *   ( ...)   (   ...    )     (cd21*A + cd22*B ) : (Y4)
     *  y2: scalar, degrees per pixel
     *  y3: scalar, degrees per pixel
     *  Y4: mixture of SIP terms (A,B) and CD matrix (cd21,cd22)
     * 
     *  These are both standard least squares problems which we solve with
     *  QR decomposition, ie
     *      min_{cd,A,B} || x - [1,u,v,p]*[s;cd;cdA+cdB]||^2 with
     *  x reference, cd,A,B unrolled parameters.
     * 
     *  We get back (for x) a vector of optimal
     *    [sx;cd11;cd12; cd11*A + cd12*B]
     *  Now we can pull out sx, cd11 and cd12 from the beginning of this vector,
     *  and call the rest of the vector [cd11*A] + [cd12*B];
     *  similarly for the y fit, we get back a vector of optimal
     *    [sy;cd21;cd22; cd21*A + cd22*B]
     *  once we have all those we can figure out A and B as follows
     *                   -1
     *    A' = [cd11 cd12]    *  [cd11*A' + cd12*B']
     *    B'   [cd21 cd22]       [cd21*A' + cd22*B']
     * 
     *  which recovers the A and B's.
     *
     */

    /*
     *  Dustin's interpretation of the above:
     *  We want to solve:
     * 
     *     min || b[M-by-1] - A[M-by-N] x[N-by-1] ||_2
     * 
     *  M = the number of correspondences.
     *  N = the number of SIP terms.
     *
     * And we want an overdetermined system, so M >= N.
     * 
     *           [ 1  u_1   v_1  u_1^2  u_1 v_1  v_1^2  ... ]
     *    mA  =  [ 1  u_2   v_2  u_2^2  u_2 v_2  v_2^2  ... ]
     *           [           ......                         ]
     *
     * Where (u_i, v_i) are *undistorted* pixel positions minus CRPIX.
     *
     *  The answers we want are:
     *
     *         [ sx                  ]
     *    x1 = [ cd11                ]
     *         [ cd12                ]
     *         [      (A)        (B) ]
     *         [ cd11*(A) + cd12*(B) ]
     *         [      (A)        (B) ]
     *
     *         [ sy                  ]
     *    x2 = [ cd21                ]
     *         [ cd22                ]
     *         [      (A)        (B) ]
     *         [ cd21*(A) + cd22*(B) ]
     *         [      (A)        (B) ]
     *
     * And the target vectors are the intermediate world coords of the
     * reference stars, in degrees.
     *
     *         [ ix_1 ]
     *    b1 = [ ix_2 ]
     *         [ ...  ]
     *
     *         [ iy_1 ]
     *    b2 = [ iy_2 ]
     *         [ ...  ]
     *
     *
     *  (where A and B are tall vectors of SIP coefficients of order 2
     *  and above)
     *
     */

    // Fill in matrix mA:
    radecdeg2xyzarr(tanin->crval[0], tanin->crval[1], xyzcrval);
    totalweight = 0.0;
    ngood = 0;
    for (i=0; i<M; i++) {
        double x=0, y=0;
        double weight = 1.0;
        double u;
        double v;
        Unused anbool ok;

        u = fieldxy[2*i + 0] - tanin->crpix[0];
        v = fieldxy[2*i + 1] - tanin->crpix[1];

        // B contains Intermediate World Coordinates (in degrees)
        // tangent-plane projection
        ok = star_coords(starxyz + 3*i, xyzcrval, TRUE, &x, &y);
        if (!ok) {
            logverb("Skipping star that cannot be projected to tangent plane\n");
            continue;
        }

        if (weights) {
            weight = weights[i];
            assert(weight >= 0.0);
            assert(weight <= 1.0);
            totalweight += weight;
            if (weight == 0.0)
                continue;
        }

        gsl_vector_set(b1, ngood, weight * rad2deg(x));
        gsl_vector_set(b2, ngood, weight * rad2deg(y));

        /* The coefficients are stored in this order:
         *   p q
         *  (0,0) = 1     <- order 0
         *  (1,0) = u     <- order 1
         *  (0,1) = v
         *  (2,0) = u^2   <- order 2
         *  (1,1) = uv
         *  (0,2) = v^2
         *  ...
         */

        j = 0;
        for (order=0; order<=sip_order; order++) {
            for (q=0; q<=order; q++) {
                p = order - q;
                assert(j >= 0);
                assert(j < N);
                assert(p >= 0);
                assert(q >= 0);
                assert(p + q <= sip_order);
                gsl_matrix_set(mA, ngood, j,
                               weight * pow(u, (double)p) * pow(v, (double)q));
                j++;
            }
        }
        assert(j == N);

        // The shift - aka (0,0) - SIP coefficient must be 1.
        assert(gsl_matrix_get(mA, i, 0) == 1.0 * weight);
        assert(fabs(gsl_matrix_get(mA, i, 1) - u * weight) < 1e-12);
        assert(fabs(gsl_matrix_get(mA, i, 2) - v * weight) < 1e-12);

        ngood++;
    }

    if (ngood == 0) {
        ERROR("No stars projected within the image\n");
        return -1;
    }

    if (weights)
        logverb("Total weight: %g\n", totalweight);

    if (ngood < M) {
        _gsl_vector_view sub_b1 = gsl_vector_subvector(b1, 0, ngood);
        _gsl_vector_view sub_b2 = gsl_vector_subvector(b2, 0, ngood);
        _gsl_matrix_view sub_mA = gsl_matrix_submatrix(mA, 0, 0, ngood, N);

        rtn = gslutils_solve_leastsquares_v(&(sub_mA.matrix), 2,
                                            &(sub_b1.vector), &x1, NULL,
                                            &(sub_b2.vector), &x2, NULL);

    } else {
        // Solve the equation.
        rtn = gslutils_solve_leastsquares_v(mA, 2, b1, &x1, NULL, b2, &x2, NULL);
    }
    if (rtn) {
        ERROR("Failed to solve SIP matrix equation!");
        return -1;
    }

    // Row 0 of X are the shift (p=0, q=0) terms.
    // Row 1 of X are the terms that multiply "u".
    // Row 2 of X are the terms that multiply "v".

    if (doshift) {
        // Grab CD.
        sipout->wcstan.cd[0][0] = gsl_vector_get(x1, 1);
        sipout->wcstan.cd[0][1] = gsl_vector_get(x1, 2);
        sipout->wcstan.cd[1][0] = gsl_vector_get(x2, 1);
        sipout->wcstan.cd[1][1] = gsl_vector_get(x2, 2);

        // Compute inv(CD)
        i = invert_2by2_arr((const double*)(sipout->wcstan.cd),
                            (double*)cdinv);
        assert(i == 0);

        // Grab the shift.
        sx = gsl_vector_get(x1, 0);
        sy = gsl_vector_get(x2, 0);

    } else {
        // Compute inv(CD)
        i = invert_2by2_arr((const double*)(sipout->wcstan.cd),
                            (double*)cdinv);
        assert(i == 0);
    }

    // Extract the SIP coefficients.
    //  (this includes the 0 and 1 order terms, which we later overwrite)
    j = 0;
    for (order=0; order<=sip_order; order++) {
        for (q=0; q<=order; q++) {
            p = order - q;
            assert(j >= 0);
            assert(j < N);
            assert(p >= 0);
            assert(q >= 0);
            assert(p + q <= sip_order);

            sipout->a[p][q] =
                cdinv[0][0] * gsl_vector_get(x1, j) +
                cdinv[0][1] * gsl_vector_get(x2, j);

            sipout->b[p][q] =
                cdinv[1][0] * gsl_vector_get(x1, j) +
                cdinv[1][1] * gsl_vector_get(x2, j);
            j++;
        }
    }
    assert(j == N);

    if (doshift) {
        // We have already dealt with the shift and linear terms, so zero them out
        // in the SIP coefficient matrix.
        sipout->a[0][0] = 0.0;
        sipout->a[0][1] = 0.0;
        sipout->a[1][0] = 0.0;
        sipout->b[0][0] = 0.0;
        sipout->b[0][1] = 0.0;
        sipout->b[1][0] = 0.0;
    }

    sip_compute_inverse_polynomials(sipout, 0, 0, 0, 0, 0, 0);

    if (doshift) {
        sU =
            cdinv[0][0] * sx +
            cdinv[0][1] * sy;
        sV =
            cdinv[1][0] * sx +
            cdinv[1][1] * sy;
        logverb("Applying shift of sx,sy = %g,%g deg (%g,%g pix) to CRVAL and CD.\n",
                sx, sy, sU, sV);

        sip_calc_inv_distortion(sipout, sU, sV, &su, &sv);

        debug("sx = %g, sy = %g\n", sx, sy);
        debug("sU = %g, sV = %g\n", sU, sV);
        debug("su = %g, sv = %g\n", su, sv);

        wcs_shift(&(sipout->wcstan), -su, -sv);
    }

    if (r1)
        gsl_vector_free(r1);
    if (r2)
        gsl_vector_free(r2);

    gsl_matrix_free(mA);
    gsl_vector_free(b1);
    gsl_vector_free(b2);
    gsl_vector_free(x1);
    gsl_vector_free(x2);

    return 0;
}





int fit_sip_coefficients(const double* starxyz,
                         const double* fieldxy,
                         const double* weights,
                         int M,
                         const tan_t* tanin1,
                         int sip_order,
                         int inv_order,
                         sip_t* sipout) {
    int sip_coeffs;
    int N;
    int i, j, p, q, order;
    double totalweight;
    int rtn;
    gsl_matrix *mA;
    gsl_vector *b1, *b2, *x1, *x2;
    gsl_vector *r1=NULL, *r2=NULL;
    tan_t tanin2;
    int ngood;
    const tan_t* tanin = &tanin2;
    // We need at least the linear terms to compute CD.
    if (sip_order < 1)
        sip_order = 1;

    // convenience: allow the user to call like:
    //    fit_sip_wcs(... &(sipout.wcstan), ..., sipout);
    memcpy(&tanin2, tanin1, sizeof(tan_t));

    memset(sipout, 0, sizeof(sip_t));
    memcpy(&(sipout->wcstan), tanin, sizeof(tan_t));
    sipout->a_order  = sipout->b_order  = sip_order;
    sipout->ap_order = sipout->bp_order = inv_order;

    // The SIP coefficients form an (order x order) upper triangular
    // matrix
    sip_coeffs = (sip_order + 1) * (sip_order + 2) / 2;
    N = sip_coeffs;

    if (M < N) {
        ERROR("Too few correspondences for the SIP order specified (%i < %i)\n", M, N);
        return -1;
    }

    mA = gsl_matrix_alloc(M, N);
    b1 = gsl_vector_alloc(M);
    b2 = gsl_vector_alloc(M);
    assert(mA);
    assert(b1);
    assert(b2);

    /**
     * We're going to fit for the "forward" SIP coefficients
     * A, B.
     *
     * SIP converts (x,y) -> distort with A,B -> (x',y') -> TAN -> RA,Dec
     *
     * We are going to hold TAN fixed here, so first convert RA,Dec
     * to (x',y'), and then compute A,B terms.
     *
     * Set up the matrix equation
     *   [ SIP polynomial terms ] [ SIP coeffs ] - [x'] ~ 0
     *
     * Where SIP polynomial terms are powers of x.
     *
     * Though rather than x, x' we'll work in CRPIX-relative units,
     * since that's what SIP does.
     */

    // Fill in matrix mA:
    totalweight = 0.0;
    ngood = 0;
    for (i=0; i<M; i++) {
        double xprime, yprime;
        double x, y;
        double weight = 1.0;
        anbool ok;

        ok = tan_xyzarr2pixelxy(tanin, starxyz + 3*i, &xprime, &yprime);
        if (!ok)
            continue;

        xprime -= tanin->crpix[0];
        yprime -= tanin->crpix[1];

        x = fieldxy[2*i  ] - tanin->crpix[0];
        y = fieldxy[2*i+1] - tanin->crpix[1];

        if (weights) {
            weight = weights[i];
            assert(weight >= 0.0);
            assert(weight <= 1.0);
            totalweight += weight;
            if (weight == 0.0)
                continue;
        }

        /// AHA!, since SIP computes an "fuv","guv" to ADD to
        /// x,y to get x',y', b is the DIFFERENCE!
        gsl_vector_set(b1, ngood, weight * (xprime - x));
        gsl_vector_set(b2, ngood, weight * (yprime - y));

        /* The coefficients are stored in this order:
         *   p q
         *  (0,0) = 1     <- order 0
         *  (1,0) = u     <- order 1
         *  (0,1) = v
         *  (2,0) = u^2   <- order 2
         *  (1,1) = uv
         *  (0,2) = v^2
         *  ...
         */

        j = 0;
        for (order=0; order<=sip_order; order++) {
            for (q=0; q<=order; q++) {
                p = order - q;
                assert(j >= 0);
                assert(j < N);
                assert(p >= 0);
                assert(q >= 0);
                assert(p + q <= sip_order);
                gsl_matrix_set(mA, ngood, j,
                               weight * pow(x, (double)p) * pow(y, (double)q));
                j++;
            }
        }
        assert(j == N);

        ngood++;
    }

    if (ngood == 0) {
        ERROR("No stars projected within the image\n");
        return -1;
    }
    if (weights)
        logverb("Total weight: %g\n", totalweight);

    if (ngood < M) {
        _gsl_vector_view sub_b1 = gsl_vector_subvector(b1, 0, ngood);
        _gsl_vector_view sub_b2 = gsl_vector_subvector(b2, 0, ngood);
        _gsl_matrix_view sub_mA = gsl_matrix_submatrix(mA, 0, 0, ngood, N);

        rtn = gslutils_solve_leastsquares_v(&(sub_mA.matrix), 2,
                                            &(sub_b1.vector), &x1, NULL,
                                            &(sub_b2.vector), &x2, NULL);

    } else {
        // Solve the equation.
        rtn = gslutils_solve_leastsquares_v(mA, 2, b1, &x1, NULL, b2, &x2, NULL);
    }
    if (rtn) {
        ERROR("Failed to solve SIP matrix equation!");
        return -1;
    }

    // Extract the SIP coefficients.
    j = 0;
    for (order=0; order<=sip_order; order++) {
        for (q=0; q<=order; q++) {
            p = order - q;
            assert(j >= 0);
            assert(j < N);
            assert(p >= 0);
            assert(q >= 0);
            assert(p + q <= sip_order);
            sipout->a[p][q] = gsl_vector_get(x1, j);
            sipout->b[p][q] = gsl_vector_get(x2, j);
            j++;
        }
    }
    assert(j == N);

    if (r1)
        gsl_vector_free(r1);
    if (r2)
        gsl_vector_free(r2);

    gsl_matrix_free(mA);
    gsl_vector_free(b1);
    gsl_vector_free(b2);
    gsl_vector_free(x1);
    gsl_vector_free(x2);

    return 0;
}


// Given a pixel offset (shift in image plane), adjust the WCS
// CRVAL to the position given by CRPIX + offset.
// Why not just
// sip_pixelxy2radec(wcs, crpix0 +- xs, crpix1 +- ys,
//                   wcs->wcstan.crval+0, wcs->wcstan.crval+1);
// The answer is that "North" changes when you move the reference point.
void wcs_shift(tan_t* wcs, double xs, double ys) {
    // UNITS: xs/ys in pixels
    // crvals in degrees, nx/nyref and theta in degrees
    double crpix0, crpix1, crval0;
    double theta, sintheta, costheta;
    double newcrval0, newcrval1;
    double newcd00,newcd01,newcd10,newcd11;
    // Save old vals
    crpix0 = wcs->crpix[0];
    crpix1 = wcs->crpix[1];
    crval0 = wcs->crval[0];

    // compute the desired projection of the new tangent point by
    // shifting the projection of the current tangent point
    wcs->crpix[0] += xs;
    wcs->crpix[1] += ys;

    // now reproject the old crpix[xy] into shifted wcs
    tan_pixelxy2radec(wcs, crpix0, crpix1, &newcrval0, &newcrval1);

    // Restore crpix
    wcs->crpix[0] = crpix0;
    wcs->crpix[1] = crpix1;

    // RA,DEC coords of new tangent point
    wcs->crval[0] = newcrval0;
    wcs->crval[1] = newcrval1;
    theta = -deg2rad(newcrval0 - crval0);
    // deltaRA = new minus old RA;
    theta *= sin(deg2rad(newcrval1));
    // multiply by the sin of the NEW Dec; at equator this correctly
    // evals to zero
    sintheta = sin(theta);
    costheta = cos(theta);

    // Fix the CD matrix since "northwards" has changed due to moving RA
    newcd00 = costheta * wcs->cd[0][0] - sintheta * wcs->cd[0][1];
    newcd01 = sintheta * wcs->cd[0][0] + costheta * wcs->cd[0][1];
    newcd10 = costheta * wcs->cd[1][0] - sintheta * wcs->cd[1][1];
    newcd11 = sintheta * wcs->cd[1][0] + costheta * wcs->cd[1][1];
    wcs->cd[0][0] = newcd00;
    wcs->cd[0][1] = newcd01;
    wcs->cd[1][0] = newcd10;
    wcs->cd[1][1] = newcd11;
}



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

