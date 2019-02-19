/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#include "os-features.h"
#include "sip-utils.h"
#include "gslutils.h"
#include "starutil.h"
#include "mathutil.h"
#include "errors.h"
#include "log.h"

double wcs_pixel_center_for_size(double size) {
    return 0.5 + 0.5 * size;
}

void tan_rotate(const tan_t* tanin, tan_t* tanout, double angle) {
    double s,c;
    double newcd[4];
    memmove(tanout, tanin, sizeof(tan_t));
    s = sin(deg2rad(angle));
    c = cos(deg2rad(angle));
    newcd[0] = c*tanin->cd[0][0] + s*tanin->cd[1][0];
    newcd[1] = c*tanin->cd[0][1] + s*tanin->cd[1][1];
    newcd[2] = -s*tanin->cd[0][0] + c*tanin->cd[1][0];
    newcd[3] = -s*tanin->cd[0][1] + c*tanin->cd[1][1];
    tanout->cd[0][0] = newcd[0];
    tanout->cd[0][1] = newcd[1];
    tanout->cd[1][0] = newcd[2];
    tanout->cd[1][1] = newcd[3];
}

int sip_ensure_inverse_polynomials(sip_t* sip) {
    if ((sip->a_order == 0 && sip->b_order == 0) ||
        (sip->ap_order > 0  && sip->bp_order > 0)) {
        return 0;
    }
    sip->ap_order = sip->bp_order = MAX(sip->a_order, sip->b_order) + 1;
    return sip_compute_inverse_polynomials(sip, 0, 0, 0, 0, 0, 0);
}

int sip_compute_inverse_polynomials(sip_t* sip, int NX, int NY,
                                    double xlo, double xhi,
                                    double ylo, double yhi) {
    int inv_sip_order;
    int M, N;
    int i, j, p, q, gu, gv;
    double maxu, maxv, minu, minv;
    double u, v, U, V;
    gsl_matrix *mA;
    gsl_vector *b1, *b2, *x1, *x2;
    tan_t* tan;

    assert(sip->a_order == sip->b_order);
    assert(sip->ap_order == sip->bp_order);
    tan = &(sip->wcstan);

    logverb("sip_compute-inverse_polynomials: A %i, AP %i\n",
            sip->a_order, sip->ap_order);

    /*
     basic idea: lay down a grid in image, for each gridpoint, push
     through the polynomial to get yourself into warped image
     coordinate (but not yet lifted onto the sky).  Then, using the
     set of warped gridpoints as inputs, fit back to their original
     grid locations as targets.
     */
    inv_sip_order = sip->ap_order;

    // Number of grid points to use:
    if (NX == 0)
        NX = 10 * (inv_sip_order + 1);
    if (NY == 0)
        NY = 10 * (inv_sip_order + 1);
    if (xhi == 0)
        xhi = tan->imagew;
    if (yhi == 0)
        yhi = tan->imageh;

    logverb("NX,NY %i,%i, x range [%f, %f], y range [%f, %f]\n",
            NX,NY, xlo, xhi, ylo, yhi);

    // Number of coefficients to solve for:
    // We only compute the upper triangle polynomial terms
    N = (inv_sip_order + 1) * (inv_sip_order + 2) / 2;

    // Number of samples to fit.
    M = NX * NY;

    mA = gsl_matrix_alloc(M, N);
    b1 = gsl_vector_alloc(M);
    b2 = gsl_vector_alloc(M);
    assert(mA);
    assert(b1);
    assert(b2);

    /*
     *  Rearranging formula (4), (5), and (6) from the SIP paper gives the
     *  following equations:
     * 
     *    +----------------------- Linear pixel coordinates in PIXELS
     *    |                        before SIP correction
     *    |                   +--- Intermediate world coordinates in DEGREES
     *    |                   |
     *    v                   v
     *                   -1
     *    U = [CD11 CD12]   * x
     *    V   [CD21 CD22]     y
     * 
     *    +---------------- PIXEL distortion delta from telescope to
     *    |                 linear coordinates
     *    |    +----------- Linear PIXEL coordinates before SIP correction
     *    |    |       +--- Polynomial U,V terms in powers of PIXELS
     *    v    v       v
     * 
     *    -f(u1,v1) =  p11 p12 p13 p14 p15 ... * ap1
     *    -f(u2,v2) =  p21 p22 p23 p24 p25 ...   ap2
     *    ...
     * 
     *    -g(u1,v1) =  p11 p12 p13 p14 p15 ... * bp1
     *    -g(u2,v2) =  p21 p22 p23 p24 p25 ...   bp2
     *    ...
     * 
     *  which recovers the A and B's.
     */

    minu = xlo - tan->crpix[0];
    maxu = xhi - tan->crpix[0];
    minv = ylo - tan->crpix[1];
    maxv = yhi - tan->crpix[1];
	
    // Sample grid locations.
    i = 0;
    for (gu=0; gu<NX; gu++) {
        for (gv=0; gv<NY; gv++) {
            double fuv, guv;
            // Calculate grid position in original image pixels
            u = (gu * (maxu - minu) / (NX-1)) + minu;
            v = (gv * (maxv - minv) / (NY-1)) + minv;
            // compute U=u+f(u,v) and V=v+g(u,v)
            sip_calc_distortion(sip, u, v, &U, &V);
            fuv = U - u;
            guv = V - v;
            // Polynomial terms...
            j = 0;
            for (p = 0; p <= inv_sip_order; p++)
                for (q = 0; q <= inv_sip_order; q++) {
                    if (p + q > inv_sip_order)
                        continue;
                    assert(j < N);
                    gsl_matrix_set(mA, i, j,
                                   pow(U, (double)p) * pow(V, (double)q));
                    j++;
                }
            assert(j == N);
            gsl_vector_set(b1, i, -fuv);
            gsl_vector_set(b2, i, -guv);
            i++;
        }
    }
    assert(i == M);

    // Solve the linear equation.
    if (gslutils_solve_leastsquares_v(mA, 2, b1, &x1, NULL, b2, &x2, NULL)) {
        ERROR("Failed to solve SIP inverse matrix equation!");
        return -1;
    }

    // Extract the coefficients
    j = 0;
    for (p = 0; p <= inv_sip_order; p++)
        for (q = 0; q <= inv_sip_order; q++) {
            if ((p + q > inv_sip_order))
                continue;
            assert(j < N);
            sip->ap[p][q] = gsl_vector_get(x1, j);
            sip->bp[p][q] = gsl_vector_get(x2, j);
            j++;
        }
    assert(j == N);

    // Check that we found values that actually invert the polynomial.
    // The error should be particularly small at the grid points.
    if (log_get_level() > LOG_VERB) {
        // rms error accumulators:
        double sumdu = 0;
        double sumdv = 0;
        int Z;
        for (gu = 0; gu < NX; gu++) {
            for (gv = 0; gv < NY; gv++) {
                double newu, newv;
                // Calculate grid position in original image pixels
                u = (gu * (maxu - minu) / (NX-1)) + minu;
                v = (gv * (maxv - minv) / (NY-1)) + minv;
                sip_calc_distortion(sip, u, v, &U, &V);
                sip_calc_inv_distortion(sip, U, V, &newu, &newv);
                sumdu += square(u - newu);
                sumdv += square(v - newv);
            }
        }
        sumdu /= (NX*NY);
        sumdv /= (NX*NY);
        debug("RMS error of inverting a distortion (at the grid points, in pixels):\n");
        debug("  du: %g\n", sqrt(sumdu));
        debug("  dv: %g\n", sqrt(sumdu));
        debug("  dist: %g\n", sqrt(sumdu + sumdv));

        sumdu = 0;
        sumdv = 0;
        Z = 1000;
        for (i=0; i<Z; i++) {
            double newu, newv;
            u = uniform_sample(minu, maxu);
            v = uniform_sample(minv, maxv);
            sip_calc_distortion(sip, u, v, &U, &V);
            sip_calc_inv_distortion(sip, U, V, &newu, &newv);
            sumdu += square(u - newu);
            sumdv += square(v - newv);
        }
        sumdu /= Z;
        sumdv /= Z;
        debug("RMS error of inverting a distortion (at random points, in pixels):\n");
        debug("  du: %g\n", sqrt(sumdu));
        debug("  dv: %g\n", sqrt(sumdu));
        debug("  dist: %g\n", sqrt(sumdu + sumdv));
    }

    gsl_matrix_free(mA);
    gsl_vector_free(b1);
    gsl_vector_free(b2);
    gsl_vector_free(x1);
    gsl_vector_free(x2);

    return 0;
}

anbool tan_pixel_is_inside_image(const tan_t* wcs, double x, double y) {
    return (x >= 1 && x <= wcs->imagew && y >= 1 && y <= wcs->imageh);
}

anbool sip_pixel_is_inside_image(const sip_t* wcs, double x, double y) {
    return tan_pixel_is_inside_image(&(wcs->wcstan), x, y);
}

anbool sip_is_inside_image(const sip_t* wcs, double ra, double dec) {
    double x,y;
    if (!sip_radec2pixelxy(wcs, ra, dec, &x, &y))
        return FALSE;
    return sip_pixel_is_inside_image(wcs, x, y);
}

anbool tan_is_inside_image(const tan_t* wcs, double ra, double dec) {
    double x,y;
    if (!tan_radec2pixelxy(wcs, ra, dec, &x, &y))
        return FALSE;
    return tan_pixel_is_inside_image(wcs, x, y);
}

int* sip_filter_stars_in_field(const sip_t* sip, const tan_t* tan,
                               const double* xyz, const double* radec,
                               int N, double** p_xy, int* inds, int* p_Ngood) {
    int i, Ngood;
    int W, H;
    double* xy = NULL;
    anbool allocd = FALSE;
	
    assert(sip || tan);
    assert(xyz || radec);
    assert(p_Ngood);

    Ngood = 0;
    if (!inds) {
        inds = malloc(N * sizeof(int));
        allocd = TRUE;
    }

    if (p_xy)
        xy = malloc(N * 2 * sizeof(double));

    if (sip) {
        W = sip->wcstan.imagew;
        H = sip->wcstan.imageh;
    } else {
        W = tan->imagew;
        H = tan->imageh;
    }

    for (i=0; i<N; i++) {
        double x, y;
        if (xyz) {
            if (sip) {
                if (!sip_xyzarr2pixelxy(sip, xyz + i*3, &x, &y))
                    continue;
            } else {
                if (!tan_xyzarr2pixelxy(tan, xyz + i*3, &x, &y))
                    continue;
            }
        } else {
            if (sip) {
                if (!sip_radec2pixelxy(sip, radec[i*2], radec[i*2+1], &x, &y))
                    continue;
            } else {
                if (!tan_radec2pixelxy(tan, radec[i*2], radec[i*2+1], &x, &y))
                    continue;
            }
        }
        // FIXME -- check half- and one-pixel FITS issues.
        if ((x < 0) || (y < 0) || (x >= W) || (y >= H))
            continue;

        inds[Ngood] = i;
        if (xy) {
            xy[Ngood * 2 + 0] = x;
            xy[Ngood * 2 + 1] = y;
        }
        Ngood++;
    }

    if (allocd)
        inds = realloc(inds, Ngood * sizeof(int));

    if (xy)
        xy = realloc(xy, Ngood * 2 * sizeof(double));
    if (p_xy)
        *p_xy = xy;

    *p_Ngood = Ngood;
	
    return inds;
}

void sip_get_radec_center(const sip_t* wcs,
                          double* p_ra, double* p_dec) {
    double px = wcs_pixel_center_for_size(wcs->wcstan.imagew);
    double py = wcs_pixel_center_for_size(wcs->wcstan.imageh);
    sip_pixelxy2radec(wcs, px, py, p_ra, p_dec);
}

void tan_get_radec_center(const tan_t* wcs,
                          double* p_ra, double* p_dec) {
    double px = wcs_pixel_center_for_size(wcs->imagew);
    double py = wcs_pixel_center_for_size(wcs->imageh);
    tan_pixelxy2radec(wcs, px, py, p_ra, p_dec);
}

double sip_get_radius_deg(const sip_t* wcs) {
    return arcsec2deg(sip_pixel_scale(wcs) * hypot(wcs->wcstan.imagew, wcs->wcstan.imageh)/2.0);
}

double tan_get_radius_deg(const tan_t* wcs) {
    return arcsec2deg(tan_pixel_scale(wcs) * hypot(wcs->imagew, wcs->imageh)/2.0);
}

void sip_get_radec_center_hms(const sip_t* wcs,
                              int* rah, int* ram, double* ras,
                              int* decsign, int* decd, int* decm, double* decs) {
    double ra, dec;
    sip_get_radec_center(wcs, &ra, &dec);
    ra2hms(ra, rah, ram, ras);
    dec2dms(dec, decsign, decd, decm, decs);
}

void sip_get_radec_center_hms_string(const sip_t* wcs,
                                     char* rastr, char* decstr) {
    double ra, dec;
    sip_get_radec_center(wcs, &ra, &dec);
    ra2hmsstring(ra, rastr);
    dec2dmsstring(dec, decstr);
}

void sip_get_field_size(const sip_t* wcs,
                        double* pw, double* ph,
                        char** units) {
    double minx = 0.5;
    double maxx = (wcs->wcstan.imagew + 0.5);
    double midx = (minx + maxx) / 2.0;
    double miny = 0.5;
    double maxy = (wcs->wcstan.imageh + 0.5);
    double midy = (miny + maxy) / 2.0;
    double ra1, dec1, ra2, dec2, ra3, dec3;
    double w, h;

    // measure width through the middle
    sip_pixelxy2radec(wcs, minx, midy, &ra1, &dec1);
    sip_pixelxy2radec(wcs, midx, midy, &ra2, &dec2);
    sip_pixelxy2radec(wcs, maxx, midy, &ra3, &dec3);
    w = arcsec_between_radecdeg(ra1, dec1, ra2, dec2) +
        arcsec_between_radecdeg(ra2, dec2, ra3, dec3);
    // measure height through the middle
    sip_pixelxy2radec(wcs, midx, miny, &ra1, &dec1);
    sip_pixelxy2radec(wcs, midx, midy, &ra2, &dec2);
    sip_pixelxy2radec(wcs, midx, maxy, &ra3, &dec3);
    h = arcsec_between_radecdeg(ra1, dec1, ra2, dec2) +
        arcsec_between_radecdeg(ra2, dec2, ra3, dec3);

    if (MIN(w, h) < 60.0) {
        *units = "arcseconds";
        *pw = w;
        *ph = h;
    } else if (MIN(w, h) < 3600.0) {
        *units = "arcminutes";
        *pw = w / 60.0;
        *ph = h / 60.0;
    } else {
        *units = "degrees";
        *pw = w / 3600.0;
        *ph = h / 3600.0;
    }
}

void sip_walk_image_boundary(const sip_t* wcs, double stepsize,
                             void (*callback)(const sip_t* wcs, double x, double y, double ra, double dec, void* token),
                             void* token) {
    int i, side;
    // Walk the perimeter of the image in steps of stepsize pixels
    double W = wcs->wcstan.imagew;
    double H = wcs->wcstan.imageh;
    {
        double Xmin = 0.5;
        double Xmax = W + 0.5;
        double Ymin = 0.5;
        double Ymax = H + 0.5;
        double offsetx[] = { Xmin, Xmax, Xmax, Xmin };
        double offsety[] = { Ymin, Ymin, Ymax, Ymax };
        double stepx[] = { +stepsize, 0, -stepsize, 0 };
        double stepy[] = { 0, +stepsize, 0, -stepsize };
        int Nsteps[] = { ceil(W/stepsize), ceil(H/stepsize), ceil(W/stepsize), ceil(H/stepsize) };

        for (side=0; side<4; side++) {
            for (i=0; i<Nsteps[side]; i++) {
                double ra, dec;
                double x, y;
                x = MIN(Xmax, MAX(Xmin, offsetx[side] + i * stepx[side]));
                y = MIN(Ymax, MAX(Ymin, offsety[side] + i * stepy[side]));
                sip_pixelxy2radec(wcs, x, y, &ra, &dec);
                callback(wcs, x, y, ra, dec, token);
            }
        }
    }
}

struct radecbounds {
    double rac, decc;
    double ramin, ramax, decmin, decmax;
};

static void radec_bounds_callback(const sip_t* wcs, double x, double y, double ra, double dec, void* token) {
    struct radecbounds* b = token;
    b->decmin = MIN(b->decmin, dec);
    b->decmax = MAX(b->decmax, dec);
    if (ra - b->rac > 180)
        // wrap-around: racenter < 180, ra has gone < 0 but been wrapped around to > 180
        ra -= 360;
    if (b->rac - ra > 180)
        // wrap-around: racenter > 180, ra has gone > 360 but wrapped around to > 0.
        ra += 360;

    b->ramin = MIN(b->ramin, ra);
    b->ramax = MAX(b->ramax, ra);
}

void sip_get_radec_bounds(const sip_t* wcs, int stepsize,
                          double* pramin, double* pramax,
                          double* pdecmin, double* pdecmax) {
    struct radecbounds b;

    sip_get_radec_center(wcs, &(b.rac), &(b.decc));
    b.ramin  = b.ramax = b.rac;
    b.decmin = b.decmax = b.decc;
    sip_walk_image_boundary(wcs, stepsize, radec_bounds_callback, &b);

    // Check for poles...
    // north pole
    if (sip_is_inside_image(wcs, 0, 90)) {
        b.ramin = 0;
        b.ramax = 360;
        b.decmax = 90;
    }
    if (sip_is_inside_image(wcs, 0, -90)) {
        b.ramin = 0;
        b.ramax = 360;
        b.decmin = -90;
    }

    if (pramin) *pramin = b.ramin;
    if (pramax) *pramax = b.ramax;
    if (pdecmin) *pdecmin = b.decmin;
    if (pdecmax) *pdecmax = b.decmax;
}

void sip_shift(const sip_t* sipin, sip_t* sipout,
               double xlo, double xhi, double ylo, double yhi) {
    memmove(sipout, sipin, sizeof(sip_t));
    tan_transform(&(sipin->wcstan), &(sipout->wcstan),
                  xlo, xhi, ylo, yhi, 1.0);
}

void tan_transform(const tan_t* tanin, tan_t* tanout,
                   double xlo, double xhi, double ylo, double yhi,
                   double scale) {
    memmove(tanout, tanin, sizeof(tan_t));
    tanout->imagew = (xhi - xlo + 1) * scale;
    tanout->imageh = (yhi - ylo + 1) * scale;
    tanout->crpix[0] = (tanout->crpix[0] - (xlo - 1)) * scale;
    tanout->crpix[1] = (tanout->crpix[1] - (ylo - 1)) * scale;
    tanout->cd[0][0] /= scale;
    tanout->cd[0][1] /= scale;
    tanout->cd[1][0] /= scale;
    tanout->cd[1][1] /= scale;
}

void tan_scale(const tan_t* tanin, tan_t* tanout,
               double scale) {
    memmove(tanout, tanin, sizeof(tan_t));
    tanout->imagew *= scale;
    tanout->imageh *= scale;

    tanout->crpix[0] = 0.5 + scale * (tanin->crpix[0] - 0.5);
    tanout->crpix[1] = 0.5 + scale * (tanin->crpix[1] - 0.5);
    tanout->cd[0][0] /= scale;
    tanout->cd[0][1] /= scale;
    tanout->cd[1][0] /= scale;
    tanout->cd[1][1] /= scale;
}

void sip_scale(const sip_t* wcsin, sip_t* wcsout,
               double scale) {
    int i, j;
    memmove(wcsout, wcsin, sizeof(sip_t));
    tan_scale(&(wcsin->wcstan), &(wcsout->wcstan), scale);
    for (i=0; i<=wcsin->a_order; i++) {
        for (j=0; j<=wcsin->a_order; j++) {
            if (i + j > wcsin->a_order)
                continue;
            wcsout->a[i][j] *= pow(scale, 1 - (i+j));
        }
    }
    for (i=0; i<=wcsin->b_order; i++) {
        for (j=0; j<=wcsin->b_order; j++) {
            if (i + j > wcsin->b_order)
                continue;
            wcsout->b[i][j] *= pow(scale, 1 - (i+j));
        }
    }
    for (i=0; i<=wcsin->ap_order; i++) {
        for (j=0; j<=wcsin->ap_order; j++) {
            if (i + j > wcsin->ap_order)
                continue;
            wcsout->ap[i][j] *= pow(scale, 1 - (i+j));
        }
    }
    for (i=0; i<=wcsin->bp_order; i++) {
        for (j=0; j<=wcsin->bp_order; j++) {
            if (i + j > wcsin->bp_order)
                continue;
            wcsout->bp[i][j] *= pow(scale, 1 - (i+j));
        }
    }
}

