/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdlib.h>
#include <math.h>

#include "coadd.h"
#include "mathutil.h"
#include "errors.h"
#include "log.h"
#include "resample.h"
#include "os-features.h"

coadd_t* coadd_new_from_wcs(anwcs_t* wcs) {
    int W,H;
    coadd_t* co;
    W = anwcs_imagew(wcs);
    H = anwcs_imagew(wcs);
    co = coadd_new(W, H);
    if (!co) {
	return NULL;
    }
    co->wcs = wcs;
    return co;
}

coadd_t* coadd_new(int W, int H) {
    coadd_t* ca = calloc(1, sizeof(coadd_t));
    ca->img = calloc((size_t)W * (size_t)H, sizeof(number));
    ca->weight = calloc((size_t)W * (size_t)H, sizeof(number));
    ca->W = W;
    ca->H = H;
    ca->resample_func = nearest_resample_f;
    return ca;
}

void coadd_set_lanczos(coadd_t* co, int Lorder) {
    lanczos_args_t* L = calloc(1, sizeof(lanczos_args_t));
    L->weighted = 0;
    L->order = Lorder;
    co->resample_token = L;
    co->resample_func = lanczos_resample_f;
}

void coadd_debug(coadd_t* co) {
    int i;
    double mn,mx;
    mn = 1e300;
    mx = -1e300;
    for (i=0; i<(co->W*co->H); i++) {
        mn = MIN(mn, co->img[i]);
        mx = MAX(mx, co->img[i]);
    }
    logmsg("Coadd img min,max %g,%g\n", mn,mx);
    mn = 1e300;
    mx = -1e300;
    for (i=0; i<(co->W*co->H); i++) {
        mn = MIN(mn, co->weight[i]);
        mx = MAX(mx, co->weight[i]);
    }
    logmsg("Weight img min,max %g,%g\n", mn,mx);
    mn = 1e300;
    mx = -1e300;
    for (i=0; i<(co->W*co->H); i++) {
        if (co->weight[i] > 0) {
            mn = MIN(mn, co->img[i] / co->weight[i]);
            mx = MAX(mx, co->img[i] / co->weight[i]);
        }
    }
    logmsg("Img/Weight min,max %g,%g\n", mn,mx);
}

typedef struct {
    double xlo, xhi, ylo, yhi;
    anwcs_t* wcs;
} check_bounds_t;

static void check_bounds(const anwcs_t* wcs, double x, double y, double ra, double dec,
                         void* token) {
    // project the RA,Dec into the co-add
    check_bounds_t* cb = (check_bounds_t*)token;
    double cx, cy;
    if (anwcs_radec2pixelxy(cb->wcs, ra, dec, &cx, &cy)) {
        ERROR("Failed to project RA,Dec (%g,%g) into coadd WCS\n", ra, dec);
        return;
    }
    cx -= 1;
    cy -= 1;
    cb->xlo = MIN(cb->xlo, cx);
    cb->xhi = MAX(cb->xhi, cx);
    cb->ylo = MIN(cb->ylo, cy);
    cb->yhi = MAX(cb->yhi, cy);
}


int coadd_add_image(coadd_t* ca, const number* img,
                    const number* weightimg,
                    number weight, const anwcs_t* wcs) {
    int W, H;
    int i, j;
    int xlo,xhi,ylo,yhi;
    check_bounds_t cb;

    W = anwcs_imagew(wcs);
    H = anwcs_imageh(wcs);

    // if check_bounds:
    cb.xlo = W;
    cb.xhi = 0;
    cb.ylo = H;
    cb.yhi = 0;
    cb.wcs = ca->wcs;
    anwcs_walk_image_boundary(wcs, 50, check_bounds, &cb);
    xlo = MAX(0,     floor(cb.xlo));
    xhi = MIN(ca->W,  ceil(cb.xhi)+1);
    ylo = MAX(0,     floor(cb.ylo));
    yhi = MIN(ca->H,  ceil(cb.yhi)+1);
    logmsg("Image projects to output image region: [%i,%i), [%i,%i)\n", xlo, xhi, ylo, yhi);

    for (i=ylo; i<yhi; i++) {
        for (j=xlo; j<xhi; j++) {
            double ra, dec;
            double px, py;
            double wt;
            double val;

            // +1 for FITS
            if (anwcs_pixelxy2radec(ca->wcs, j+1, i+1, &ra, &dec)) {
                ERROR("Failed to project pixel (%i,%i) through output WCS\n", j, i);
                continue;
            }
            if (anwcs_radec2pixelxy(wcs, ra, dec, &px, &py)) {
                ERROR("Failed to project pixel (%i,%i) through input WCS\n", j, i);
                continue;
            }
            // -1 for FITS
            px -= 1;
            py -= 1;

            if (px < 0 || px >= W)
                continue;
            if (py < 0 || py >= H)
                continue;

            val = ca->resample_func(px, py, img, weightimg, W, H, &wt,
                                    ca->resample_token);
            ca->img[i*ca->W + j] += val * weight;
            ca->weight[i*ca->W + j] += wt * weight;
        }
        logverb("Row %i of %i\n", i+1, ca->H);
    }
    return 0;
}


number* coadd_get_snapshot(coadd_t* co, number* outimg,
                           number badpix) {
    int i;
    if (!outimg)
	outimg = calloc((size_t)co->W * (size_t)co->H, sizeof(number));

    for (i=0; i<(co->W * co->H); i++) {
        if (co->weight[i] == 0)
            outimg[i] = badpix;
        else
            outimg[i] = co->img[i] / co->weight[i];
    }
    return outimg;
}


// divide "img" by "weight"; set img=badpix where weight=0.
void coadd_divide_by_weight(coadd_t* ca, number badpix) {
    int i;
    for (i=0; i<(ca->W * ca->H); i++) {
        if (ca->weight[i] == 0)
            ca->img[i] = badpix;
        else
            ca->img[i] /= ca->weight[i];
    }
}

void coadd_free(coadd_t* ca) {
    free(ca->img);
    free(ca->weight);
    free(ca);
}

number* coadd_create_weight_image_from_range(const number* img, int W, int H,
                                             number lowval, number highval) {
    int i;
    number* weight = malloc((size_t)W*(size_t)H*sizeof(number));
    for (i=0; i<(W*H); i++) {
        if (img[i] <= lowval)
            weight[i] = 0;
        else if (img[i] >= highval)
            weight[i] = 0;
        else
            weight[i] = 1;
    }
    return weight;
}

void coadd_weight_image_mask_value(const number* img, int W, int H,
                                   number* weight, number badval) {
    int i;
    for (i=0; i<(W*H); i++) {
        if (img[i] == badval) {
            weight[i] = 0;
        }
    }
}
