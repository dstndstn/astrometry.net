/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include "astrometry/anwcs.h"
#include "astrometry/resample.h"

typedef float number;

/*
 typedef bool (*isbadpix)(const number* img, const number* weightimg, int ix, int iy, void* token)
 badpixfunc_t;
 */

typedef struct {
    number* img;
    number* weight;
    int W, H;
    anwcs_t* wcs;

    double (*resample_func)(double px, double py,
                            const number* img, const number* weightimg,
                            int W, int H,
                            double* wt,
                            //badpixfunc_t badpix,
                            //void* isbadpix_token,
                            void* resample_token);
    void* resample_token;
} coadd_t;

coadd_t* coadd_new(int W, int H);

coadd_t* coadd_new_from_wcs(anwcs_t* wcs);

void coadd_set_lanczos(coadd_t* co, int Lorder);

int coadd_add_image(coadd_t* c, const number* img, const number* weightimg,
                    number weight, const anwcs_t* wcs);
//, badpixfunc_t badpix, void* badpix_token);

// divide "img" by "weight"; set img=badpix where weight=0.
void coadd_divide_by_weight(coadd_t* c, number badpix);

number* coadd_get_snapshot(coadd_t* c, number* outimg, number badpix);

void coadd_free(coadd_t* c);

void coadd_debug(coadd_t* co);

number* coadd_create_weight_image_from_range(const number* img, int W, int H,
                                             number lowval, number highval);

void coadd_weight_image_mask_value(const number* img, int W, int H,
                                   number* weight, number badval);
