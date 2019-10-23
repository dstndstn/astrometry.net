/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef SIMPLEXY2_H
#define SIMPLEXY2_H

#include "astrometry/an-bool.h"

#define SIMPLEXY_DEFAULT_DPSF        1.0
#define SIMPLEXY_DEFAULT_PLIM        8.0
#define SIMPLEXY_DEFAULT_DLIM        1.0
#define SIMPLEXY_DEFAULT_SADDLE      5.0
#define SIMPLEXY_DEFAULT_MAXPER     1000
#define SIMPLEXY_DEFAULT_MAXSIZE    2000
#define SIMPLEXY_DEFAULT_HALFBOX     100
#define SIMPLEXY_DEFAULT_MAXNPEAKS 100000

#define SIMPLEXY_U8_DEFAULT_PLIM     4.0
#define SIMPLEXY_U8_DEFAULT_SADDLE   2.0

struct simplexy_t {
    /******
     Inputs
     ******/
    float *image;
    unsigned char* image_u8;
    int nx;
    int ny;
    /* gaussian psf width (sigma, not FWHM) */
    float dpsf;
    /* significance to keep */
    float plim;
    /* closest two peaks can be */
    float dlim;
    /* saddle difference (in sig) */
    float saddle;
    /* maximum number of peaks per object */
    int maxper;
    /* maximum number of peaks total */
    int maxnpeaks;
    /* maximum size for extended objects */
    int maxsize;
    /* size for sliding sky estimation box */
    int halfbox;

    // (boolean) don't do background subtraction.
    int nobgsub;

    // global background.
    float globalbg;

    // (boolean) invert the image before processing (for black-on-white images)
    int invert;

    // If set to non-zero, the given sigma value will be used;
    // otherwise a value will be estimated.
    float sigma;

    /******
     Outputs
     ******/
    float *x;
    float *y;
    float *flux;
    float *background;
    int npeaks;

    // Lanczos-interpolated flux and backgrounds;
    // measured if Lorder > 0.
    int Lorder;
    float* fluxL;
    float* backgroundL;

    /***
     Debug
     ***/
    // The filename for saving the background-subtracted FITS image.
    const char* bgimgfn;
    const char* maskimgfn;
    const char* blobimgfn;
    const char* bgsubimgfn;
    const char* smoothimgfn;
};
typedef struct simplexy_t simplexy_t;

void simplexy_set_defaults(simplexy_t* s);

// Really this is for limited-dynamic-range images, not u8 as such...
void simplexy_set_u8_defaults(simplexy_t* i);

// Set default values for any fields that are zero.
void simplexy_fill_in_defaults(simplexy_t* s);
void simplexy_fill_in_defaults_u8(simplexy_t* s);

int simplexy_run(simplexy_t* s);

void simplexy_free_contents(simplexy_t* s);

void simplexy_clean_cache();

#endif
