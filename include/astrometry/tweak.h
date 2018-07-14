/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef _TWEAK_INTERNAL_H
#define _TWEAK_INTERNAL_H

#include "astrometry/an-bool.h"
#include "astrometry/kdtree.h"
#include "astrometry/bl.h"
#include "astrometry/sip.h"
#include "astrometry/starutil.h"
#include "astrometry/starxy.h"

// These flags represent the work already done on a tweak problem
enum tweak_flags {
    TWEAK_HAS_SIP                   = 0x1,
    TWEAK_HAS_IMAGE_XY              = 0x2,
    TWEAK_HAS_IMAGE_XYZ             = 0x4,
    TWEAK_HAS_IMAGE_AD              = 0x8,
    TWEAK_HAS_REF_XY                = 0x10, 
    TWEAK_HAS_REF_XYZ               = 0x20, 
    TWEAK_HAS_REF_AD                = 0x40, 
    TWEAK_HAS_CORRESPONDENCES       = 0x100,
    TWEAK_HAS_COARSLY_SHIFTED       = 0x800,
    TWEAK_HAS_FINELY_SHIFTED        = 0x1000,
    TWEAK_HAS_REALLY_FINELY_SHIFTED = 0x2000,
    TWEAK_HAS_LINEAR_CD             = 0x4000,
};

typedef struct tweak_s {
    sip_t* sip;
    // bitfield of tweak_flags
    unsigned int state; 

    // Sources in the image
    int n;
    // pixel x,y
    double *x;
    double *y;
    // CACHED:
    // RA,Dec
    double *a;
    double *d;
    // vector on the unit sphere
    double *xyz;

    // Sources in the catalog
    int n_ref;
    // RA,Dec
    double *a_ref;
    double *d_ref;
    // unit vector on the sphere
    double *xyz_ref;
    // CACHED:
    // pixel
    double *x_ref;
    double *y_ref;

    // Correspondences
    il* image;
    il* ref;
    dl* dist2;
    dl* weight;

    // Size of Hough space for shift
    double mindx, mindy, maxdx, maxdy;

    // Size of last run shift operation
    double xs, ys;

    // Trees used for finding correspondences
    kdtree_t* kd_image;
    kdtree_t* kd_ref;

    // star jitter, in arcseconds.
    double jitter;

    // (computed from jitter); star jitter in distance-squared on the unit sphere.
    double jitterd2;

    // Weighted or unweighted fit?
    anbool weighted_fit;

    // push SIP shift term onto CRPIX, or CRVAL?
    // traditional behavior is CRPIX; ie push_crval = FALSE.
    //anbool push_crval;


} tweak_t;

tweak_t* tweak_new();
void tweak_init(tweak_t*);
void tweak_push_wcs_tan(tweak_t* t, const tan_t* wcs);
void tweak_push_ref_xyz(tweak_t* t, const double* xyz, int n);
void tweak_push_ref_ad(tweak_t* t, const double* a, const double *d, int n);
void tweak_push_ref_ad_array(tweak_t* t, const double* ad, int n);
void tweak_push_image_xy(tweak_t* t, const starxy_t* xy);
void tweak_push_correspondence_indices(tweak_t* t, il* image, il* ref, dl* distsq, dl* weight);

unsigned int tweak_advance_to(tweak_t* t, unsigned int flag);
void tweak_clear(tweak_t* t);
void tweak_dump_ascii(tweak_t* t);
void tweak_skip_shift(tweak_t* t);
char* tweak_get_state_string(const tweak_t* t);
void tweak_go_to(tweak_t* t, unsigned int flag);
void tweak_clear_correspondences(tweak_t* t);
void tweak_clear_on_sip_change(tweak_t* t);
void tweak_clear_image_ad(tweak_t* t);
void tweak_clear_ref_xy(tweak_t* t);
void tweak_clear_image_xyz(tweak_t* t);
void tweak_free(tweak_t* t);

void tweak_iterate_to_order(tweak_t* t, int maxorder, int iterations);

sip_t* tweak_just_do_it(const tan_t* wcs, const starxy_t* imagexy,
                        const double* starxyz,
                        const double* star_ra, const double* star_dec,
                        const double* star_radec,
                        int nstars, double jitter_arcsec,
                        int order, int inverse_order, int iterations,
                        anbool weighted, anbool skip_shift);


// TEST
void tchebyshev_tweak(tweak_t* t, int W, int H);

#endif
