/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef AUGMENT_XYLIST_H
#define AUGMENT_XYLIST_H

#include <time.h>

#include "astrometry/starxy.h"
#include "astrometry/an-bool.h"
#include "astrometry/bl.h"
#include "astrometry/an-opts.h"

#define SCALE_UNITS_DEG_WIDTH 0
#define SCALE_UNITS_ARCMIN_WIDTH 1
#define SCALE_UNITS_ARCSEC_PER_PIX 2
#define SCALE_UNITS_FOCAL_MM 3

struct augment_xylist_s {
    char* tempdir;

    int verbosity;
    anbool no_delete_temp;

    // contains ranges of depths as pairs of ints.
    il* depths;
    // contains ranges of fields as pairs of ints.
    il* fields;

    int cutobjs;

    sl* verifywcs;
    il* verifywcs_ext;

    sip_t* predistort;

    double pixel_xscale;
    
    // FITS columns copied from index to RDLS output
    sl* tagalong;
    anbool tagalong_all;

    // column to sort RDLS output by; prefix with "-" for descending order.
    char* sort_rdls;

    // input files
    char* imagefn;
    char* xylsfn;
    char* solvedinfn;

    anbool assume_fits_image;

    // output files
    char* axyfn;
    char* cancelfn;
    char* solvedfn;
    char* matchfn;
    char* rdlsfn;
    // SCAMP reference catalog
    char* scampfn;
    char* wcsfn;
    char* corrfn;
    char* keepxylsfn;
    char* pnmfn;

    time_t wcs_last_mod;

    anbool keep_fitsimg;
    char* fitsimgfn;
    int fitsimgext;
    
    // FITS extension to read image from
    int extension;

    // set during augment_xylist: is the input image or xyls FITS?
    anbool isfits;

    anbool guess_scale;
    anbool pnm;
    anbool force_ppm;

    anbool use_source_extractor;
    char* source_extractor_path;
    char* source_extractor_config;

    int W;
    int H;

    double scalelo;
    double scalehi;

    int scaleunit;

    int parity;

    float cpulimit;

    anbool tweak;
    int tweakorder;

    anbool no_removelines;
    anbool no_bg_subtraction;

    int uniformize;

    anbool invert_image;

    float image_sigma;
    float image_nsigma;

    char* xcol;
    char* ycol;
    char* sortcol;
    char* bgcol;

    // WCS reference point
    anbool set_crpix;
    anbool set_crpix_center;
    double crpix[2];

    anbool sort_ascending;
    anbool resort;

    double codetol;
    double pixelerr;

    double odds_to_tune_up;
    double odds_to_solve;
    double odds_to_bail;
    double odds_to_stoplooking;

    int downsample;

    anbool dont_augment;

    anbool verify_uniformize;
    anbool verify_dedup;

    // try to verify FITS input images?
    anbool try_verify;

    // fractions
    double quadsize_min;
    double quadsize_max;

    // for searching only within indexes that are near some estimated position.
    double ra_center;
    double dec_center;
    double search_radius;
};
typedef struct augment_xylist_s augment_xylist_t;

int parse_scale_units(const char* str);

int augment_xylist(augment_xylist_t* args,
                   const char* executable_path);

void augment_xylist_init(augment_xylist_t* args);

void augment_xylist_free_contents(augment_xylist_t* args);

void augment_xylist_print_help(FILE* fid);

void augment_xylist_add_options(bl* opts);

int augment_xylist_parse_option(char argchar, char* optarg, augment_xylist_t* axyargs);

void augment_xylist_print_special_opts(an_option_t* opt, bl* opts,
                                       int index,
                                       FILE* fid, void* extra);

#endif


