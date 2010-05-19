/*
 This file is part of the Astrometry.net suite.
 Copyright 2008-2009 Dustin Lang.

 The Astrometry.net suite is free software; you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA	 02110-1301 USA
 */

#ifndef AUGMENT_XYLIST_H
#define AUGMENT_XYLIST_H

#include <time.h>

#include "starxy.h"
#include "an-bool.h"
#include "bl.h"
#include "an-opts.h"

struct augment_xylist_s {
    char* tempdir;

    int verbosity;
    bool no_delete_temp;

    // contains ranges of depths as pairs of ints.
    il* depths;
    // contains ranges of fields as pairs of ints.
    il* fields;

	int cutobjs;

    sl* verifywcs;

	// FITS columns copied from index to RDLS output
	sl* tagalong;
	bool tagalong_all;

	// column to sort RDLS output by; prefix with "-" for descending order.
	char* sort_rdls;

    // input files
    char* imagefn;
    char* xylsfn;
    char* solvedinfn;

    // output files
    char* outfn;
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

    bool keep_fitsimg;
    char* fitsimgfn;

    // FITS extension to read image from
    int extension;

    // set during augment_xylist: is the input image or xyls FITS?
    bool isfits;

    bool guess_scale;
    bool pnm;
    bool force_ppm;

	bool use_sextractor;
	char* sextractor_path;
	char* sextractor_config;

    int W;
    int H;

    double scalelo;
    double scalehi;
    char* scaleunits;

    int parity;

    float cpulimit;

    bool tweak;
    int tweakorder;

    bool no_fits2fits;
	bool no_removelines;
	bool no_fix_sdss;
	bool no_bg_subtraction;

	int uniformize;

	bool invert_image;

	float image_sigma;

    char* xcol;
    char* ycol;
    char* sortcol;
	char* bgcol;

	// WCS reference point
	bool set_crpix;
	bool set_crpix_center;
	double crpix[2];

    bool sort_ascending;
    bool resort;

    double codetol;
    double pixelerr;

	double odds_to_solve;
	double odds_to_bail;
	double odds_to_stoplooking;

    int downsample;

    bool dont_augment;

    // try to verify FITS input images?
    bool try_verify;

    // fractions
    double quadsize_min;
    double quadsize_max;

    // for searching only within indexes that are near some estimated position.
    double ra_center;
    double dec_center;
    double search_radius;
};
typedef struct augment_xylist_s augment_xylist_t;


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


