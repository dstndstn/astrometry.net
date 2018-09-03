/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef AN_HD_H
#define AN_HD_H

#include "astrometry/kdtree.h"
#include "astrometry/bl.h"

/**

 This code allows easy access to the Henry Draper catalog stored in
 kd-tree format.  Get the required hd.fits file either by the easy way:

 wget http://data.astrometry.net/hd.fits

 Or the hard way (which involves downloading > 500 MB of Tycho-2
 catalog from Denmark!):

 make hd.fits   (in the astrometry.net catalog/ directory)

 (note, this way has become considerably more difficult as some URLs
 have disappeared.)

 **/


struct hd_entry {
    // J2000.0 degrees
    double ra;
    double dec;

    int hd;
};
typedef struct hd_entry hd_entry_t;

struct hd_catalog {
    char* fn;
    kdtree_t* kd;
};
typedef struct hd_catalog hd_catalog_t;

hd_catalog_t* henry_draper_open(const char* fn);

// N stars
int henry_draper_n(const hd_catalog_t* hd);

void henry_draper_close(hd_catalog_t* hd);

bl* henry_draper_get(hd_catalog_t* hd,
                     double racenter, double deccenter,
                     double radius_in_arcsec);

#endif
