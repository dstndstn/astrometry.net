/*
  This file is part of the Astrometry.net suite.
  Copyright 2008, 2009 Dustin Lang.

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

#ifndef AN_HD_H
#define AN_HD_H

#include "kdtree.h"
#include "bl.h"

/**

 This code allows easy access to the Henry Draper catalog stored in
 kd-tree format.  Get the required hd.fits file either by the easy way:

 wget http://trac.astrometry.net/export/13120/binary/henry-draper/hd.fits

 Or the hard way (which involves downloading > 500 MB of Tycho-2 catalog from Denmark!):

 make hd.fits   (in the astrometry.net util/ directory)

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
