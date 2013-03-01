/*
 This file is part of the Astrometry.net suite.
 Copyright 2008 Dustin Lang.

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

#ifndef SCAMP_CATALOG_H
#define SCAMP_CATALOG_H

#include "fitstable.h"
#include "qfits.h"
#include "an-bool.h"

struct scamp_catalog {
    fitstable_t* table;
    anbool ref;
};
typedef struct scamp_catalog scamp_cat_t;

// These definitions are imported from SExtractor 2.5.0, and should
// be OR'd together to form the "flags" field.

// "The object has neighbours, bright and close enough to significantly bias
// the photometry; or bad pixels.
#define		SCAMP_FLAG_CROWDED	0x0001
// The object was blended with another one.
#define		SCAMP_FLAG_MERGED	0x0002
// At least one pixel is saturated
#define		SCAMP_FLAG_SATUR	0x0004
// The object is too close to an image boundary
#define		SCAMP_FLAG_TRUNC	0x0008
// Aperture data incorrect
#define		SCAMP_FLAG_APERT_PB	0x0010
// Isophotal data incorrect
#define		SCAMP_FLAG_ISO_PB	0x0020
// Memory overflow during deblending (!)
#define		SCAMP_FLAG_DOVERFLOW	0x0040
// Memory overflow during extraction (!)
#define		SCAMP_FLAG_OVERFLOW	0x0080

struct scamp_catalog_object {
    double x;
    double y;
    double err_a;
    double err_b;
    double err_theta;
    double flux;
    double err_flux;
    int16_t flags;
};
typedef struct scamp_catalog_object scamp_obj_t;

struct scamp_reference_object {
    double ra;
    double dec;
    double err_a;
    double err_b;
    //double err_theta;
    double mag;
    double err_mag;
    //int16_t flags;
};
typedef struct scamp_reference_object scamp_ref_t;

scamp_cat_t* scamp_catalog_open_for_writing(const char* filename,
                                            anbool reference);

int scamp_catalog_write_field_header(scamp_cat_t* scamp, const qfits_header* hdr);

int scamp_catalog_write_object(scamp_cat_t* scamp, const scamp_obj_t* obj);

int scamp_catalog_write_reference(scamp_cat_t* scamp, const scamp_ref_t* ref);

int scamp_catalog_close(scamp_cat_t* scamp);

#endif
