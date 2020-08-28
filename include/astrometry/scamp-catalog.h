/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef SCAMP_CATALOG_H
#define SCAMP_CATALOG_H

#include "astrometry/fitstable.h"
#include "astrometry/qfits_header.h"
#include "astrometry/an-bool.h"

struct scamp_catalog {
    fitstable_t* table;
    anbool ref;
};
typedef struct scamp_catalog scamp_cat_t;

// These definitions are imported from Source Extractor 2.5.0, and should
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
