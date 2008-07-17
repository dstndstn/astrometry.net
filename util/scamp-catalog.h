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

struct scamp_catalog {
    fitstable_t* table;
};
typedef struct scamp_catalog scamp_cat_t;

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

scamp_cat_t* scamp_catalog_open_for_writing(const char* filename);

int scamp_catalog_write_field_header(scamp_cat_t* scamp, const qfits_header* hdr);

int scamp_catalog_write_object(scamp_cat_t* scamp, const scamp_obj_t* obj);

int scamp_catalog_close(scamp_cat_t* scamp);

#endif
