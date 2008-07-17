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

#include "scamp-catalog.h"
#include "qfits.h"

#include "cutest.h"

// This is really a demo, not a test.
void test_scampcat_1(CuTest* tc) {
    qfits_header* hdr;
    char* fn = create_temp_file("scampcat", "/tmp");
    printf("Creating scamp catalog: %s\n", fn);
    scamp_cat_t* scamp = scamp_catalog_open_for_writing(fn, FALSE);
    scamp_obj_t obj;

    hdr = qfits_header_default();
    qfits_header_add(hdr, "HELLO", "WORLD", "Comment?", NULL);

    if (scamp_catalog_write_field_header(scamp, hdr)) {
        printf("Failed to write field header.\n");
        exit(-1);
    }

    obj.x = 42;
    obj.y = 55;
    obj.err_a = 4;
    obj.err_b = 6;
    obj.err_theta = 67.5;
    obj.flux = 4400;
    obj.err_flux = 100;
    obj.flags = 0;

    if (scamp_catalog_write_object(scamp, &obj)) {
        printf("Failed to write object.\n");
        exit(-1);
    }
    if (scamp_catalog_close(scamp)) {
        printf("Failed to close.\n");
        exit(-1);
    }
}

// This is really a demo, not a test.
void test_scampref_2(CuTest* tc) {
    qfits_header* hdr;
    char* fn = create_temp_file("scampref", "/tmp");
    printf("Creating scamp catalog: %s\n", fn);
    scamp_cat_t* scamp = scamp_catalog_open_for_writing(fn, TRUE);
    scamp_ref_t obj;

    hdr = qfits_header_default();
    qfits_header_add(hdr, "HELLO", "WORLD", "Comment?", NULL);

    if (scamp_catalog_write_field_header(scamp, hdr)) {
        printf("Failed to write field header.\n");
        exit(-1);
    }

    obj.ra = 42;
    obj.dec = 55;
    obj.err_a = 4;
    obj.err_b = 6;
    obj.mag = 14.5;
    obj.err_mag = 0.25;

    if (scamp_catalog_write_reference(scamp, &obj)) {
        printf("Failed to write ref object.\n");
        exit(-1);
    }
    if (scamp_catalog_close(scamp)) {
        printf("Failed to close.\n");
        exit(-1);
    }
}

