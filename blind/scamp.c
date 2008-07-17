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

#include "scamp.h"
#include "scamp-catalog.h"
#include "sip_qfits.h"
#include "errors.h"
#include "log.h"

int scamp_write_field(const qfits_header* imageheader,
                      const sip_t* wcs,
                      const starxy_t* xy,
                      const char* filename) {
    scamp_cat_t* scamp;
    qfits_header* hdr;
    int i;

    if (!imageheader)
        hdr = qfits_table_prim_header_default();
    else
        hdr = qfits_header_copy(imageheader);

    sip_add_to_header(hdr, wcs);

    scamp = scamp_catalog_open_for_writing(filename);
    if (!scamp) {
        return -1;
    }

    if (scamp_catalog_write_field_header(scamp, hdr)) {
        return -1;
    }
    qfits_header_destroy(hdr);

    for (i=0; i<starxy_n(xy); i++) {
        scamp_obj_t obj;
        obj.x = starxy_getx(xy, i);
        obj.y = starxy_gety(xy, i);
        obj.err_a = 1.0;
        obj.err_b = 1.0;
        obj.err_theta = 0.0;
        if (xy->flux)
            obj.flux = xy->flux[i];
        else
            obj.flux = 0.0;
        obj.err_flux = 1.0;
        obj.flags = 0;
        if (scamp_catalog_write_object(scamp, &obj)) {
            return -1;
        }
    }

    if (scamp_catalog_close(scamp)) {
        return -1;
    }
    return 0;
}

