/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "anwcs.h"
#include "an-bool.h"
#include "starutil.h"
#include "bl.h"
#include "xylist.h"
#include "rdlist.h"
#include "errors.h"

int wcs_rd2xy(const char* wcsfn, int wcsext,
              const char* rdlsfn, const char* xylsfn,
              const char* racol, const char* deccol,
              int forcetan, int forcewcslib,
              il* fields) {
    xylist_t* xyls = NULL;
    rdlist_t* rdls = NULL;
    anwcs_t* wcs = NULL;
    int i;
    anbool alloced_fields = FALSE;
    int rtn = -1;

    // read WCS.
    if (forcewcslib) {
        wcs = anwcs_open_wcslib(wcsfn, wcsext);
    } else if (forcetan) {
        wcs = anwcs_open_tan(wcsfn, wcsext);
    } else {
        wcs = anwcs_open(wcsfn, wcsext);
    }
    if (!wcs) {
        ERROR("Failed to read WCS file \"%s\", extension %i", wcsfn, wcsext);
        return -1;
    }

    // read RDLS.
    rdls = rdlist_open(rdlsfn);
    if (!rdls) {
        ERROR("Failed to read an RA,Dec list from file %s", rdlsfn);
        goto bailout;
    }
    if (racol)
        rdlist_set_raname(rdls, racol);
    if (deccol)
        rdlist_set_decname(rdls, deccol);

    // write XYLS.
    xyls = xylist_open_for_writing(xylsfn);
    if (!xyls) {
        ERROR("Failed to open file %s to write XYLS", xylsfn);
        goto bailout;
    }
    if (xylist_write_primary_header(xyls)) {
        ERROR("Failed to write header to XYLS file %s", xylsfn);
        goto bailout;
    }

    if (!fields) {
        alloced_fields = TRUE;
        fields = il_new(16);
    }
    if (!il_size(fields)) {
        // add all fields.
        int NF = rdlist_n_fields(rdls);
        for (i=1; i<=NF; i++)
            il_append(fields, i);
    }

    for (i=0; i<il_size(fields); i++) {
        int fieldnum = il_get(fields, i);
        int j;
        starxy_t xy;
        rd_t rd;

        if (!rdlist_read_field_num(rdls, fieldnum, &rd)) {
            ERROR("Failed to read rdls file \"%s\" field %i", rdlsfn, fieldnum);
            goto bailout;
        }

        starxy_alloc_data(&xy, rd_n(&rd), FALSE, FALSE);

        if (xylist_write_header(xyls)) {
            ERROR("Failed to write xyls field header");
            goto bailout;
        }

        for (j=0; j<rd_n(&rd); j++) {
            double x, y, ra, dec;
            ra  = rd_getra (&rd, j);
            dec = rd_getdec(&rd, j);
            if (anwcs_radec2pixelxy(wcs, ra, dec, &x, &y)) {
                static double nan = 1.0/0.0;
                ERROR("Point RA,Dec = (%g,%g) projects to the opposite side of the sphere", ra, dec);
                starxy_set(&xy, j, nan, nan);
                continue;
            }
            starxy_set(&xy, j, x, y);
        }
        if (xylist_write_field(xyls, &xy)) {
            ERROR("Failed to write xyls field");
            goto bailout;
        }
        if (xylist_fix_header(xyls)) {
            ERROR("Failed to fix xyls field header");
            goto bailout;
        }
        xylist_next_field(xyls);

        starxy_free_data(&xy);
        rd_free_data(&rd);
    }

    if (xylist_fix_primary_header(xyls) ||
        xylist_close(xyls)) {
        ERROR("Failed to fix header of XYLS file");
        goto bailout;
    }
    xyls = NULL;

    if (rdlist_close(rdls)) {
        ERROR("Failed to close RDLS file");
        goto bailout;
    }
    rdls = NULL;

    rtn = 0;

 bailout:
    if (alloced_fields)
        il_free(fields);
    if (rdls)
        rdlist_close(rdls);
    if (xyls)
        xylist_close(xyls);
    if (wcs)
        anwcs_free(wcs);
    return rtn;
}

