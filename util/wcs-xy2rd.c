/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anwcs.h"
#include "an-bool.h"
#include "qfits.h"
#include "starutil.h"
#include "bl.h"
#include "xylist.h"
#include "rdlist.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"

int wcs_xy2rd(const char* wcsfn, int ext,
			  const char* xylsfn, const char* rdlsfn,
              const char* xcol, const char* ycol,
			  anbool forcetan,
			  anbool forcewcslib,
              il* fields) {
	rdlist_t* rdls = NULL;
	xylist_t* xyls = NULL;
	anwcs_t* wcs = NULL;
	int i;
    int rtn = -1;
    anbool alloced_fields = FALSE;

	// read WCS.
	if (forcewcslib) {
		wcs = anwcs_open_wcslib(wcsfn, ext);
	} else if (forcetan) {
		wcs = anwcs_open_tan(wcsfn, ext);
	} else {
		wcs = anwcs_open(wcsfn, ext);
	}
	if (!wcs) {
		ERROR("Failed to read WCS file \"%s\", extension %i", wcsfn, ext);
		return -1;
	}

	// read XYLS.
	xyls = xylist_open(xylsfn);
	if (!xyls) {
		ERROR("Failed to read an xylist from file %s", xylsfn);
		goto bailout;
	}
    xylist_set_include_flux(xyls, FALSE);
    xylist_set_include_background(xyls, FALSE);
	if (xcol)
		xylist_set_xname(xyls, xcol);
	if (ycol)
		xylist_set_yname(xyls, ycol);

	// write RDLS.
	rdls = rdlist_open_for_writing(rdlsfn);
	if (!rdls) {
		ERROR("Failed to open file %s to write RDLS.\n", rdlsfn);
		goto bailout;
	}
	if (rdlist_write_primary_header(rdls)) {
		ERROR("Failed to write header to RDLS file %s.\n", rdlsfn);
		goto bailout;
	}

    if (!fields) {
        alloced_fields = TRUE;
        fields = il_new(16);
    }
	if (!il_size(fields)) {
		// add all fields.
		int NF = xylist_n_fields(xyls);
		for (i=1; i<=NF; i++)
			il_append(fields, i);
	}

	logverb("Processing %i extensions...\n", il_size(fields));
	for (i=0; i<il_size(fields); i++) {
		int fieldind = il_get(fields, i);
        starxy_t xy;
        rd_t rd;
		int j;

        if (!xylist_read_field_num(xyls, fieldind, &xy)) {
			ERROR("Failed to read xyls file %s, field %i", xylsfn, fieldind);
			goto bailout;
        }

		if (rdlist_write_header(rdls)) {
			ERROR("Failed to write rdls field header to %s", rdlsfn);
			goto bailout;
		}

        rd_alloc_data(&rd, starxy_n(&xy));

		for (j=0; j<starxy_n(&xy); j++) {
            double x, y, ra, dec;
            x = starxy_getx(&xy, j);
            y = starxy_gety(&xy, j);
			anwcs_pixelxy2radec(wcs, x, y, &ra, &dec);
            rd_setra (&rd, j, ra);
            rd_setdec(&rd, j, dec);
		}

        if (rdlist_write_field(rdls, &rd)) {
            ERROR("Failed to write rdls field to %s", rdlsfn);
			goto bailout;
        }
        rd_free_data(&rd);
        starxy_free_data(&xy);

		if (rdlist_fix_header(rdls)) {
			ERROR("Failed to fix rdls field header for %s", rdlsfn);
			goto bailout;
		}

        rdlist_next_field(rdls);
	}

	if (rdlist_fix_primary_header(rdls) ||
		rdlist_close(rdls)) {
		ERROR("Failed to fix header of RDLS file %s", rdlsfn);
		goto bailout;
	}
	rdls = NULL;

	if (xylist_close(xyls)) {
		ERROR("Failed to close XYLS file %s", xylsfn);
		goto bailout;
	}
	xyls = NULL;
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
