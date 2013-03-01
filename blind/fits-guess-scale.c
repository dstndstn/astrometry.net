/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "fits-guess-scale.h"
#include "qfits.h"
#include "sip.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"

int fits_guess_scale(const char* infn,
                     sl** p_methods, dl** p_scales) {
	qfits_header* hdr;

	hdr = qfits_header_read(infn);
	if (!hdr) {
		ERROR("Failed to read FITS header");
        return -1;
	}
    fits_guess_scale_hdr(hdr, p_methods, p_scales);
    qfits_header_destroy(hdr);
    return 0;
}

static void addscale(sl* methods, dl* scales,
                     const char* method, double scale) {
    if (methods)
        sl_append(methods, method);
    if (scales)
        dl_append(scales, scale);
}

void fits_guess_scale_hdr(const qfits_header* hdr,
                          sl** p_methods, dl** p_scales) {
	sip_t sip;
	double val;
	anbool gotsip = FALSE;
    char* errstr;

    sl* methods = NULL;
    dl* scales = NULL;

    if (p_methods) {
        if (!*p_methods)
            *p_methods = sl_new(4);
        methods = *p_methods;
    }
    if (p_scales) {
        if (!*p_scales)
            *p_scales = dl_new(4);
        scales = *p_scales;
    }

	memset(&sip, 0, sizeof(sip_t));

    errors_start_logging_to_string();

	if (sip_read_header(hdr, &sip)) {
        val = sip_pixel_scale(&sip);
        if (val != 0.0) {
            addscale(methods, scales, "sip", val);
            gotsip = TRUE;
		}
	}
    errstr = errors_stop_logging_to_string("\n  ");
    logverb("fits-guess-scale: failed to read SIP/TAN header:\n  %s\n", errstr);
    free(errstr);

	if (!gotsip) {
		// it might have a correct CD matrix but be missing other parts (eg CRVAL)
		double cd11, cd12, cd21, cd22;
		double errval = -HUGE_VAL;
		cd11 = qfits_header_getdouble(hdr, "CD1_1", errval);
		cd12 = qfits_header_getdouble(hdr, "CD1_2", errval);
		cd21 = qfits_header_getdouble(hdr, "CD2_1", errval);
		cd22 = qfits_header_getdouble(hdr, "CD2_2", errval);
		if ((cd11 != errval) && (cd12 != errval) && (cd21 != errval) && (cd22 != errval)) {
			val = cd11 * cd22 - cd12 * cd21;
			if (val != 0.0)
                addscale(methods, scales, "cd", sqrt(fabs(val)));
		}
	}

	val = qfits_header_getdouble(hdr, "PIXSCALE", -1.0);
	if (val != -1.0)
        addscale(methods, scales, "pixscale", val);

    /* Why all this?
     val = qfits_header_getdouble(hdr, "PIXSCAL1", -1.0);
     if (val != -1.0) {
     if (val != 0.0) {
     printf("scale pixscal1 %g\n", val);
     } else {
     val = atof(qfits_pretty_string(qfits_header_getstr(hdr, "PIXSCAL1")));
     if (val != 0.0) {
     printf("scale pixscal1 %g\n", val);
     }
     }
     }
     */

     val = qfits_header_getdouble(hdr, "PIXSCAL1", 0.0);
     if (val != 0.0)
         addscale(methods, scales, "pixscal1", val);

     val = qfits_header_getdouble(hdr, "PIXSCAL2", 0.0);
     if (val != 0.0)
         addscale(methods, scales, "pixscal2", val);

     val = qfits_header_getdouble(hdr, "PLATESC", 0.0);
     if (val != 0.0)
         addscale(methods, scales, "platesc", val);

	val = qfits_header_getdouble(hdr, "CCDSCALE", 0.0);
     if (val != 0.0)
         addscale(methods, scales, "ccdscale", val);

	val = qfits_header_getdouble(hdr, "CDELT1", 0.0);
     if (val != 0.0)
         addscale(methods, scales, "cdelt1", 3600.0 * fabs(val));
}

