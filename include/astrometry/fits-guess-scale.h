/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef FITS_GUESS_SCALE_H
#define FITS_GUESS_SCALE_H

#include "astrometry/qfits_header.h"

#include "astrometry/bl.h"

int fits_guess_scale(const char* infn,
                     sl** p_methods, dl** p_scales);

/*
 If p_methods is not NULL but *p_methods is NULL, allocates a new sl; otherwise
 uses *p_methods.  Ditto for p_scales.  So do:

 sl* methods = NULL;
 fits_guess_scale_hdr(hdr, &methods, NULL);
 // ... process it...
 sl_free2(methods);

 OR

 sl* methods = sl_new(4);
 fits_guess_scale_hdr(hdr, &methods, NULL);
 // ... process it...
 sl_free2(methods);

 */
void fits_guess_scale_hdr(const qfits_header* hdr,
                          sl** p_methods, dl** p_scales);

#endif
