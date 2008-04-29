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

#ifndef FITS_GUESS_SCALE_H
#define FITS_GUESS_SCALE_H

#include "bl.h"
#include "qfits.h"

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
