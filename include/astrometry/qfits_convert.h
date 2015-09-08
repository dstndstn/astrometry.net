/** Copyright 2009 Dustin Lang.
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 Converts a single pixel from C to FITS format.

 "ctype" = PTYPE_FLOAT, PTYPE_INT, etc.
 "fitstype" = BPP_8_UNSIGNED, BPP_16_SIGNED, etc.

 "cval": pointer to C value (input)
 "fitsval": pointer to FITS value (output)

Does byte-swapping, if necessary.
 */
int qfits_pixel_ctofits(int ctype, int fitstype,
						const void* cval, void* fitsval);


/**
 Returns the size in bytes of the given C pixel type.
 */
int qfits_pixel_ctype_size(int ctype);

/**
 Returns the size in bytes of the given FITS pixel type.
 */
int qfits_pixel_fitstype_size(int fitstype);
