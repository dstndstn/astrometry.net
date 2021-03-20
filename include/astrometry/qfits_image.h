/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_image.h,v 1.9 2006/02/23 11:04:17 yjung Exp $
 *
 * This file is part of the ESO QFITS Library
 * Copyright (C) 2001-2004 European Southern Observatory
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 * $Author: yjung $
 * $Date: 2006/02/23 11:04:17 $
 * $Revision: 1.9 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_IMAGE_H
#define QFITS_IMAGE_H

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

/** C pixel types. */
#define PTYPE_FLOAT        0
#define PTYPE_INT          1
#define PTYPE_DOUBLE       2
#define PTYPE_UINT8        3  /** Astrometry.net only */
#define PTYPE_INT16        4  /** Astrometry.net only */

/* FITS pixel depths */
/* FITS BITPIX=8 */
#define BPP_8_UNSIGNED        (8)
/* FITS BITPIX=16 */
#define BPP_16_SIGNED        (16)
/* FITS BITPIX=32 */
#define BPP_32_SIGNED        (32)
/* FITS BITPIX=-32 */
#define BPP_IEEE_FLOAT      (-32)
/* FITS BITPIX=-64 */
#define BPP_IEEE_DOUBLE     (-64)
/* Default BITPIX for output */
#define BPP_DEFAULT         BPP_IEEE_FLOAT

/* Compute the number of bytes per pixel for a given BITPIX value */
#define BYTESPERPIXEL(x)    (   ((x) == BPP_8_UNSIGNED) ?     1 : \
                                ((x) == BPP_16_SIGNED)  ?     2 : \
                                ((x) == BPP_32_SIGNED)  ?     4 : \
                                ((x) == BPP_IEEE_FLOAT) ?     4 : \
                                ((x) == BPP_IEEE_DOUBLE) ?    8 : 0 ) 

/*-----------------------------------------------------------------------------
                                   New types
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Alias for unsigned char

  A 'byte' is just an alias for an unsigned char. It is only defined
  for readability.
 */
/*----------------------------------------------------------------------------*/
typedef unsigned char byte;


/*----------------------------------------------------------------------------*/
/**
  @brief    qfits dumper control object

  This structure offers various control parameters to dump a pixel
  buffer to a FITS file. The buffer will be dumped as requested
  to the requested file in append mode. Of course, the requested file
  must be writeable for the operation to succeed.

  The following example demonstrates how to save a linear ramp sized
  100x100 to a FITS file with BITPIX=16. Notice that this code only
  dumps the pixel buffer, no header information is provided in this
  case.

  @code
    int   i, j;
    int * ibuf;
    qfitsdumper    qd;

    // Fill a buffer with 100x100 int pixels
    ibuf = malloc(100 * 100 * sizeof(int));
    for (j=0; j<100; j++) {
        for (i=0; i<100; i++) {
            ibuf[i+j*100] = i+j;
        }
    }

    qd.filename  = "out.fits";     // Output file name
    qd.npix      = 100 * 100;      // Number of pixels
    qd.ptype     = PTYPE_INT;      // Input buffer type
    qd.ibuf      = ibuf;           // Set buffer pointer
    qd.out_ptype = BPP_16_SIGNED;  // Save with BITPIX=16

    // Dump buffer to file (error checking omitted for clarity)
    qfits_pixdump(&qd);

    free(ibuf);
  @endcode
  
  If the provided output file name is "STDOUT" (all capitals), the
  function will dump the pixels to the stdout steam (usually the console,
  could have been re-directed).
 */
/*----------------------------------------------------------------------------*/
typedef struct qfitsdumper {

    /** Name of the file to dump to, "STDOUT" to dump to stdout */
    const char     *    filename;
    /** Number of pixels in the buffer to dump */
    int            npix;
    /** Buffer type: PTYPE_FLOAT, PTYPE_INT or PTYPE_DOUBLE */
    int            ptype;

    /** Pointer to input integer pixel buffer */
    const int        *    ibuf;
    /** Pointer to input float pixel buffer */
    const float    *    fbuf;
    /** Pointer to input double pixel buffer */
    const double    *    dbuf;

	/** Pointer to generic pixel buffer. */
	const void* vbuf;

    /** Requested BITPIX in output FITS file */
    int            out_ptype;
} qfitsdumper;

/*-----------------------------------------------------------------------------
                               Function prototypes
 -----------------------------------------------------------------------------*/

int qfits_pixdump(const qfitsdumper *);

#endif
