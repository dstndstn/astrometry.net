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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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
  @brief    qfits loader control object

  This structure serves two purposes: input and output for the qfits
  pixel loading facility. To request pixels from a FITS file, you
  need to allocate (statically or dynamically) such a structure and
  fill up the input fields (filename, xtension number, etc.) to specify
  the pixels you want from the file.

  Before performing the actual load, you must pass the initialized
  structure to qfitsloader_init() which will check whether the operation
  is feasible or not (check its returned value).

  If the operation was deemed feasible, you can proceed to load the pixels,
  passing the same structure to qfits_loadpix() which will fill up the
  output fields of the struct. Notice that a pixel buffer will have been
  allocated (through malloc or mmap) and placed into the structure. You
  need to call free() on this pointer when you are done with it,
  typically in the image or cube destructor.

  The qfitsloader_init() function is also useful to probe a FITS file
  for useful informations, like getting the size of images in the file,
  the pixel depth, or data offset.

  Example of a code that prints out various informations about
  a plane to load, without actually loading it:

  @code
int main(int argc, char * argv[])
{
    qfitsloader    ql;

    ql.filename = argv[1];
    ql.xtnum    = 0;
    ql.pnum     = 0;

    if (qfitsloader_init(&ql)!=0) {
        printf("cannot read info about %s\n", argv[1]);
        return -1;
    }

    printf(    "file         : %s\n"
            "xtnum        : %d\n"
            "pnum         : %d\n"
            "# xtensions  : %d\n"
            "size X       : %d\n"
            "size Y       : %d\n"
            "planes       : %d\n"
            "bitpix       : %d\n"
            "datastart    : %d\n"
            "datasize     : %d\n"
            "bscale       : %g\n"
            "bzero        : %g\n",
            ql.filename,
            ql.xtnum,
            ql.pnum,
            ql.exts,
            ql.lx,
            ql.ly,
            ql.np,
            ql.bitpix,
            ql.seg_start,
            ql.seg_size,
            ql.bscale,
            ql.bzero);
    return 0;
}
  @endcode
 */
/*----------------------------------------------------------------------------*/
typedef struct qfitsloader {

    /** Private field to see if structure has been initialized */
    int            _init;
    
    /** input: Name of the file you want to read pixels from */
    char    *    filename;
    /** input: xtension number you want to read */
    int            xtnum;
    /** input: Index of the plane you want, from 0 to np-1 */
    int            pnum;
    /** input: Pixel type you want (PTYPE_FLOAT, PTYPE_INT or PTYPE_DOUBLE) */
    int            ptype;
    /** input: Guarantee file copy or allow file mapping */
    int         map;

    /** output: Total number of extensions found in file */
    int            exts;
    /** output: Size in X of the requested plane */
    int            lx;
    /** output: Size in Y of the requested plane */
    int            ly;
    /** output: Number of planes present in this extension */
    int            np;
    /** output: BITPIX for this extension */
    int            bitpix;
    /** output: Start of the data segment (in bytes) for your request */
    int            seg_start;
    /** output: Size of the data segment (in bytes) for your request */
    int         seg_size;
    /** output: BSCALE found for this extension */
    double        bscale;
    /** output: BZERO found for this extension */
    double        bzero;

    /** output: Pointer to pixel buffer loaded as integer values */
    int        *    ibuf;
    /** output: Pointer to pixel buffer loaded as float values */
    float    *    fbuf;
    /** output: Pointer to pixel buffer loaded as double values */
    double    *    dbuf;

	// internal: allocated buffer.
	void* pixbuffer;

} qfitsloader;


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

int qfitsloader_init(qfitsloader *);
void qfitsloader_free_buffers(qfitsloader *);
int qfits_loadpix(qfitsloader *);
int qfits_loadpix_window(qfitsloader *, int, int, int, int);
int qfits_pixdump(const qfitsdumper *);

#endif
