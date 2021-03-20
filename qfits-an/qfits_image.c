/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_image.c,v 1.12 2006/02/23 11:25:25 yjung Exp $
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
 * $Date: 2006/02/23 11:25:25 $
 * $Revision: 1.12 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "qfits_config.h"
#include "qfits_image.h"
#include "qfits_rw.h"
#include "qfits_header.h"
#include "qfits_byteswap.h"
#include "qfits_tools.h"
#include "qfits_error.h"
#include "qfits_memory.h"
#include "qfits_std.h"

/*-----------------------------------------------------------------------------
                                Defines
 -----------------------------------------------------------------------------*/

#define QFITSLOADERINIT_MAGIC   0xcafe

/*-----------------------------------------------------------------------------
                            Function prototypes
 -----------------------------------------------------------------------------*/

static byte * qfits_pixdump_float(const float *, int, int);
static byte * qfits_pixdump_int(const int *, int, int);
static byte * qfits_pixdump_double(const double *, int, int);

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_image Pixel loader for FITS images.
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                            Function codes
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Dump a pixel buffer to an output FITS file in append mode.
  @param    qd  qfitsdumper control object.
  @return   int 0 if Ok, -1 otherwise.

  This function takes in input a qfitsdumper control object. This object
  must be allocated beforehand and contain valid references to the data
  to save, and how to save it.

  The minimum fields to fill are:

  - filename: Name of the FITS file to dump to.
  - npix: Number of pixels in the buffer to be dumped.
  - ptype: Type of the passed buffer (PTYPE_FLOAT, PTYPE_INT, PTYPE_DOUBLE)
  - out_ptype: Requested FITS BITPIX for the output.

  One of the following fields must point to the corresponding pixel
  buffer:

  - ibuf for an int pixel buffer (ptype=PTYPE_INT)
  - fbuf for a float pixel buffer (ptype=PTYPE_FLOAT)
  - dbuf for a double pixel buffer (ptype=PTYPE_DOUBLE)

  This is a fairly low-level function, in the sense that it does not
  check that the output file already contains a proper header or even
  that the file it is appending to is indeed a FITS file. It will
  convert the pixel buffer to the requested BITPIX type and append
  the data to the file, without padding with zeros. See qfits_zeropad()
  about padding.

  If the given output file name is "STDOUT" (all caps), the dump
  will be performed to stdout.
 */
/*----------------------------------------------------------------------------*/
int qfits_pixdump(const qfitsdumper * qd)
{
    FILE    *   f_out;
    byte    *   buf_out;
    int         buf_free;
    int         buf_sz;

    /* Check inputs */
    if (qd==NULL) return -1;
    if (qd->filename==NULL) return -1;
    switch (qd->ptype) {
        case PTYPE_FLOAT:
        if (qd->fbuf==NULL) return -1;
        break;

        case PTYPE_DOUBLE:
        if (qd->dbuf==NULL) return -1;
        break;

        case PTYPE_INT:
        if (qd->ibuf==NULL) return -1;
        break;

        default:
        return -1;
    }
    if (qd->npix <= 0) {
        qfits_error("Negative or NULL number of pixels specified");
        return -1;
    }

    /*
     * Special cases: input buffer is identical to requested format.
     * This is only possible on big-endian machines, since FITS is
     * big-endian only.
     */
    buf_out = NULL;
    buf_free = 1;
#ifdef WORDS_BIGENDIAN
    if (qd->ptype==PTYPE_FLOAT && qd->out_ptype==-32) {
        buf_out = (byte*)qd->fbuf;
        buf_free=0;
    } else if (qd->ptype==PTYPE_DOUBLE && qd->out_ptype==-64) {
        buf_out = (byte*)qd->dbuf;
        buf_free=0;
    } else if (qd->ptype==PTYPE_INT && qd->out_ptype==32) {
        buf_out = (byte*)qd->ibuf;
        buf_free=0;
    }
#endif
    buf_sz = qd->npix * BYTESPERPIXEL(qd->out_ptype);

    /* General case */
    if (buf_out==NULL) {
        switch (qd->ptype) {
            /* Convert buffer */
            case PTYPE_FLOAT:
            buf_out = qfits_pixdump_float(  qd->fbuf,
                                            qd->npix,
                                            qd->out_ptype);
            break;

            /* Convert buffer */
            case PTYPE_INT:
            buf_out = qfits_pixdump_int(    qd->ibuf,
                                            qd->npix,
                                            qd->out_ptype);
            break;

            /* Convert buffer */
            case PTYPE_DOUBLE:
            buf_out = qfits_pixdump_double( qd->dbuf,
                                             qd->npix,
                                             qd->out_ptype);
            break;
        }
    }
    if (buf_out==NULL) {
        qfits_error("cannot dump pixel buffer");
        return -1;
    }

    /* Dump buffer */
    if (!strncmp(qd->filename, "STDOUT", 6)) {
        f_out = stdout;
    } else {
        f_out = fopen(qd->filename, "a");
    }
    if (f_out==NULL) {
        qfits_free(buf_out);
        return -1;
    }
    fwrite(buf_out, buf_sz, 1, f_out);
    if (buf_free) {
        qfits_free(buf_out);
    }
    if (f_out!=stdout) {
        fclose(f_out);
    }
    return 0;
}

/**@}*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Convert a float pixel buffer to a byte buffer.
  @param    buf     Input float buffer.
  @param    npix    Number of pixels in the input buffer.
  @param    ptype   Requested output BITPIX type.
  @return   1 pointer to a newly allocated byte buffer.

  This function converts the given float buffer to a buffer of bytes
  suitable for dumping to a FITS file (i.e. big-endian, in the
  requested pixel type). The returned pointer must be deallocated
  using the qfits_free() function.
 */
/*----------------------------------------------------------------------------*/
static byte * qfits_pixdump_float(const float * buf, int npix, int ptype)
{
    byte    *   buf_out;
    register byte * op;
    int         i;
    int         lpix;
    short       spix;
    double      dpix;

    buf_out = qfits_malloc((size_t)npix * (size_t)BYTESPERPIXEL(ptype));
    op = buf_out;
    switch (ptype) {
        case 8:
        /* Convert from float to 8 bits */
        for (i=0; i<npix; i++) {
            if (buf[i]>255.0) {
                *op++ = (byte)0xff;
            } else if (buf[i]<0.0) {
                *op++ = (byte)0x00;
            } else {
                *op++ = (byte)buf[i];
            }
        }
        break;

        case 16:
        /* Convert from float to 16 bits */
        for (i=0; i<npix; i++) {
            if (buf[i]>32767.0) {
                *op++ = (byte)0x7f;
                *op++ = (byte)0xff;
            } else if (buf[i]<-32768.0) {
                *op++ = (byte)0x80;
                *op++ = 0x00;
            } else {
                spix = (short)buf[i];
                *op++ = (spix >> 8);
                *op++ = (spix & (byte)0xff);
            }
        }
        break;

        case 32:
        /* Convert from float to 32 bits */
        for (i=0; i<npix; i++) {
            if (buf[i] > 2147483647.0) {
                *op++ = (byte)0x7f;
                *op++ = (byte)0xff;
                *op++ = (byte)0xff;
                *op++ = (byte)0xff;
            } else if (buf[i]<-2147483648.0) {
                *op++ = (byte)0x80;
                *op++ = (byte)0x00;
                *op++ = (byte)0x00;
                *op++ = (byte)0x00;
            } else {
                lpix = (int)buf[i];
                *op++ = (byte)(lpix >> 24);
                *op++ = (byte)(lpix >> 16) & 0xff;
                *op++ = (byte)(lpix >> 8 ) & 0xff;
                *op++ = (byte)(lpix) & 0xff;
            }
        }
        break;

        case -32:
        /* Convert from float to float */
        memcpy(op, buf, npix * sizeof(float));
#ifndef WORDS_BIGENDIAN
        for (i=0; i<npix; i++) {
            qfits_swap_bytes(op, 4);
            op++;
            op++;
            op++;
            op++;
        }
#endif
        break;

        case -64:
        /* Convert from float to double */
        for (i=0; i<npix; i++) {
            dpix = (double)buf[i];
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(&dpix, 8);
#endif
            memcpy(op, &dpix, 8);
            op += 8;
        }
        break;

        default:
			qfits_error("Pixel type %i not supported yet", ptype);
        buf_out = NULL;
        break;
    }
    return buf_out;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Convert an int pixel buffer to a byte buffer.
  @param    buf     Input int buffer.
  @param    npix    Number of pixels in the input buffer.
  @param    ptype   Requested output BITPIX type.
  @return   1 pointer to a newly allocated byte buffer.

  This function converts the given int buffer to a buffer of bytes
  suitable for dumping to a FITS file (i.e. big-endian, in the
  requested pixel type). The returned pointer must be deallocated
  using the qfits_free() function.
 */
/*----------------------------------------------------------------------------*/
static byte * qfits_pixdump_int(const int * buf, int npix, int ptype)
{
    byte    *   buf_out;
    register byte * op;
    int i;

    short   spix;
    float   fpix;
    double  dpix;

    buf_out = qfits_malloc((size_t)npix * (size_t)BYTESPERPIXEL(ptype));
    op = buf_out;
    switch (ptype) {
        case 8:
        /* Convert from int32 to 8 bits */
        for (i=0; i<npix; i++) {
            if (buf[i]>255) {
                *op++ = (byte)0xff;
            } else if (buf[i]<0) {
                *op++ = (byte)0x00;
            } else {
                *op++ = (byte)buf[i];
            }
        }
        break;

        case 16:
        /* Convert from int32 to 16 bits */
        for (i=0; i<npix; i++) {
            if (buf[i]>32767) {
                spix = 32767;
            } else if (buf[i]<-32768) {
                spix = -32768;
            } else {
                spix = (short)buf[i];
            }
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(&spix, 2);
#endif
            memcpy(op, &spix, 2);
            op += 2;
        }
        break;

        case 32:
        /* Convert from int32 to 32 bits */
        memcpy(op, buf, npix * sizeof(int));
#ifndef WORDS_BIGENDIAN
        for (i=0; i<npix; i++) {
            qfits_swap_bytes(op, 4);
            op+=4;
        }
#endif
        break;

        case -32:
        /* Convert from int32 to float */
        for (i=0; i<npix; i++) {
            fpix = (float)buf[i];
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(&fpix, 4);
#endif
            memcpy(op, &fpix, 4);
            op += 4;
        }
        break;

        case -64:
        /* Convert from int32 to double */
        for (i=0; i<npix; i++) {
            dpix = (double)buf[i];
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(&dpix, 8);
#endif
            memcpy(op, &dpix, 8);
            op += 8;
        }
        break; 

        default:
			qfits_error("Pixel type %i not supported yet", ptype);
        buf_out = NULL;
        break;
    }
    return buf_out;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Convert a double pixel buffer to a byte buffer.
  @param    buf     Input double buffer.
  @param    npix    Number of pixels in the input buffer.
  @param    ptype   Requested output BITPIX type.
  @return   1 pointer to a newly allocated byte buffer.

  This function converts the given double buffer to a buffer of bytes
  suitable for dumping to a FITS file (i.e. big-endian, in the
  requested pixel type). The returned pointer must be deallocated
  using the qfits_free() function.
 */
/*----------------------------------------------------------------------------*/
static byte * qfits_pixdump_double(const double * buf, int npix, int ptype)
{
    byte    *   buf_out;
    register byte  * op;
    int i;

    short   spix;
    float   fpix;
    int     lpix;

    buf_out = qfits_malloc((size_t)npix * (size_t)BYTESPERPIXEL(ptype));
    op = buf_out;
    switch (ptype) {
        case 8:
        /* Convert from double to 8 bits */
        for (i=0; i<npix; i++) {
            if (buf[i]>255.0) {
                *op++ = (byte)0xff;
            } else if (buf[i]<0.0) {
                *op++ = (byte)0x00;
            } else {
                *op++ = (byte)buf[i];
            }
        }
        break;

        case 16:
        /* Convert from double to 16 bits */
        for (i=0; i<npix; i++) {
            if (buf[i]>32767.0) {
                spix = 32767;
            } else if (buf[i]<-32768.0) {
                spix = -32768;
            } else {
                spix = (short)buf[i];
            }
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(&spix, 2);
#endif
            memcpy(op, &spix, 2);
            op += 2;
        }
        break;

        case 32:
        /* Convert from double to 32 bits */
        for (i=0; i<npix; i++) {
            if (buf[i] > 2147483647.0) {
                lpix = 2147483647;
            } else if (buf[i] < -2147483648.0) {
                lpix = -2147483647;
            } else {
                lpix = (int)buf[i];
            }
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(&lpix, 4);
#endif
            memcpy(op, &lpix, 4);
            op += 4;
        }
        break;

        case -32:
        /* Convert from double to float */
        for (i=0; i<npix; i++) {
            fpix = (float)buf[i];
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(&fpix, 4);
#endif
            memcpy(op, &fpix, 4);
            op += 4;
        }
        break;

        case -64:
        /* Convert from double to double */
        memcpy(op, buf, npix * 8);
#ifndef WORDS_BIGENDIAN
        for (i=0; i<npix; i++) {
            qfits_swap_bytes(op, 8);
            op += 8;
        }
#endif
        break; 

        default:
			qfits_error("pixel type %i not supported yet", ptype);
        buf_out = NULL;
        break;
    }
    return buf_out;
}

/* Test code */
#ifdef TESTPIXIO
static void qfitsloader_dump(qfitsloader * ql)
{
    fprintf(stderr,
            "file      : %s\n"
            "xtnum     : %d\n"
            "pnum      : %d\n"
            "ptype     : %d\n"
            "lx        : %d\n"
            "ly        : %d\n"
            "np        : %d\n"
            "bitpix    : %d\n"
            "seg_start : %d\n"
            "bscale    : %g\n"
            "bzero     : %g\n"
            "ibuf      : %p\n"
            "fbuf      : %p\n"
            "dbuf      : %p\n",
            ql->filename,
            ql->xtnum,
            ql->pnum,
            ql->ptype,
            ql->lx,
            ql->ly,
            ql->np,
            ql->bitpix,
            ql->seg_start,
            ql->bscale,
            ql->bzero,
            ql->ibuf,
            ql->fbuf,
            ql->dbuf);
}

int main (int argc, char * argv[])
{
    qfitsloader ql;

    if (argc<2) {
        printf("use: %s <FITS>\n", argv[0]);
        return 1;
    }

    ql.filename = argv[1];
    ql.xtnum    = 0;
    ql.pnum     = 0;
    ql.ptype    = PTYPE_FLOAT;

    if (qfits_loadpix(&ql)!=0) {
        printf("error occurred during loading: abort\n");
        return -1;
    }
    qfitsloader_dump(&ql);
    printf("pix[0]=%g\n"
           "pix[100]=%g\n"
           "pix[10000]=%g\n",
           ql.fbuf[0],
           ql.fbuf[100],
           ql.fbuf[10000]);
    qfits_free(ql.fbuf);

    ql.ptype   = PTYPE_INT;
    if (qfits_loadpix(&ql)!=0) {
        printf("error occurred during loading: abort\n");
        return -1;
    }
    qfitsloader_dump(&ql);
    printf("pix[0]=%d\n"
           "pix[100]=%d\n"
           "pix[10000]=%d\n",
           ql.ibuf[0],
           ql.ibuf[100],
           ql.ibuf[10000]);
    qfits_free(ql.ibuf);


    ql.ptype   = PTYPE_DOUBLE;
    if (qfits_loadpix(&ql)!=0) {
        printf("error occurred during loading: abort\n");
        return -1;
    }
    qfitsloader_dump(&ql);
    printf("pix[0]=%g\n"
           "pix[100]=%g\n"
           "pix[10000]=%g\n",
           ql.dbuf[0],
           ql.dbuf[100],
           ql.dbuf[10000]);
    qfits_free(ql.dbuf);

    return 0;
}
#endif
