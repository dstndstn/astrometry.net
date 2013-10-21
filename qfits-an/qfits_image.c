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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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

#include "config.h"

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

static float * qfits_pixin_float(byte *, int, int, double, double);
static int * qfits_pixin_int(byte *, int, int, double, double);
static double * qfits_pixin_double(byte *, int, int, double, double);
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
  @brief    Initialize a qfitsloader control object.
  @param    ql  qfitsloader object to initialize.
  @return   int 0 if Ok, -1 if error occurred.

  This function expects a qfitsloader object with a number of input
  fields correctly filled in. The minimum fields to set are:

  - filename: Name of the file to examine.
  - xtnum: Extension number to examine (0 for main section).
  - pnum: Plane number in the requested extension.
  - map : loading mode - flag to know if the file has to be mapped

  You can go ahead with these fields only if you only want to get
  file information for this plane in this extension. If you want
  to later load the plane, you must additionally fill the 'ptype'
  field to a correct value (PTYPE_INT, PTYPE_FLOAT, PTYPE_DOUBLE)
  before calling qfits_loadpix() so that it knows which conversion
  to perform.

  This function is basically a probe sent on a FITS file to ask
  qfits if loading these data would be Ok or not. The actual loading
  is performed by qfits_loadpix() afterwards.
 */
/*----------------------------------------------------------------------------*/
int qfitsloader_init(qfitsloader * ql)
{
    qfits_header    *   fh;

    int     n_ext;
    int     seg_start;
    int     seg_size;
    int     bitpix, naxis, naxis1, naxis2, naxis3;
    char *  xt_type;
    char xt_type_2[FITS_LINESZ+1];
    char *  sval;
    struct stat sta;

    /* Check passed object is allocated */
    if (ql==NULL) {
        qfits_error("pixio: NULL loader");
        return -1;
    }

    /* Object must contain a filename */
    if (ql->filename == NULL) {
        qfits_error("pixio: NULL filename in loader");
        return -1;
    }
    /* Check requested file exists and contains data */
    if (stat(ql->filename, &sta)!=0) {
        qfits_error("no such file: %s", ql->filename);
        return -1;
    }
    if (sta.st_size<1) {
        qfits_error("empty file: %s", ql->filename);
        return -1;
    }

    /* Requested extension number must be positive */
    if (ql->xtnum<0) {
        qfits_error("pixio: negative xtnum in loader");
        return -1;
    }
    /* Requested returned pixel type must be legal */
    if ((ql->ptype!=PTYPE_INT) &&
        (ql->ptype!=PTYPE_FLOAT) &&
        (ql->ptype!=PTYPE_DOUBLE)) {
        qfits_error("pixio: invalid ptype in loader");
        return -1;
    }
    /* Check requested file is FITS */
    if (qfits_is_fits(ql->filename)!=1) {
        qfits_error("pixio: not a FITS file: %s", ql->filename);
        return -1;
    }
    /* Get number of extensions for this file */
    n_ext = qfits_query_n_ext(ql->filename);
    if (n_ext==-1) {
        qfits_error("pixio: cannot get number of extensions in %s",
                    ql->filename);
        return -1;
    }
    /* Check requested extension falls within range */
    if (ql->xtnum > n_ext) {
        qfits_error("pixio: requested extension %d but file %s has %d\n",
                    ql->xtnum,
                    ql->filename,
                    n_ext);
        return -1;
    }
    ql->exts = n_ext;
    /* Get segment offset and size for the requested buffer */
    if (qfits_get_datinfo(ql->filename,
                          ql->xtnum,
                          &seg_start,
                          &seg_size)!=0) {
        qfits_error("pixio: cannot get seginfo for %s extension %d",
                    ql->filename,
                    ql->xtnum);
        return -1;
    }
    /* Check segment size is consistent with file size */
    if (sta.st_size < (seg_start+seg_size)) {
        return -1;
    }
    ql->seg_start = seg_start;
    ql->seg_size  = seg_size;

    /* Get file header */
    fh = qfits_header_readext(ql->filename, ql->xtnum);
    if (fh==NULL) {
        qfits_error("pixio: cannot read header from ext %d in %s",
                    ql->xtnum,
                    ql->filename);
        return -1;
    }
    /* If the requested image is within an extension */
    if (ql->xtnum>0) {
        /* Check extension is an image */
        xt_type = qfits_header_getstr(fh, "XTENSION");
        if (xt_type==NULL) {
            qfits_error("pixio: cannot read extension type for ext %d in %s",
                        ql->xtnum,
                        ql->filename);
            qfits_header_destroy(fh);
            return -1;
        }
        qfits_pretty_string_r(xt_type, xt_type_2);
        if (strcmp(xt_type_2, "IMAGE")) {
            qfits_error(
                "pixio: not an image -- extension %d in %s has type [%s]",
                ql->xtnum,
                ql->filename,
                xt_type_2);
            qfits_header_destroy(fh);
            return -1;
        }
    }

    /* Get file root informations */
    bitpix = qfits_header_getint(fh, "BITPIX", -1);
    naxis  = qfits_header_getint(fh, "NAXIS",  -1);
    naxis1 = qfits_header_getint(fh, "NAXIS1", -1);
    naxis2 = qfits_header_getint(fh, "NAXIS2", -1);
    naxis3 = qfits_header_getint(fh, "NAXIS3", -1);

    /* Get BSCALE and BZERO if available */
    sval = qfits_header_getstr(fh, "BSCALE");
    if (sval==NULL) {
        ql->bscale = 1.0;
    } else {
        ql->bscale = atof(sval);
    }
    sval = qfits_header_getstr(fh, "BZERO");
    if (sval==NULL) {
        ql->bzero = 0.0;
    } else {
        ql->bzero = atof(sval);
    }

    /* Destroy header */
    qfits_header_destroy(fh);

    /* Check BITPIX is present */
    if (bitpix==-1) {
        qfits_error("pixio: missing BITPIX in file %s", ql->filename);
        return -1;
    }
    /* Check BITPIX is valid */
    if ((bitpix!=   8) &&
        (bitpix!=  16) &&
        (bitpix!=  32) &&
        (bitpix!= -32) &&
        (bitpix!= -64)) {
        qfits_error("pixio: invalid BITPIX (%d) in file %s",
                    bitpix,
                    ql->filename);
        return -1;
    }
    ql->bitpix = bitpix;

    /* Check NAXIS is present and valid */
    if (naxis<0) {
        qfits_error("pixio: no NAXIS in file %s", ql->filename);
        return -1;
    }
    if (naxis==0) {
        qfits_error("pixio: no pixel in ext %d of file %s",
                    ql->xtnum,
                    ql->filename);
        return -1;
    }
    if (naxis>3) {
        qfits_error("pixio: NAXIS>3 (%d) unsupported", naxis);
        return -1;
    }
    /* NAXIS1 must always be present */
    if (naxis1<0) {
        qfits_error("pixio: no NAXIS1 in file %s", ql->filename);
        return -1;
    }
    /* Update dimension fields in loader struct */
    ql->lx = 1;
    ql->ly = 1;
    ql->np = 1;

    switch (naxis) {
        case 1:
        ql->lx = naxis1;
        break;

        case 3:
        if (naxis3<0) {
            qfits_error("pixio: no NAXIS3 in file %s", ql->filename);
            return -1;
        }
        ql->np = naxis3;
        /* No break statement: intended to go through next case */

        case 2:
        if (naxis2<0) {
            qfits_error("pixio: no NAXIS2 in file %s", ql->filename);
            return -1;
        }
        ql->ly = naxis2;
        ql->lx = naxis1;
        break;
    }
    /* Check that requested plane number falls within range */
    if (ql->pnum >= ql->np) {
        qfits_error("pixio: requested plane %d but NAXIS3=%d",
                    ql->pnum,
                    ql->np);
        return -1;
    }

	ql->pixbuffer = NULL;
	ql->ibuf = NULL;
	ql->fbuf = NULL;
	ql->dbuf = NULL;

    /* Everything Ok, fields have been filled along. */
    /* Mark the structure as initialized */
    ql->_init = QFITSLOADERINIT_MAGIC;
    return 0;
}

void qfitsloader_free_buffers(qfitsloader * ql) {
	free(ql->pixbuffer);
	ql->pixbuffer = NULL;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Load a pixel buffer for one complete image.
  @param    ql  Allocated and initialized qfitsloader control object.
  @return   int 0 if Ok, -1 if error occurred.
  @see      qfits_loadpix_window
*/
/*----------------------------------------------------------------------------*/
int qfits_loadpix(qfitsloader * ql)
{
    if (ql==NULL) return -1;
    return qfits_loadpix_window(ql, 1, 1, ql->lx, ql->ly);
}
    
/*----------------------------------------------------------------------------*/
/**
  @brief    Load a pixel buffer for one image window
  @param    ql  Allocated and initialized qfitsloader control object.
  @param    llx 
  @param    lly     Position of the window (start with (1,1))
  @param    urx 
  @param    ury 
  @return   int 0 if Ok, -1 if error occurred.

  This function performs a load of a pixel buffer into memory. It
  expects an allocated and initialized qfitsloader object in input.
  See qfitsloader_init() about initializing the object.

  This function will fill up the ibuf/fbuf/dbuf field, depending
  on the requested pixel type (resp. int, float or double).
  
  If llx lly urx and ury do not specify the whole image, ql->map must be 0, 
  we do not want to mmap a file an load only a part of it.
 */
/*----------------------------------------------------------------------------*/
int qfits_loadpix_window(
        qfitsloader     *   ql,
        int                 llx,
        int                 lly,
        int                 urx,
        int                 ury)
{
    byte    *   fptr;
    size_t      fsize;
    int         datastart;
    int         imagesize, window_size, linesize;
    FILE    *   lfile;
    int         dataread;
    int         nx, ny;
    int         i;

    /* Check inputs */
    if (ql==NULL) return -1;
    if (ql->_init != QFITSLOADERINIT_MAGIC) {
        qfits_error("pixio: called with unitialized obj");
        return -1;
    }
    if (llx>urx || lly>ury || llx<1 || lly<1 || urx>ql->lx || ury>ql->ly) {
        qfits_error("pixio: invalid window specification");
        return -1;
    }

    /* No map if only a zone is specified */
    if (llx != 1 || lly != 1 || urx != ql->lx || ury != ql->ly) {
        if (ql->map == 1) {
            qfits_error("pixio: cannot mmap for a part of the image");
            return -1;
        }
    }

    /* Initialise */
    nx = urx-llx+1;
    ny = ury-lly+1;
    imagesize = ql->lx * ql->ly * BYTESPERPIXEL(ql->bitpix);
    window_size = nx * ny * BYTESPERPIXEL(ql->bitpix);
    linesize = nx * BYTESPERPIXEL(ql->bitpix);
    datastart = ql->seg_start + ql->pnum * imagesize;

    /* Check loading mode */
    if (ql->map) {
        /* Map the file in */
        fptr = (byte*)qfits_falloc(ql->filename, datastart, &fsize);
        if (fptr==NULL) {
            qfits_error("pixio: cannot falloc(%s)", ql->filename);
            return -1;
        }
    } else {
        /* Open the file */
        if ((lfile=fopen(ql->filename, "r"))==NULL) {
            qfits_error("pixio: cannot open %s", ql->filename);
            return -1;
        }
        /* Go to the start of the image */
        if (fseek(lfile, datastart, SEEK_SET)!=0) {
            qfits_error("pixio: cannot seek %s", ql->filename);
            fclose(lfile);
            return -1;
        }
        /* Go to the start of the zone */
        if (fseek(lfile, (llx-1+(lly-1)*ql->lx)*BYTESPERPIXEL(ql->bitpix), 
                    SEEK_CUR)!=0) {
            qfits_error("pixio: cannot seek %s", ql->filename);
            fclose(lfile);
            return -1;
        }
        
        fptr = (byte*)qfits_malloc(window_size);
       
        /* Only a window is specified */
        if (llx != 1 || lly != 1 || urx != ql->lx || ury != ql->ly) {
            /* Loop on the lines */
            for (i=0; i<ny; i++) {
                /* Read the file */
                dataread=fread(fptr+i*linesize, sizeof(byte), linesize, lfile);
                if (dataread!=linesize) {
                    qfits_free(fptr);
                    fclose(lfile);
                    qfits_error("pixio: cannot read from %s", ql->filename);
                    return -1;
                }
                /* Go to the next line */
                if (fseek(lfile,ql->lx*BYTESPERPIXEL(ql->bitpix)-linesize, 
                           SEEK_CUR)!=0){
                    qfits_error("pixio: cannot seek %s", ql->filename);
                    fclose(lfile);
                    return -1;
                }
            }
            fclose(lfile);
        } else {
        /* The whole image is specified */
            dataread = fread(fptr, sizeof(byte), window_size, lfile);
            fclose(lfile);
            if (dataread!=window_size) {
                qfits_free(fptr);
                qfits_error("pixio: cannot read from %s", ql->filename);
                return -1;
            }
        }
    }
        
    /* Initialize buffer pointers */
    ql->ibuf = NULL;
    ql->fbuf = NULL;
    ql->dbuf = NULL;
	ql->pixbuffer = NULL;

    /*
     * Special cases: mapped file is identical to requested format.
     * This is only possible on big-endian machines, since FITS is
     * big-endian only.
     */
#ifdef WORDS_BIGENDIAN
    if (ql->ptype==PTYPE_FLOAT && ql->bitpix==-32) {
        ql->fbuf = (float*)fptr;
        return 0;
    }
    if (ql->ptype==PTYPE_DOUBLE && ql->bitpix==-64) {
        ql->dbuf = (double*)fptr;
        return 0;
    }
    if (ql->ptype==PTYPE_INT && ql->bitpix==32) {
        ql->ibuf = (int*)fptr;
        return 0;
    }
#endif

    /* General case: fallback to dedicated conversion function */
    switch (ql->ptype) {
        case PTYPE_FLOAT:
        ql->fbuf = qfits_pixin_float(   fptr,
                                        nx * ny,
                                        ql->bitpix,
                                        ql->bscale,
                                        ql->bzero);
		ql->pixbuffer = ql->fbuf;
        break;

        case PTYPE_INT:
        ql->ibuf = qfits_pixin_int( fptr,
                                    nx * ny,
                                    ql->bitpix,
                                    ql->bscale,
                                    ql->bzero);
		ql->pixbuffer = ql->ibuf;
        break;

        case PTYPE_DOUBLE:
        ql->dbuf = qfits_pixin_double(  fptr,
                                        nx * ny,
                                        ql->bitpix,
                                        ql->bscale,
                                        ql->bzero);
		ql->pixbuffer = ql->dbuf;
        break;
    }
   
    if (ql->map) {
        qfits_fdealloc((char*)fptr, datastart, fsize);
    } else {
        qfits_free(fptr);
    }

    if (ql->ibuf==NULL && ql->fbuf==NULL && ql->dbuf==NULL) {
        qfits_error("pixio: error during conversion");
        return -1;
    }
    return 0;
}

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

    buf_out = qfits_malloc(npix * BYTESPERPIXEL(ptype));
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

    buf_out = qfits_malloc(npix * BYTESPERPIXEL(ptype));
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

    buf_out = qfits_malloc(npix * BYTESPERPIXEL(ptype));
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

/*----------------------------------------------------------------------------*/
/**
  @brief    Load a pixel buffer as floats.
  @param    p_source    Pointer to source buffer (as bytes).
  @param    npix        Number of pixels to load.
  @param    bitpix      FITS BITPIX in original file.
  @param    bscale      FITS BSCALE in original file.
  @param    bzero       FITS BZERO in original file.
  @return   1 pointer to a newly allocated buffer of npix floats.

  This function takes in input a pointer to a byte buffer as given
  in the original FITS file (big-endian format). It converts the
  buffer to an array of float (whatever representation is used for
  floats by this platform is used) and returns the newly allocated
  buffer, or NULL if an error occurred.

  The returned buffer must be deallocated using qfits_free().
 */
/*----------------------------------------------------------------------------*/
static float * qfits_pixin_float(
        byte    *   p_source,
        int         npix,
        int         bitpix,
        double      bscale,
        double      bzero)
{
    int         i;
    float   *   baseptr;
    float   *   p_dest;
    double      dpix;
    short       spix;
    int         lpix;
    float       fpix;
    byte        XLpix[8];
    

    baseptr = p_dest = qfits_malloc(npix * sizeof(float));
	if (!baseptr) {
		qfits_error("Failed to allocate a buffer to convert the image to float format.\n");
		return NULL;
	}
    switch (bitpix) {

        case 8:
        /* No swapping needed */
        for (i=0; i<npix; i++) {
            p_dest[i] = (float)((double)p_source[i] * bscale + bzero);
        }
        break;

        case 16:
        for (i=0; i<npix; i++) {
            memcpy(&spix, p_source, 2);
            p_source += 2;
#ifndef WORDS_BIGENDIAN
            spix = qfits_swap_bytes_16(spix);
#endif
            *p_dest++ = (float)(bscale * (double)spix + bzero);
        }
        break;

        case 32:
        for (i=0; i<npix; i++) {
            memcpy(&lpix, p_source, 4);
            p_source += 4;
#ifndef WORDS_BIGENDIAN
            lpix = qfits_swap_bytes_32(lpix);
#endif
            *p_dest++ = (float)(bscale * (double)lpix + bzero);
        }
        break;

        case -32:
        for (i=0; i<npix; i++) {
            memcpy(&lpix, p_source, 4);
            p_source += 4;
#ifndef WORDS_BIGENDIAN
            lpix = qfits_swap_bytes_32(lpix);
#endif
            memcpy(&fpix, &lpix, 4);
            *p_dest++ = (float)((double)fpix * bscale + bzero);
        }
        break;

        case -64:
        for (i=0; i<npix; i++) {
            XLpix[0] = *p_source ++;
            XLpix[1] = *p_source ++;
            XLpix[2] = *p_source ++;
            XLpix[3] = *p_source ++;
            XLpix[4] = *p_source ++;
            XLpix[5] = *p_source ++;
            XLpix[6] = *p_source ++;
            XLpix[7] = *p_source ++;
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(XLpix, 8);
#endif
            //dpix = *((double*)XLpix);
			memcpy(&dpix, XLpix, sizeof(double));
            *p_dest ++ = (float)(bscale * dpix + bzero);
        }
        break;
    }
    return baseptr;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Load a pixel buffer as ints.
  @param    p_source    Pointer to source buffer (as bytes).
  @param    npix        Number of pixels to load.
  @param    bitpix      FITS BITPIX in original file.
  @param    bscale      FITS BSCALE in original file.
  @param    bzero       FITS BZERO in original file.
  @return   1 pointer to a newly allocated buffer of npix ints.

  This function takes in input a pointer to a byte buffer as given
  in the original FITS file (big-endian format). It converts the
  buffer to an array of int (whatever representation is used for
  int by this platform is used) and returns the newly allocated
  buffer, or NULL if an error occurred.

  The returned buffer must be deallocated using qfits_free().
 */
/*----------------------------------------------------------------------------*/
static int * qfits_pixin_int(
        byte    *   p_source,
        int         npix,
        int         bitpix,
        double      bscale,
        double      bzero)
{
    int         i;
    int     *   p_dest;
    int     *   baseptr;
    double      dpix;
    short       spix;
    int         lpix;
    float       fpix;
    byte        XLpix[8];
    
    baseptr = p_dest = qfits_malloc(npix * sizeof(int));
    switch (bitpix) {

        case 8:
        /* No swapping needed */
        for (i=0; i<npix; i++) {
            p_dest[i] = (int)((double)p_source[i] * bscale + bzero);
        }
        break;

        case 16:
        for (i=0; i<npix; i++) {
            memcpy(&spix, p_source, 2);
            p_source += 2;
#ifndef WORDS_BIGENDIAN
            spix = qfits_swap_bytes_16(spix);
#endif
            *p_dest++ = (int)(bscale * (double)spix + bzero);
        }
        break;

        case 32:
        for (i=0; i<npix; i++) {
            memcpy(&lpix, p_source, 4);
            p_source += 4;
#ifndef WORDS_BIGENDIAN
            lpix = qfits_swap_bytes_32(lpix);
#endif
            *p_dest++ = (int)(bscale * (double)lpix + bzero);
        }
        break;

        case -32:
        for (i=0; i<npix; i++) {
            memcpy(&lpix, p_source, 4);
            p_source += 4;
#ifndef WORDS_BIGENDIAN
            lpix = qfits_swap_bytes_32(lpix);
#endif
            memcpy(&fpix, &lpix, 4);
            *p_dest++ = (int)((double)fpix * bscale + bzero);
        }
        break;

        case -64:
        for (i=0; i<npix; i++) {
            XLpix[0] = *p_source ++;
            XLpix[1] = *p_source ++;
            XLpix[2] = *p_source ++;
            XLpix[3] = *p_source ++;
            XLpix[4] = *p_source ++;
            XLpix[5] = *p_source ++;
            XLpix[6] = *p_source ++;
            XLpix[7] = *p_source ++;
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(XLpix, 8);
#endif
            //dpix = *((double*)XLpix);
			memcpy(&dpix, XLpix, sizeof(double));
            *p_dest ++ = (int)(bscale * dpix + bzero);
        }
        break;
    }
    return baseptr;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Load a pixel buffer as doubles.
  @param    p_source    Pointer to source buffer (as bytes).
  @param    npix        Number of pixels to load.
  @param    bitpix      FITS BITPIX in original file.
  @param    bscale      FITS BSCALE in original file.
  @param    bzero       FITS BZERO in original file.
  @return   1 pointer to a newly allocated buffer of npix doubles.

  This function takes in input a pointer to a byte buffer as given
  in the original FITS file (big-endian format). It converts the
  buffer to an array of double (whatever representation is used for
  int by this platform is used) and returns the newly allocated
  buffer, or NULL if an error occurred.

  The returned buffer must be deallocated using qfits_free().
 */
/*----------------------------------------------------------------------------*/
static double * qfits_pixin_double(
        byte    *   p_source,
        int         npix,
        int         bitpix,
        double      bscale,
        double      bzero)
{
    int         i;
    double  *   p_dest;
    double  *   baseptr;
    double      dpix;
    short       spix;
    int         lpix;
    float       fpix;
    byte        XLpix[8];
    

    baseptr = p_dest = qfits_malloc(npix * sizeof(double));
    switch (bitpix) {

        case 8:
        /* No swapping needed */
        for (i=0; i<npix; i++) {
            p_dest[i] = (double)((double)p_source[i] * bscale + bzero);
        }
        break;

        case 16:
        for (i=0; i<npix; i++) {
            memcpy(&spix, p_source, 2);
            p_source += 2;
#ifndef WORDS_BIGENDIAN
            spix = qfits_swap_bytes_16(spix);
#endif
            *p_dest++ = (double)(bscale * (double)spix + bzero);
        }
        break;

        case 32:
        for (i=0; i<npix; i++) {
            memcpy(&lpix, p_source, 4);
            p_source += 4;
#ifndef WORDS_BIGENDIAN
            lpix = qfits_swap_bytes_32(lpix);
#endif
            *p_dest++ = (double)(bscale * (double)lpix + bzero);
        }
        break;

        case -32:
        for (i=0; i<npix; i++) {
            memcpy(&lpix, p_source, 4);
            p_source += 4;
#ifndef WORDS_BIGENDIAN
            lpix = qfits_swap_bytes_32(lpix);
#endif
            memcpy(&fpix, &lpix, 4);
            *p_dest++ = (double)((double)fpix * bscale + bzero);
        }
        break;

        case -64:
        for (i=0; i<npix; i++) {
            XLpix[0] = *p_source ++;
            XLpix[1] = *p_source ++;
            XLpix[2] = *p_source ++;
            XLpix[3] = *p_source ++;
            XLpix[4] = *p_source ++;
            XLpix[5] = *p_source ++;
            XLpix[6] = *p_source ++;
            XLpix[7] = *p_source ++;
#ifndef WORDS_BIGENDIAN
            qfits_swap_bytes(XLpix, 8);
#endif
            //dpix = *((double*)XLpix);
			memcpy(&dpix, XLpix, sizeof(double));
            *p_dest ++ = (double)(bscale * dpix + bzero);
        }
        break;
    }
    return baseptr;
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
