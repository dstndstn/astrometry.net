/* $Id: test_pixio.c,v 1.9 2007/01/10 12:29:58 yjung Exp $
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
 * $Date: 2007/01/10 12:29:58 $
 * $Revision: 1.9 $
 * $Name: qfits-6_2_0 $
 */

/*---------------------------------------------------------------------------
                                   Includes
 ---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "qfits_header.h"
#include "qfits_image.h"
#include "qfits_md5.h"
#include "qfits_memory.h"

/* Get test data sets */
#include "pixset.h"

/*---------------------------------------------------------------------------
                                   Defines
 ---------------------------------------------------------------------------*/

/* 5 possible values for FITS BITPIX */
#define FITS_NBITPIX    5

/* Name of temporary directory */
#define TEST_DIRNAME    "pixio_dat"

/*---------------------------------------------------------------------------
                               Static variables
 ---------------------------------------------------------------------------*/
/* List of all FITS BITPIX possible values */
static int fits_bitpix[FITS_NBITPIX] = {8, 16, 32, -32, -64};

/* List of all DATAMD5 signatures for the generated files */
static char md5sigs[FITS_NBITPIX][33] =
{
    "980c2e1bd6f6418e000b4f2bba665e69",
    "df8420dce7a4e6b3edb23171b9112d1d",
    "3b50d1cf34f3f5687c8f00ebbecc74b4",
    "f502ad23f6b250cb0f9f333c50314751",
    "d77c59b732df2a65c036dbf2fb180ad3"
};

/*---------------------------------------------------------------------------
                              Function codes
 ---------------------------------------------------------------------------*/

/* Print out a comment */
static void say(char * fmt, ...)
{
    va_list ap ;
    fprintf(stdout, "qtest:\t\t");
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fprintf(stdout, "\n");
}

/* Print out an error message */
static void fail(char * fmt, ...)
{
    va_list ap ;
    fprintf(stderr, "qtest: error: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
}


/* Dump data to a test file */
static int test_dumpfile(qfitsdumper * qd)
{
    qfits_header    *   qh ;
    FILE            *   out ;
    char                s_out[8];

    qh = qfits_header_default();
    qfits_header_add(qh, "NAXIS", "2", "Number of axes", NULL);
    sprintf(s_out, "%d", TEST_LX);
    qfits_header_add(qh, "NAXIS1", s_out, "size in x", NULL);
    sprintf(s_out, "%d", TEST_LY);
    qfits_header_add(qh, "NAXIS2", s_out, "size in y", NULL);
    sprintf(s_out, "%d", qd->out_ptype);
    qfits_header_add(qh, "BITPIX", s_out, "Bits per pixel", NULL);

    out = fopen(qd->filename, "w");
    if (out==NULL) {
        fail("cannot create test file");
        qfits_header_destroy(qh);
        return 1 ;
    }
    if (qfits_header_dump(qh, out)!=0) {
        fail("cannot dump header");
        qfits_header_destroy(qh);
        return 1 ;
    }
    fclose(out);
    qfits_header_destroy(qh);

    if (qfits_pixdump(qd)!=0) {
        fail("cannot dump data section");
        remove(qd->filename);
        return 1 ;
    }
    qfits_zeropad(qd->filename);
    return 0 ;
}


/*
 * For this first test, the idea is to have the same array of pixels
 * defined three times: as int, as float, and as double. Pixels in the
 * three arrays contain strictly the same values but under different data
 * types.
 * All arrays are successively dumped to all possible FITS pixel types
 * (BITPIX), yielding 5 data types per input array. A first check is to
 * ensure that all input pixel types can be dumped to all possible BITPIX
 * values.
 * The second thing to test is that FITS files with the same BITPIX have
 * the same MD5 data signature, i.e. that dumping from two different data
 * types yields the same FITS file.
 */

static int test_dumpdatasets(void)
{
    int             err ;
    char            filename[1024] ;
    char        *   ptype_name ;
    qfitsdumper     qd ;
    int             i ;
    const char  *   datamd5 ;

    err = 0 ;
    qd.npix  = TEST_NPIX ;

    say("-----> Data dump tests");

    /* Create working directory to avoid flooding the current one */
    say("creating directory %s", TEST_DIRNAME);
    if (mkdir(TEST_DIRNAME, 0777)!=0) {
        fail("cannot create directory '%s'", TEST_DIRNAME);
        return 1 ;
    }
    /* Set pointers to the relevant pixel buffers */
    qd.ibuf = intpix_set ;
    qd.fbuf = floatpix_set ;
    qd.dbuf = doublepix_set ;

    /* Create file from the int array with all possible FITS types */
    qd.ptype = PTYPE_INT ;
    ptype_name = "int" ;

    /* Loop on all FITS types */
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        qd.out_ptype = fits_bitpix[i];
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                qd.out_ptype);
        qd.filename = filename ;
        say("dumping data % 6s -> BITPIX=% 3d", ptype_name, qd.out_ptype);
        if (test_dumpfile(&qd)!=0) {
            err++ ;
        } else {
            /* Verify MD5 signature */
            datamd5 = qfits_datamd5(filename);
            if (strcmp(datamd5, md5sigs[i])) {
                fail("MD5 signature does not match");
                err++ ;
            }
        }
    }

    /* Create file from the float array with all possible FITS types */
    qd.ptype = PTYPE_FLOAT ;
    ptype_name = "float" ;

    /* Loop on all FITS types */
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        qd.out_ptype = fits_bitpix[i];
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                qd.out_ptype);
        qd.filename = filename ;
        say("dumping data % 6s -> BITPIX=% 3d", ptype_name, qd.out_ptype);
        if (test_dumpfile(&qd)!=0) {
            err++ ;
        } else {
            /* Verify MD5 signature */
            datamd5 = qfits_datamd5(filename);
            if (strcmp(datamd5, md5sigs[i])) {
                fail("MD5 signature does not match");
                err++ ;
            }
        }
    }

    /* Create file from the double array with all possible FITS types */
    qd.ptype = PTYPE_DOUBLE ;
    ptype_name = "double" ;

    /* Loop on all FITS types */
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        qd.out_ptype = fits_bitpix[i];
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                qd.out_ptype);
        qd.filename = filename ;
        say("dumping data % 6s -> BITPIX=% 3d", ptype_name, qd.out_ptype);
        if (test_dumpfile(&qd)!=0) {
            err++ ;
        } else {
            /* Verify MD5 signature */
            datamd5 = qfits_datamd5(filename);
            if (strcmp(datamd5, md5sigs[i])) {
                fail("MD5 signature does not match");
                err++ ;
            }
        }
    }

    return err ;
}

/*
 * Simple comparison function for the int array
 */
static int check_intpix(int * arr, int bitpix)
{
    int i ;
    int pix ;
    int pixerr ;

    pixerr=0 ;
    for (i=0 ; i<TEST_NPIX ; i++) {
        pix = intpix_set[i] ;
        if (bitpix==8) {
            if (intpix_set[i]<0)
                pix=0 ;
            else if (intpix_set[i]>255)
                pix=255 ;
        } else if (bitpix==16) {
            if (intpix_set[i]<-32768) 
                pix = -32768 ;
            else if (intpix_set[i]>32767)
                pix = 32767 ;
        }
        if (arr[i]!=pix)
            pixerr++ ;
    }
    return pixerr ;
}

/*
 * Simple comparison function for the float array
 */
static int check_floatpix(float * arr, int bitpix)
{
    int i ;
    float pix ;
    int pixerr ;

    pixerr=0 ;
    for (i=0 ; i<TEST_NPIX ; i++) {
        pix = intpix_set[i] ;
        if (bitpix==8) {
            if (intpix_set[i]<0)
                pix=0 ;
            else if (intpix_set[i]>255)
                pix=255 ;
        } else if (bitpix==16) {
            if (intpix_set[i]<-32768) 
                pix = -32768 ;
            else if (intpix_set[i]>32767)
                pix = 32767 ;
        }
        if (fabs((double)(arr[i]-pix))>1e-7)
            pixerr++ ;
    }
    return pixerr ;
}

/*
 * Simple comparison function for the double array
 */
static int check_doublepix(double * arr, int bitpix)
{
    int i ;
    double pix ;
    int pixerr ;

    pixerr=0 ;
    for (i=0 ; i<TEST_NPIX ; i++) {
        pix = intpix_set[i] ;
        if (bitpix==8) {
            if (intpix_set[i]<0)
                pix=0 ;
            else if (intpix_set[i]>255)
                pix=255 ;
        } else if (bitpix==16) {
            if (intpix_set[i]<-32768) 
                pix = -32768 ;
            else if (intpix_set[i]>32767)
                pix = 32767 ;
        }
        if (fabs(arr[i]-pix)>1e-7)
            pixerr++ ;
    }
    return pixerr ;
}

/*
 * This second test tries to read back the data written in the first test,
 * and compares the values with what was initially sent to the files.
 */
static int test_readdatasets(void)
{
    char    filename[1024];
    int     i ;
    qfitsloader ql ;
    int     err, pixerr ;
    char *  ptype_name ;

    err=0 ;

    say("-----> Data read tests");
    ql.xtnum = 0 ;
    ql.pnum  = 0 ;
    ql.map   = 0 ;
    
    /* Test reading back from all files as int buffer */
    ptype_name = "int" ;
    ql.ptype   = PTYPE_INT ;

    say("-----> int reading tests");
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                fits_bitpix[i]);
        ql.filename = filename ;
        say("reading data from file: %s", filename);
        /* Initialize loader */
        if (qfitsloader_init(&ql)!=0) {
            fail("cannot initialize loader on file %s", filename);
            err++ ;
        } else {
            /* Check image size */
            if (ql.lx!=TEST_LX || ql.ly!=TEST_LY) {
                fail("wrong image size: %dx%d should be %dx%d",
                     ql.lx,
                     ql.ly,
                     TEST_LX,
                     TEST_LY);
                err++ ;
            }
            /* Try loading pixel buffer */
            if (qfits_loadpix(&ql)!=0) {
                fail("cannot load pixel buffer");
                err++ ;
            } else {
                /* Verify pixel values */
                pixerr = check_intpix(ql.ibuf, fits_bitpix[i]);
                if (pixerr) {
                    fail("%d pixels have errors", pixerr);
                    err += pixerr ;
                }
                qfits_free(ql.ibuf) ;
            }
        }
    }

    /* Test reading back from all files as int buffer */
    ptype_name = "float" ;
    ql.ptype   = PTYPE_FLOAT ;

    say("-----> float reading tests");
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                fits_bitpix[i]);
        ql.filename = filename ;
        say("reading data from file: %s", filename);
        /* Initialize loader */
        if (qfitsloader_init(&ql)!=0) {
            fail("cannot initialize loader on file %s", filename);
            err++ ;
        } else {
            /* Check image size */
            if (ql.lx!=TEST_LX || ql.ly!=TEST_LY) {
                fail("wrong image size: %dx%d should be %dx%d",
                     ql.lx,
                     ql.ly,
                     TEST_LX,
                     TEST_LY);
                err++ ;
            }
            /* Try loading pixel buffer */
            if (qfits_loadpix(&ql)!=0) {
                fail("cannot load pixel buffer");
                err++ ;
            } else {
                /* Verify pixel values */
                pixerr = check_floatpix(ql.fbuf, fits_bitpix[i]);
                if (pixerr) {
                    fail("%d pixels have errors", pixerr);
                    err += pixerr ;
                }
                qfits_free(ql.fbuf) ;
            }
        }
    }

    /* Test reading back from all files as double buffer */
    ptype_name = "double" ;
    ql.ptype   = PTYPE_DOUBLE ;

    say("-----> double reading tests");
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                fits_bitpix[i]);
        ql.filename = filename ;
        say("reading data from file: %s", filename);
        /* Initialize loader */
        if (qfitsloader_init(&ql)!=0) {
            fail("cannot initialize loader on file %s", filename);
            err++ ;
        } else {
            /* Check image size */
            if (ql.lx!=TEST_LX || ql.ly!=TEST_LY) {
                fail("wrong image size: %dx%d should be %dx%d",
                     ql.lx,
                     ql.ly,
                     TEST_LX,
                     TEST_LY);
                err++ ;
            }
            /* Try loading pixel buffer */
            if (qfits_loadpix(&ql)!=0) {
                fail("cannot load pixel buffer");
                err++ ;
            } else {
                /* Verify pixel values */
                pixerr = check_doublepix(ql.dbuf, fits_bitpix[i]);
                if (pixerr) {
                    fail("%d pixels have errors", pixerr);
                    err += pixerr ;
                }
                qfits_free(ql.dbuf) ;
            }
        }
    }

    /* Test reading back a window from all files as double buffer */
    ptype_name = "double" ;
    ql.ptype   = PTYPE_DOUBLE ;

    say("-----> window double reading tests");
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                fits_bitpix[i]);
        ql.filename = filename ;
        say("reading data from file: %s", filename);
        /* Initialize loader */
        if (qfitsloader_init(&ql)!=0) {
            fail("cannot initialize loader on file %s", filename);
            err++ ;
        } else {
            /* Check image size */
            if (ql.lx!=TEST_LX || ql.ly!=TEST_LY) {
                fail("wrong image size: %dx%d should be %dx%d",
                     ql.lx,
                     ql.ly,
                     TEST_LX,
                     TEST_LY);
                err++ ;
            }
            /* Try loading pixel buffer */
            if (qfits_loadpix_window(&ql, 4, 28, 4, 28)!=0) {
                fail("cannot load pixel buffer");
                err++ ;
            } else {
                /* Verify pixel values */
                qfits_free(ql.dbuf) ;
            }
        }
    }
    return err ;
}

static void test_cleanup(void)
{
    int i ;
    char filename[1024];
    char * ptype_name ;

    /* Delete all data files */
    say("cleaning up data files...");
    for (i=0 ; i<FITS_NBITPIX ; i++) {
        ptype_name = "int" ;
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                fits_bitpix[i]);
        remove(filename);

        ptype_name = "float" ;
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                fits_bitpix[i]);
        remove(filename);

        ptype_name = "double" ;
        sprintf(filename,
                "%s/pixio_%s_%d.fits",
                TEST_DIRNAME,
                ptype_name,
                fits_bitpix[i]);
        remove(filename);
    }
    say("removing directory '%s'", TEST_DIRNAME);
    rmdir(TEST_DIRNAME);
}

int main(int argc, char * argv[])
{
    int err ;
    int i ;
    int keep ;

    keep=0 ;
    for (i=1 ; i<argc ; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("use: %s [options]\n"
                   "options are:\n"
                   "\n"
                   "\t-k       To keep test data\n"
                   "\n"
                   "\n", argv[0]);
            return 1 ;
        } else if (!strcmp(argv[i], "-k")) {
            keep=1 ;
        } else {
            printf("ignored option: %s\n", argv[i]);
        }
    }

    err=0 ;
    err += test_dumpdatasets();
    err += test_readdatasets();

    if (!keep) {
        test_cleanup();
    }
    fprintf(stderr, "total error(s): %d\n", err);
    return err ;
}

