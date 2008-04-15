/* $Id: test_qfits.c,v 1.15 2007/01/10 12:29:58 yjung Exp $
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
 * $Revision: 1.15 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>

#include "qfits_header.h"
#include "qfits_image.h"
#include "qfits_tools.h"
#include "qfits_rw.h"
#include "qfits_md5.h"
#include "qfits_memory.h"

/*-----------------------------------------------------------------------------
                                   Define
 -----------------------------------------------------------------------------*/

#define QFITSTEST1             "QFITS.fits"
#define QFITSTEST2             "QFITS_ext.fits"

#define REFSIG                "6569daba7b124febfa0cd7813f555774"

/*-----------------------------------------------------------------------------
                                   Functions
 -----------------------------------------------------------------------------*/

static float float_array_orig[] =
{
    1.0, 2.0, 0.0, -1.0, -2.0, 1e-4, -1e-4, 1e-6, -1e-6,
    1.2345678, 3.1415926535, 19.71
};

static int int_array_orig[] =
{
    -32768, -16384, -8192, -4096, -1023, 0,
     1023, 2048, 8191, 16387, 32767, 65536
};

static double double_array_orig[] =
{
    1.0, 2.0, 0.0, -1.0, -2.0, 1e-4, -1e-4, 1e-6, -1e-6,
    1.2345678, 3.1415926535, 19.71
};


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

/*---------------------------------------------------------------------------
                                main
 ---------------------------------------------------------------------------*/

int test_qfits_header_sort(void)
{
    qfits_header    *    qh ;
    
    /* Create header */
    qh = qfits_header_new();
    qfits_header_append(qh, "EXTEND", "value", "comment", NULL);
    qfits_header_append(qh, "COMMENT", "value", "comment", NULL);
    qfits_header_append(qh, "TFORM", "value", "comment", NULL);
    qfits_header_append(qh, "BITPIX", "value", "comment", NULL);
    qfits_header_append(qh, "NAXIS", "value", "comment", NULL);
    qfits_header_append(qh, "NAXIS2", "value", "comment", NULL);
    qfits_header_append(qh, "NAXIS1", "value", "comment", NULL);

    /* Dump the header */
    qfits_header_dump(qh, NULL) ;

    /* Sort the header */
    qfits_header_sort(&qh) ;

    /* Dump the header */
    qfits_header_dump(qh, NULL) ;
    
    /* Destroy the header */
    qfits_header_destroy(qh) ;
    return 0 ;
}

int test_qfits_filecreate(char * filename)
{
    qfits_header    *    qh ;
    FILE            *    out ;
    qfitsdumper            qd ;
    int                    i ;

    say("-----> Header creation");
    say("Creating blank header");
    qh = qfits_header_new();
    if (qh==NULL) {
        fail("qfits_header_new() failed");
        return 1 ;
    }
    say("Destroying blank header");
    /* Destroy header now */
    qfits_header_destroy(qh);

    /* Create minimal header (SIMPLE/END) */
    say("Creating minimal header");
    qh = qfits_header_default();
    if (qh==NULL) {
        fail("qfits_header_default() failed");
        return 1 ;
    }

    say("Inserting primary keywords");
    /* Insert XTENSION marker */
    qfits_header_add(qh, "EXTEND", "T", "xtension might be present", NULL);

    /* Insert a string */
    qfits_header_add(qh, "KEY01", "value01", "comment 01", NULL);
    /* Insert an int */
    qfits_header_add(qh, "KEY02", "2", "comment 02", NULL);
    /* Insert a double */
    qfits_header_add(qh, "KEY03", "3.0", "comment 03", NULL);
    /* Insert a complex */
    qfits_header_add(qh, "KEY04", "4.0 4.2", "comment 04", NULL);
    /* Insert a boolean */
    qfits_header_add(qh, "KEY05", "T", "comment 05", NULL);


    say("Inserting history keywords");
    /* Insert HISTORY keys */
    qfits_header_add(qh, "HISTORY", "1 history field", NULL, NULL);
    qfits_header_add(qh, "HISTORY", "2 history field", NULL, NULL);
    qfits_header_add(qh, "HISTORY", "3 history field", NULL, NULL);
    qfits_header_add(qh, "HISTORY", "4 history field", NULL, NULL);

    say("Inserting comment keywords");
    /* Insert COMMENT keys */
    qfits_header_add(qh, "COMMENT", "1 comment field", NULL, NULL);
    qfits_header_add(qh, "COMMENT", "2 comment field", NULL, NULL);
    qfits_header_add(qh, "COMMENT", "3 comment field", NULL, NULL);
    qfits_header_add(qh, "COMMENT", "4 comment field", NULL, NULL);

    say("Inserting hierarch keywords");
    /* Insert HIERARCH ESO keys in reverse DICB order */
    qfits_header_add(qh, "HIERARCH ESO NULL A", "0.0", "not DICB", NULL);
    qfits_header_add(qh, "HIERARCH ESO NULL B", "0.0", "not DICB", NULL);
    qfits_header_add(qh, "HIERARCH ESO NULL C", "0.0", "not DICB", NULL);

    qfits_header_add(qh, "PRO.A", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "PRO.B", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "PRO.C", "0.0", "DICB compliant", NULL);

    qfits_header_add(qh, "HIERARCH ESO LOG A", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "HIERARCH ESO LOG B", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "HIERARCH ESO LOG C", "0.0", "DICB compliant", NULL);

    qfits_header_add(qh, "INS.A", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "INS.B", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "INS.C", "0.0", "DICB compliant", NULL);

    qfits_header_add(qh, "HIERARCH ESO TEL A", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "HIERARCH ESO TEL B", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "HIERARCH ESO TEL C", "0.0", "DICB compliant", NULL);

    qfits_header_add(qh, "GEN.A", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "GEN.B", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "GEN.C", "0.0", "DICB compliant", NULL);

    qfits_header_add(qh, "HIERARCH ESO TPL A", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "HIERARCH ESO TPL B", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "HIERARCH ESO TPL C", "0.0", "DICB compliant", NULL);

    qfits_header_add(qh, "OBS.A", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "OBS.B", "0.0", "DICB compliant", NULL);
    qfits_header_add(qh, "OBS.C", "0.0", "DICB compliant", NULL);

    say("Inserting mandatory keywords");
    /* Insert mandatory keys in reverse order */
    qfits_header_add(qh, "NAXIS2", "10", "NAXIS2 comment", NULL);
    qfits_header_add(qh, "NAXIS1", "11", "NAXIS1 comment", NULL);
    qfits_header_add(qh, "NAXIS",  "2", "NAXIS comment", NULL);
    qfits_header_add(qh, "BITPIX",  "-32", "BITPIX comment", NULL);

    /* Dump header to file */
    say("Opening file for output");
    out = fopen(filename, "w");
    if (out==NULL) {
        fail("cannot create test file");
        qfits_header_destroy(qh);
        return 1 ;
    }
    say("Dumping header to file");
    if (qfits_header_dump(qh, out)!=0) {
        fail("cannot dump header");
        qfits_header_destroy(qh);
        return 1 ;
    }
    say("Destroying built header");
    qfits_header_destroy(qh);
    fclose(out);

    say("-----> Dumping pixels");
    /* Allocate data segment and save it to FITS file */
    qd.fbuf = qfits_malloc(11 * 10 * sizeof(float));
    for (i=0 ; i<(11*10) ; i++) {
        qd.fbuf[i]=i*0.2 ;
    }

    qd.filename  = filename ;
    qd.npix      = 11 * 10 ;
    qd.ptype     = PTYPE_FLOAT ;
    qd.out_ptype = -32 ;

    if (qfits_pixdump(&qd)!=0) {
        fail("cannot save data to test file");
        qfits_free(qd.fbuf);
        return 1 ;
    }
    qfits_free(qd.fbuf);

    /* Zero-pad the output file */
    qfits_zeropad(filename);
    return 0 ;
}

int check_key(qfits_header * qh, char * key, char * expval)
{
    char * val ;
    int    err=0 ;

    val = qfits_header_getstr(qh, key);
    if (val==NULL) {
        fail("missing key in header");
        err++ ;
    } else {
        val = qfits_pretty_string(val);
        if (strcmp(val, expval)) {
            fail("wrong value for key in header");
            err++ ;
        }
    }
    return err ;
}

int test_qfitsheader_read(char * filename)
{
    qfits_header    *    qh ;
    char            *    val ;
    int                    err ;
    int                    keytype ;

    err=0 ;
    say("-----> Header reading test");
    /* Read header from source */
    say("Reading header from file");
    qh = qfits_header_read(filename);
    if (qh==NULL) {
        fail("cannot read test file");
        return 1 ;
    }
    say("Querying mandatory keys");
    err += check_key(qh, "SIMPLE", "T");
    err += check_key(qh, "NAXIS", "2");
    err += check_key(qh, "NAXIS1", "11");
    err += check_key(qh, "NAXIS2", "10");
    err += check_key(qh, "BITPIX", "-32");

    say("Querying base keys");
    err += check_key(qh, "KEY01", "value01");
    err += check_key(qh, "KEY02", "2");
    err += check_key(qh, "KEY03", "3.0");
    err += check_key(qh, "KEY04", "4.0 4.2");
    err += check_key(qh, "KEY05", "T");

    say("Checking key types");
    val = qfits_header_getstr(qh, "KEY01");
    keytype = qfits_get_type(val);
    if (keytype!=QFITS_STRING) {
        printf("val=[%s] type is %d\n", val, keytype);
        fail("wrong identified type for KEY01 (string)");
        err++;
    }
    val = qfits_header_getstr(qh, "KEY02");
    keytype = qfits_get_type(val);
    if (keytype!=QFITS_INT) {
        fail("wrong identified type for KEY02 (int)");
        err++;
    }
    val = qfits_header_getstr(qh, "KEY03");
    keytype = qfits_get_type(val);
    if (keytype!=QFITS_FLOAT) {
        fail("wrong identified type for KEY03 (float)");
        err++;
    }
    val = qfits_header_getstr(qh, "KEY04");
    keytype = qfits_get_type(val);
    if (keytype!=QFITS_COMPLEX) {
        fail("wrong identified type for KEY04 (complex)");
        err++;
    }
    val = qfits_header_getstr(qh, "KEY05");
    keytype = qfits_get_type(val);
    if (keytype!=QFITS_BOOLEAN) {
        fail("wrong identified type for KEY05 (boolean)");
        err++;
    }

    say("Querying hierarch keys");
    err += check_key(qh, "HIERARCH ESO PRO A", "0.0");
    err += check_key(qh, "PRO.B", "0.0");
    err += check_key(qh, "pro.c", "0.0");

    err += check_key(qh, "ins.a", "0.0");
    err += check_key(qh, "ins.b", "0.0");
    err += check_key(qh, "ins.c", "0.0");

    err += check_key(qh, "gen.a", "0.0");
    err += check_key(qh, "gen.b", "0.0");
    err += check_key(qh, "gen.c", "0.0");

    err += check_key(qh, "obs.a", "0.0");
    err += check_key(qh, "obs.b", "0.0");
    err += check_key(qh, "obs.c", "0.0");

    err += check_key(qh, "tpl.a", "0.0");
    err += check_key(qh, "tpl.b", "0.0");
    err += check_key(qh, "tpl.c", "0.0");

    err += check_key(qh, "tel.a", "0.0");
    err += check_key(qh, "tel.b", "0.0");
    err += check_key(qh, "tel.c", "0.0");

    err += check_key(qh, "log.a", "0.0");
    err += check_key(qh, "log.b", "0.0");
    err += check_key(qh, "log.c", "0.0");

    err += check_key(qh, "null.a", "0.0");
    err += check_key(qh, "null.b", "0.0");
    err += check_key(qh, "null.c", "0.0");

    say("Removing keys");

    qfits_header_del(qh, "PRO.A");
    qfits_header_del(qh, "pro.b");
    qfits_header_del(qh, "HIERARCH ESO PRO C");

    if (qfits_header_getstr(qh, "HIERARCH ESO PRO A")!=NULL) 
        err ++ ;
    if (qfits_header_getstr(qh, "PRO.B")!=NULL) 
        err ++ ;
    if (qfits_header_getstr(qh, "pro.c")!=NULL) 
        err ++ ;

    say("Modifying keys");

    qfits_header_destroy(qh);
    return err ;
}

int test_qfitsheader_browse(char * filename)
{
    qfits_header    *    qh ;
    char key[80], val[80], com[80] ;
    int     i ;
    int  err ;

    say("-----> Header browsing test");
    /* Read header from source */
    say("Reading header from file");
    qh = qfits_header_read(filename);
    if (qh==NULL) {
        fail("cannot read test file");
        return 1 ;
    }

    err=0 ;
    for (i=0 ; i<qh->n ; i++) {
        if (qfits_header_getitem(qh, i, key, val, com, NULL)!=0) {
            fail("cannot read header item");
            err++ ;
        }
    }
    qfits_header_destroy(qh);
    return err ;
}


int test_qfitsdata_load(char * filename)
{
    qfitsloader    ql ;
    int    i ;
    int    err ;
    float diff ;

    err=0 ;
    say("-----> Data loading test");
    ql.filename    = filename ;
    ql.xtnum    = 0 ;
    ql.pnum        = 0 ;
    ql.ptype    = PTYPE_FLOAT ;
    ql.map      = 1 ;

    say("Initializing loader");
    if (qfitsloader_init(&ql)!=0) {
        fail("cannot initialize loader on test file");
        return 1 ;
    }
    if (ql.lx != 11) {
        fail("wrong size in X");
        err++ ;
    }
    if (ql.ly != 10) {
        fail("wrong size in Y");
        err++ ;
    }

    say("Loading pixel buffer");
    if (qfits_loadpix(&ql)!=0) {
        fail("cannot load data from test file");
        return 1 ;
    }

    for (i=0 ; i<(11*10) ; i++) {
        diff = ql.fbuf[i] - (float)i * 0.2 ;
        if (diff>1e-4) {
            fail("diff in pix value");
            err++ ;
        }
    }
    qfits_free(ql.fbuf);
    return err ;

}

int test_qfits_filecreate_ext(char * filename)
{
    qfits_header*    qh ;
    qfitsdumper        qd ;
    FILE        *    out ;
    const char  *    sig ;

    say("-----> File with multiple extensions");
    /* Create minimal FITS header for main */
    say("Creating default header");
    qh = qfits_header_default() ;
    if (qh==NULL) {
        fail("cannot create default header");
        return 1 ;
    }
    qfits_header_add(qh, "BITPIX", "8", "no data in main section", NULL);
    qfits_header_add(qh, "NAXIS", "0", "no data in main section", NULL);
    qfits_header_add(qh, "EXTEND", "T", "Extensions are present", NULL);

    say("Dumping header to test file");
    out = fopen(filename, "w");
    if (out==NULL) {
        fail("cannot create test file");
        qfits_header_destroy(qh);
        return 1 ;
    }
    qfits_header_dump(qh, out);
    fclose(out);
    qfits_header_destroy(qh);

    say("Creating first extension with float pixels");
    qh = qfits_header_new();
    if (qh==NULL) {
        fail("cannot create extension header 1");
        return 1 ;
    }
    qfits_header_append(qh, "XTENSION", "T", "Ext 1", NULL);
    qfits_header_append(qh, "BITPIX", "-32", "bpp", NULL);
    qfits_header_append(qh, "NAXIS", "2", "axes", NULL);
    qfits_header_append(qh, "NAXIS1", "6", "size in x", NULL);
    qfits_header_append(qh, "NAXIS2", "2", "size in y", NULL);
    qfits_header_append(qh, "END", NULL, NULL, NULL);

    say("Dumping ext header 1 to test file");
    out = fopen(filename, "a");
    if (out==NULL) {
        fail("cannot append to test file");
        qfits_header_destroy(qh);
        return 1 ;
    }
    qfits_header_dump(qh, out);
    fclose(out);
    qfits_header_destroy(qh);

    say("Dumping float array");

    qd.filename = filename ;
    qd.npix      = 12 ;
    qd.ptype     = PTYPE_FLOAT ;
    qd.out_ptype = -32 ;
    qd.fbuf         = float_array_orig ;

    if (qfits_pixdump(&qd)!=0) {
        fail("cannot save data to test file");
        qfits_free(qd.fbuf);
        return 1 ;
    }
    /* Zero-pad the output file */
    qfits_zeropad(filename);

    say("Creating second extension with int pixels");
    qh = qfits_header_new();
    if (qh==NULL) {
        fail("cannot create extension header 1");
        return 1 ;
    }
    qfits_header_append(qh, "XTENSION", "T", "Ext 1", NULL);
    qfits_header_append(qh, "BITPIX", "32", "bpp", NULL);
    qfits_header_append(qh, "NAXIS", "2", "axes", NULL);
    qfits_header_append(qh, "NAXIS1", "6", "size in x", NULL);
    qfits_header_append(qh, "NAXIS2", "2", "size in y", NULL);
    qfits_header_append(qh, "END", NULL, NULL, NULL);

    say("Dumping ext header 2 to test file");
    out = fopen(filename, "a");
    if (out==NULL) {
        fail("cannot append to test file");
        qfits_header_destroy(qh);
        return 1 ;
    }
    qfits_header_dump(qh, out);
    fclose(out);
    qfits_header_destroy(qh);

    say("Dumping int array");

    qd.filename = filename ;
    qd.npix      = 12 ;
    qd.ptype     = PTYPE_INT ;
    qd.out_ptype = 32 ;
    qd.ibuf         = int_array_orig ;

    if (qfits_pixdump(&qd)!=0) {
        fail("cannot save data to test file");
        qfits_free(qd.fbuf);
        return 1 ;
    }
    /* Zero-pad the output file */
    qfits_zeropad(filename);

    say("Creating third extension with double pixels");
    qh = qfits_header_new();
    if (qh==NULL) {
        fail("cannot create extension header 3");
        return 1 ;
    }
    qfits_header_append(qh, "XTENSION", "T", "Ext 1", NULL);
    qfits_header_append(qh, "BITPIX", "-64", "bpp", NULL);
    qfits_header_append(qh, "NAXIS", "2", "axes", NULL);
    qfits_header_append(qh, "NAXIS1", "6", "size in x", NULL);
    qfits_header_append(qh, "NAXIS2", "2", "size in y", NULL);
    qfits_header_append(qh, "END", NULL, NULL, NULL);

    say("Dumping ext header 3 to test file");
    out = fopen(filename, "a");
    if (out==NULL) {
        fail("cannot append to test file");
        qfits_header_destroy(qh);
        return 1 ;
    }
    qfits_header_dump(qh, out);
    fclose(out);
    qfits_header_destroy(qh);

    say("Dumping double array");

    qd.filename = filename ;
    qd.npix      = 12 ;
    qd.ptype     = PTYPE_DOUBLE ;
    qd.out_ptype = -64 ;
    qd.dbuf         = double_array_orig ;

    if (qfits_pixdump(&qd)!=0) {
        fail("cannot save data to test file");
        qfits_free(qd.fbuf);
        return 1 ;
    }
    /* Zero-pad the output file */
    qfits_zeropad(filename);

    /* Get MD5 for the test file */
    sig = qfits_datamd5(filename);
    if (strcmp(sig, REFSIG)) {
        fail("test file signature does not match");
        return 1 ;
    }
    say("File DATAMD5 signature is Ok");

    return 0 ;
}

int main(int argc, char * argv[])
{
    int    err ;

    err=0 ;

    /* Test on simple FITS file */
    err += test_qfits_header_sort();
    err += test_qfits_filecreate(QFITSTEST1);
    err += test_qfitsheader_read(QFITSTEST1);
    err += test_qfitsheader_browse(QFITSTEST1);
    err += test_qfitsdata_load(QFITSTEST1);
    remove(QFITSTEST1);

    /* Test on FITS file with extensions */
    err += test_qfits_filecreate_ext(QFITSTEST2);
    remove(QFITSTEST2);

    fprintf(stderr, "total error(s): %d\n", err);
    return err ;
}
