/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_rw.c,v 1.11 2006/02/23 11:08:59 yjung Exp $
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
 * $Date: 2006/02/23 11:08:59 $
 * $Revision: 1.11 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#include "qfits_rw.h"

#include "qfits_cache.h"
#include "qfits_card.h"
#include "qfits_std.h"
#include "qfits_tools.h"
#include "qfits_error.h"
#include "qfits_memory.h"

/*-----------------------------------------------------------------------------
                                Define
 -----------------------------------------------------------------------------*/

/* FITS magic number */
#define FITS_MAGIC            "SIMPLE"
/* Size of the FITS magic number */
#define FITS_MAGIC_SZ        6

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_rw    FITS header reading/writing
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                            Function codes
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Read a FITS header from a file to an internal structure.
  @param    filename    Name of the file to be read
  @return   Pointer to newly allocated qfits_header or NULL in error case.

  This function parses a FITS (main) header, and returns an allocated
  qfits_header object. The qfits_header object contains a linked-list of
  key "tuples". A key tuple contains:

  - A keyword
  - A value
  - A comment
  - An original FITS line (as read from the input file)

  Direct access to the structure is not foreseen, use accessor
  functions in fits_h.h

  Value, comment, and original line might be NULL pointers.
 */
/*----------------------------------------------------------------------------*/
qfits_header * qfits_header_read(const char * filename)
{
    /* Forward job to readext */
    return qfits_header_readext(filename, 0);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Read a FITS header from a 'hdr' file.
  @param    filename    Name of the file to be read
  @return   Pointer to newly allocated qfits_header or NULL in error case

  This function parses a 'hdr' file, and returns an allocated qfits_header 
  object. A hdr file is an ASCII format were the header is written with a 
  carriage return after each line. The command dfits typically displays 
  a hdr file.
 */
/*----------------------------------------------------------------------------*/
qfits_header * qfits_header_read_hdr(const char * filename)
{
	char getval_buf[FITS_LINESZ+1];
	char getkey_buf[FITS_LINESZ+1];
	char getcom_buf[FITS_LINESZ+1];
    qfits_header    *   hdr;
    FILE            *   in;
    char                line[81];
    char            *   key,
                    *   val,
                    *   com;
    int                 i, j;

    /* Check input */
    if (filename==NULL) return NULL;

    /* Initialise */
    key = val = com = NULL; 

    /* Open the file */
    if ((in=fopen(filename, "r"))==NULL) {
        qfits_error("cannot read file \"%s\"", filename);
        return NULL;
    }
    
    /* Create the header */
    hdr = qfits_header_new();
    
    /* Go through the file */
    while (fgets(line, 81, in)!=NULL) {
        for (i=0; i<81; i++) {
            if (line[i] == '\n') {
                for (j=i; j<81; j++) line[j] = ' ';
                line[80] = '\0';
                break;
            }
        }
        if (!strcmp(line, "END")) {
            line[3] = ' ';
            line[4] = '\0';
        }
        
        /* Rule out blank lines */
        if (!is_blank_line(line)) {

            /* Get key, value, comment for the current line */
            key = qfits_getkey_r(line, getkey_buf);
            val = qfits_getvalue_r(line, getval_buf);
            com = qfits_getcomment_r(line, getcom_buf);

            /* If key or value cannot be found, trigger an error */
            if (key==NULL) {
                qfits_header_destroy(hdr);
                fclose(in);
                return NULL;
            }
            /* Append card to linked-list */
            qfits_header_append(hdr, key, val, com, NULL);
        }
    }
    fclose(in);

    /* The last key should be 'END' */
    if (strlen(key)!=3) {
        qfits_header_destroy(hdr);
        return NULL;
    } 
    if (key[0]!='E' || key[1]!='N' || key[2]!='D') {
        qfits_header_destroy(hdr);
        return NULL;
    }
    
    return hdr;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Read a FITS header from a 'hdr' string
  @param    hdr_str         String containing the hdr file
  @param    nb_char         Number of characters in the string
  @return   Pointer to newly allocated qfits_header or NULL in error case

  This function parses a 'hdr' string, and returns an allocated qfits_header 
  object. 
 */
/*----------------------------------------------------------------------------*/
qfits_header * qfits_header_read_hdr_string(
        const unsigned char *   hdr_str,
        int                     nb_char)
{
	char getval_buf[FITS_LINESZ+1];
	char getkey_buf[FITS_LINESZ+1];
	char getcom_buf[FITS_LINESZ+1];
    qfits_header    *   hdr;
    char                line[81];
    char            *   key,
                    *   val,
                    *   com;
    int                 ind;
    int                 i, j;

    /* Check input */
    if (hdr_str==NULL) return NULL;

    /* Initialise */
    key = val = com = NULL; 

    /* Create the header */
    hdr = qfits_header_new();
    
    /* Go through the file */
    ind = 0;
    while (ind <= nb_char - 80) {
        strncpy(line, (char*)hdr_str + ind, 80);
        line[80] = '\0';
        for (i=0; i<81; i++) {
            if (line[i] == '\n') {
                for (j=i; j<81; j++) line[j] = ' ';
                line[80] = '\0';
                break;
            }
        }
        if (!strcmp(line, "END")) {
            line[3] = ' ';
            line[4] = '\0';
        }
        
        /* Rule out blank lines */
        if (!is_blank_line(line)) {

            /* Get key, value, comment for the current line */
            key = qfits_getkey_r(line, getkey_buf);
            val = qfits_getvalue_r(line, getval_buf);
            com = qfits_getcomment_r(line, getcom_buf);

            /* If key or value cannot be found, trigger an error */
            if (key==NULL) {
                qfits_header_destroy(hdr);
                return NULL;
            }
            /* Append card to linked-list */
            qfits_header_append(hdr, key, val, com, NULL);
        }
        ind += 80;
    }

    /* The last key should be 'END' */
    if (strlen(key)!=3) {
        qfits_header_destroy(hdr);
        return NULL;
    } 
    if (key[0]!='E' || key[1]!='N' || key[2]!='D') {
        qfits_header_destroy(hdr);
        return NULL;
    }
    
    return hdr;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Read an extension header from a FITS file.
  @param    filename    Name of the FITS file to read
  @param    xtnum       Extension number to read, starting from 0.
  @return   Newly allocated qfits_header structure.

  Strictly similar to qfits_header_read() but reads headers from
  extensions instead. If the requested xtension is 0, this function
  returns the main header.

  Returns NULL in case of error.
 */
/*----------------------------------------------------------------------------*/
qfits_header * qfits_header_readext(const char * filename, int xtnum)
{
	char getval_buf[FITS_LINESZ+1];
	char getkey_buf[FITS_LINESZ+1];
	char getcom_buf[FITS_LINESZ+1];
    qfits_header*   hdr;
    int             n_ext;
    char            line[81];
    char        *   where;
    char        *   start;
    char        *   key,
                *   val,
                *   com;
    int             seg_start;
    int             seg_size;
    size_t          size;

    /* Check input */
    if (filename==NULL || xtnum<0) {
		qfits_error("null string or invalid ext num.");
        return NULL;
	}

    /* Check that there are enough extensions */
    if (xtnum>0) {
        n_ext = qfits_query_n_ext(filename);
        if (xtnum>n_ext) {
			qfits_error("invalid ext num: %i > %i.", xtnum, n_ext);
            return NULL;
        }
    }

    /* Get offset to the extension header */
    if (qfits_get_hdrinfo(filename, xtnum, &seg_start, &seg_size)!=0) {
		qfits_error("qfits_get_hdrinfo failed.");
        return NULL;
    }

    /* Memory-map the input file */
    start = qfits_falloc((char *)filename, seg_start, &size);
    if (start==NULL) {
		qfits_error("qfits_falloc failed; maybe you're out of memory (or address space)?");
		return NULL;
	}

    hdr   = qfits_header_new();
    where = start;
    while (1) {
        memcpy(line, where, 80);
        line[80] = '\0';

        /* Rule out blank lines */
        if (!is_blank_line(line)) {

            /* Get key, value, comment for the current line */
            key = qfits_getkey_r(line, getkey_buf);
            val = qfits_getvalue_r(line, getval_buf);
            com = qfits_getcomment_r(line, getcom_buf);

            /* If key or value cannot be found, trigger an error */
            if (key==NULL) {
                qfits_header_destroy(hdr);
                hdr = NULL;
                break;
            }
            /* Append card to linked-list */
            qfits_header_append(hdr, key, val, com, line);
            /* Check for END keyword */
            if (strlen(key)==3)
                if (key[0]=='E' &&
                    key[1]=='N' &&
                    key[2]=='D')
                    break;
        }
        where += 80;
        /* If reaching the end of file, trigger an error */
        if ((int)(where-start)>=(int)(seg_size+80)) {
            qfits_header_destroy(hdr);
            hdr = NULL;
            break;
        }
    }
    qfits_fdealloc(start, seg_start, size);
    return hdr;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Pad an existing file with zeros to a multiple of 2880.
  @param    filename    Name of the file to pad.
  @return   void

  This function simply pads an existing file on disk with enough zeros
  for the file size to reach a multiple of 2880, as required by FITS.
 */
/*----------------------------------------------------------------------------*/
void qfits_zeropad(const char * filename)
{
    struct stat sta;
    int         size;
    int         remaining;
    FILE    *   out;
    char    *   buf;

    if (filename==NULL) return;

    /* Get file size in bytes */
    if (stat(filename, &sta)!=0) {
        return;
    }
    size = (int)sta.st_size;
    /* Compute number of zeros to pad */
    remaining = size % FITS_BLOCK_SIZE;
    if (remaining==0) return;
    remaining = FITS_BLOCK_SIZE - remaining;

    /* Open file, dump zeros, exit */
    if ((out=fopen(filename, "a"))==NULL)
        return;
    buf = qfits_calloc(remaining, sizeof(char));
    fwrite(buf, 1, remaining, out);
    fclose(out);
    qfits_free(buf);
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Identify if a file is a FITS file.
  @param    filename name of the file to check
  @return   int 0, 1, or -1

  Returns 1 if the file name looks like a valid FITS file. Returns
  0 else. If the file does not exist, returns -1.
 */
/*----------------------------------------------------------------------------*/
int qfits_is_fits(const char * filename)
{
    FILE  *   fp;
    char  *   magic;
    int       isfits;

    if (filename==NULL) return -1;
    if ((fp = fopen(filename, "r"))==NULL) {
        qfits_error("cannot open file [%s]: %s", filename, strerror(errno));
        return -1;
    }

    magic = qfits_calloc(FITS_MAGIC_SZ+1, sizeof(char));
    if (fread(magic, 1, FITS_MAGIC_SZ, fp) != FITS_MAGIC_SZ) {
		qfits_error("failed to read file [%s]: %s", filename, strerror(errno));
		return -1;
	}
    fclose(fp);
    magic[FITS_MAGIC_SZ] = '\0';
    if (strstr(magic, FITS_MAGIC)!=NULL)
        isfits = 1;
    else
        isfits = 0;
    qfits_free(magic);
    return isfits;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Retrieve offset to start and size of a header in a FITS file.
  @param    filename    Name of the file to examine
  @param    xtnum       Extension number (0 for main)
  @param    seg_start   Segment start in bytes (output)
  @param    seg_size    Segment size in bytes (output)
  @return   int 0 if Ok, -1 otherwise.

  This function retrieves the two most important informations about
  a header in a FITS file: the offset to its beginning, and the size
  of the header in bytes. Both values are returned in the passed
  pointers to ints. It is Ok to pass NULL for any pointer if you do
  not want to retrieve the associated value.

  You must provide an extension number for the header, 0 meaning the
  main header in the file.
 */
/*----------------------------------------------------------------------------*/
int qfits_get_hdrinfo(
        const char  *   filename,
        int             xtnum,
        int         *   seg_start,
        int         *   seg_size)
{
    if (filename==NULL || xtnum<0 || (seg_start==NULL && seg_size==NULL)) {
        return -1;
    }
    if (seg_start!=NULL) {
        *seg_start = qfits_query(filename, QFITS_QUERY_HDR_START | xtnum);
        if (*seg_start<0)
            return -1;
    }
    if (seg_size!=NULL) {
        *seg_size = qfits_query(filename, QFITS_QUERY_HDR_SIZE | xtnum);
        if (*seg_size<0)
            return -1;
    }
    return 0;
}

int qfits_get_hdrinfo_long(
        const char  *   filename,
        int             xtnum,
        off_t       *   seg_start,
        size_t      *   seg_size) {
    if (filename==NULL || xtnum<0 || (seg_start==NULL && seg_size==NULL)) {
        return -1;
    }
    if (seg_start!=NULL) {
        *seg_start = qfits_query_long(filename, QFITS_QUERY_HDR_START | xtnum);
        if (*seg_start ==QFITS_QUERY_ERROR)
            return -1;
    }
    if (seg_size!=NULL) {
        *seg_size = qfits_query_long(filename, QFITS_QUERY_HDR_SIZE | xtnum);
        if (*seg_size == QFITS_QUERY_ERROR)
            return -1;
    }
    return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Retrieve offset to start and size of a data section in a file.
  @param    filename    Name of the file to examine.
  @param    xtnum       Extension number (0 for main).
  @param    seg_start   Segment start in bytes (output).
  @param    seg_size    Segment size in bytes (output).
  @return   int 0 if Ok, -1 otherwise.

  This function retrieves the two most important informations about
  a data section in a FITS file: the offset to its beginning, and the size
  of the section in bytes. Both values are returned in the passed
  pointers to ints. It is Ok to pass NULL for any pointer if you do
  not want to retrieve the associated value.

  You must provide an extension number for the header, 0 meaning the
  main header in the file.
 */
/*----------------------------------------------------------------------------*/
int qfits_get_datinfo(
        const char  *   filename,
        int             xtnum, 
        int         *   seg_start,
        int         *   seg_size)
{
    if (filename==NULL || xtnum<0 || (seg_start==NULL && seg_size==NULL)) {
        return -1;
    }
    if (seg_start!=NULL) {
        *seg_start = qfits_query(filename, QFITS_QUERY_DAT_START | xtnum);
        if (*seg_start<0)
            return -1;
    }
    if (seg_size!=NULL) {
        *seg_size = qfits_query(filename, QFITS_QUERY_DAT_SIZE | xtnum);
        if (*seg_size<0)
            return -1;
    }
    return 0;  
}

int qfits_get_datinfo_long(
        const char  *   filename,
        int             xtnum, 
        off_t       *   seg_start,
        size_t      *   seg_size)
{
    if (filename==NULL || xtnum<0 || (seg_start==NULL && seg_size==NULL)) {
        return -1;
    }
    if (seg_start!=NULL) {
        *seg_start = qfits_query_long(filename, QFITS_QUERY_DAT_START | xtnum);
        if (*seg_start == QFITS_QUERY_ERROR)
            return -1;
    }
    if (seg_size!=NULL) {
        *seg_size = qfits_query_long(filename, QFITS_QUERY_DAT_SIZE | xtnum);
        if (*seg_size == QFITS_QUERY_ERROR)
            return -1;
    }
    return 0;  
}


/**@}*/

int is_blank_line(const char * s)
{
    int     i;

    for (i=0; i<(int)strlen(s); i++) {
        if (s[i]!=' ') return 0;
    }
    return 1;
}   
