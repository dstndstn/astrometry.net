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
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
    if (hdr_str==NULL) {
        printf("Header string is null; returning null\n");
        return NULL;
    }
    //printf("Parsing header string of length %i\n", nb_char);

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
                printf("Failed to parse line: %s\n", line);
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
        printf("Last key not END\n");
        return NULL;
    } 
    if (key[0]!='E' || key[1]!='N' || key[2]!='D') {
        qfits_header_destroy(hdr);
        printf("Last key not END\n");
        return NULL;
    }
    
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

/**@}*/

int is_blank_line(const char * s)
{
    int     i;

    for (i=0; i<(int)strlen(s); i++) {
        if (s[i]!=' ') return 0;
    }
    return 1;
}   
