/* $Id: stripfits.c,v 1.4 2006/02/17 10:26:07 yjung Exp $
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
 * $Date: 2006/02/17 10:26:07 $
 * $Revision: 1.4 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Include
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>

/*-----------------------------------------------------------------------------
                                Define
 -----------------------------------------------------------------------------*/

#define NM_SIZ          512
#define FITS_BLSZ       2880
#define ONE_MEGABYTE    (1024 * 1024)

#ifndef SEEK_SET
#define SEEK_SET    0
#endif

/*-----------------------------------------------------------------------------
                            Function prototypes
 -----------------------------------------------------------------------------*/

static int dump_pix(char*, char*);
static int get_FITS_header_size(char *) ;
static int filesize(char*) ;
static int get_bitpix(char*) ;
static int get_npix(char*) ;

/*-----------------------------------------------------------------------------
                                Main
 -----------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    if (argc<3) {
        printf("\n\n") ;
        printf("use: %s <in> <out>\n", argv[0]) ;
        printf("\n") ;
        printf("\t<in> is a valid FITS file in the current directory\n") ;
        printf("\t<out> is the name of the output file\n") ;
        printf("\n\n") ;
        return 0 ;
    }
    if (!strcmp(argv[1], argv[2])) {
        fprintf(stderr, "cannot convert a file to itself\n") ;
        fprintf(stderr, "specify another name for the output\n") ;
        return -1 ;
    }
    return dump_pix(argv[1], argv[2]);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    dump pixels from a FITS file to a binary file 
  @param    name_in     Input file name
  @param    name_out    Output file name
  @return   int 0 if Ok, anything else otherwise
  Heavy use of mmap() to speed up the process
 */
/*----------------------------------------------------------------------------*/
static int dump_pix(
        char    *   name_in,
        char    *   name_out)
{
    int         fd_in,
                fd_out ;
    char    *   buf_in ;
    char    *   buf_out ;
    int         fsize_in,
                fsize_out,
                header_size ;
    int         npix ;
    
    /*
     * Open input file and get information we need:
     * - pixel depth
     * - number of pixels
     */

    fsize_in = filesize(name_in) ;
    header_size = get_FITS_header_size(name_in) ;
    if ((fd_in = open(name_in, O_RDONLY)) == -1) {
        fprintf(stderr, "cannot open file %s: aborting\n", name_in) ;
        return -1 ;
    }
    buf_in = (char*)mmap(0, fsize_in, PROT_READ, MAP_SHARED, fd_in, 0) ;
    if (buf_in == (char*)-1) {
        perror("mmap") ;
        fprintf(stderr, "cannot mmap file: %s\n", name_in) ;
        close(fd_in) ;
        return -1 ;
    }
    if (get_bitpix(buf_in) != -32) {
        fprintf(stderr, "only 32-bit IEEE floating point format supported\n");
        close(fd_in) ; munmap(buf_in, fsize_in) ;
        return -1 ;
    }

    /*
     * Compute the size of the output file
     * same header size, + pixel area + blank padding
     */
    npix = get_npix(buf_in) ;
    if (npix < 1) {
        fprintf(stderr, "cannot compute number of pixels\n");
        close(fd_in) ;
        munmap(buf_in, fsize_in) ;
        return -1 ;
    }
    fsize_out = npix * 4 ;
    /*
     * Now create the output file and fill it with zeros, then mmap it.
     * The permissions are rw-rw-r-- by default.
     */
    if ((fd_out=creat(name_out,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH))==-1){
        perror("creat") ;
        fprintf(stderr, "cannot create file %s: aborting\n", name_out) ;
        close(fd_in) ; munmap(buf_in, fsize_in) ;
        return -1 ;
    }

    buf_out = malloc(fsize_out) ;
    if (buf_out == NULL) {
        fprintf(stderr, "not enough memory\n");
        fprintf(stderr, "failed to allocate %d bytes\n", fsize_out);
        close(fd_in) ; munmap(buf_in, fsize_in) ;
        return -1;
    }
    write(fd_out, buf_out, fsize_out);
    close(fd_out);
    free(buf_out);

    fd_out = open(name_out, O_RDWR);
    if (fd_out==-1) {
        fprintf(stderr, "error opening file %s\n", name_out);
        close(fd_in) ; munmap(buf_in, fsize_in) ;
        return -1;
    }
    buf_out = (char*)mmap(0,
                         fsize_out,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED,
                         fd_out,
                         0) ;
    if (buf_out == (char*)-1) {
        perror("mmap") ;
        fprintf(stderr, "cannot mmap file: %s\n", name_out) ;
        munmap(buf_in, fsize_in) ; close(fd_in) ; close(fd_out) ;
        return -1 ;
    }
    /*
     * Copy FITS header from input to output, modify BITPIX
     */
    memcpy(buf_out, buf_in+header_size, fsize_out) ;
    /*
     * Close, unmap, goodbye
     */
    close(fd_in) ;
    close(fd_out) ;
    munmap(buf_in, fsize_in) ;
    munmap(buf_out, fsize_out) ;
    return 0 ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    compute the size (in bytes) of a FITS header
  @param    name    FITS file name
  @return   unsigned long
  Should always be a multiple of 2880. This implementation assumes only that
  80 characters are found per line.
 */
/*----------------------------------------------------------------------------*/
static int get_FITS_header_size(char * name)
{
    FILE    *   in ;
    char        line[81] ;
    int         found = 0 ;
    int         count ;
    int         hs ;

    if ((in = fopen(name, "r")) == NULL) {
        fprintf(stderr, "cannot open %s: aborting\n", name) ;
        return 0 ;
    }
    count = 0 ;
    while (!found) {
        if (fread(line, 1, 80, in)!=80) {
            break ;
        }
        count ++ ;
        if (!strncmp(line, "END ", 4)) {
            found = 1 ;
        }
    }
    fclose(in);

    if (!found) return 0 ;
    /*
     * The size is the number of found cards times 80,
     * rounded to the closest higher multiple of 2880.
     */
    hs = count * 80 ;
    if ((hs % FITS_BLSZ) != 0) {
        hs = (1+(hs/FITS_BLSZ)) * FITS_BLSZ ;
    }
    return hs ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    how many bytes can I read from this file
  @param    filename    File name   
  @return   size of the file in bytes
  Strongly non portable. Only on Unix systems!
 */
/*----------------------------------------------------------------------------*/
static int filesize(char *filename)
{
    int          size ;
    struct stat fileinfo ;

    /* POSIX compliant  */
    if (stat(filename, &fileinfo) != 0) {
        size = (int)0 ;
    } else {
        size = (int)fileinfo.st_size ;
    }
    return size ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    gets the value of BITPIX from a FITS header
  @param    buf     allocated char buffer containing the whole FITS header
  @return   int 8 16 32 -32 or -64 or 0 if cannot find it
 */
/*----------------------------------------------------------------------------*/
static int get_bitpix(char * buf)
{
    int     bitpix ;
    char *  where ;

    where = strstr(buf, "BITPIX") ;
    if (where == NULL) {
        fprintf(stderr, "cannot find BITPIX in header: aborting\n") ;
        return 0 ;
    }
    sscanf(where, "%*[^=] = %d", &bitpix) ;
    /*
     * Check the returned value makes sense
     */
    if ((bitpix != 8) &&
        (bitpix != 16) &&
        (bitpix != 32) &&
        (bitpix != -32) &&
        (bitpix != -64)) {
        bitpix = 0 ;
    }
    return bitpix ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    retrieves how many pixels in a FITS file from the header
  @param    buf     allocated char buffer containing the whole FITS header
  @return   unsigned long: # of pixels in the file
  Does not support extensions!
 */
/*----------------------------------------------------------------------------*/
static int get_npix(char * buf)
{
    int     naxes ;
    int     npix ;
    int         naxis ;
    char      * where ;
    char        lookfor[80] ;
    int         i ;

    where = strstr(buf, "NAXIS") ;
    if (where == NULL) {
        fprintf(stderr, "cannot find NAXIS in header: aborting\n") ;
        return 0 ;
    }
    sscanf(where, "%*[^=] = %d", &naxes) ;
    if ((naxes<1) || (naxes>999)) {
        fprintf(stderr, "illegal value for %s: %d\n", lookfor, naxes) ;
        return 0 ;
    }
    npix = 1 ;
    for (i=1 ; i<=naxes ; i++) {
        sprintf(lookfor, "NAXIS%d", i) ;
        where = strstr(buf, lookfor) ;
        if (where == NULL) {
            fprintf(stderr, "cannot find %s in header: aborting\n",
                    lookfor) ;
            return 0 ;
        }
        sscanf(where, "%*[^=] = %d", &naxis) ;
        if (naxis<1) {
            fprintf(stderr, "error: found %s=%d\n", lookfor, naxis);
            return 0 ;
        }
        npix *= (int)naxis ;
    }
    return npix ;
}

