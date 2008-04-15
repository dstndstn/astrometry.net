/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qextract.c,v 1.9 2006/02/20 09:45:24 yjung Exp $
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
 * $Date: 2006/02/20 09:45:24 $
 * $Revision: 1.9 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Includes
 -----------------------------------------------------------------------------*/

#include "qfits_table.h" 
#include "qfits_image.h" 
#include "qfits_tools.h"
#include "qfits_rw.h" 

/*-----------------------------------------------------------------------------
                               Function prototypes
 -----------------------------------------------------------------------------*/

static int textract_write_ext(char *, int) ;
static int iextract_write_ext(char *, int) ;
static char prog_desc[] = "Extract and write FITS extensions" ;
static void usage(char *) ;

/*-----------------------------------------------------------------------------
                                    Main 
 -----------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    char    *       name_in ;
    int             ext ;
    int             nb_ext ;
    int             ext_type ;
    
    if (argc<3) usage(argv[0]) ;
    
    /* Get input file name  and extension number */
    name_in = strdup(argv[1]) ;
    ext = (int)atoi(argv[2]) ;
   
    /* Check if the file is a FITS file */
    if (!qfits_is_fits(name_in)) {
        printf("[%s] is not a FITS file\n", name_in) ;
        free(name_in) ;
        return -1 ;
    }
    
    /* Check if the extension is valid */
    nb_ext = qfits_query_n_ext(name_in) ;
    if (nb_ext < ext) {
        printf("Only %d extensions in this file\n", nb_ext) ;
        free(name_in) ;
        return -1 ;
    }

    /* Check if it's a table or an image */
    ext_type = qfits_is_table(name_in, ext) ;
   
    switch (ext_type) {
        case QFITS_BINTABLE:
        case QFITS_ASCIITABLE:
            if (textract_write_ext(name_in, ext) == -1) {    
                printf("cannot read-write extension no %d\n", ext) ;
                free(name_in) ;
                return -1 ;
            }
            break ;
        case QFITS_INVALIDTABLE:
            if (iextract_write_ext(name_in, ext) == -1) {    
                printf("cannot read-write extension no %d\n", ext) ;
                free(name_in) ;
                return -1 ;
            }
            break ;
        default:
            printf("Unrecognized FITS type\n") ;
            free(name_in) ;
            return -1 ;
    }
    free(name_in) ;
    
    return 0 ;
}

/*-----------------------------------------------------------------------------
                                  Functions code
 -----------------------------------------------------------------------------*/

static void usage(char * pname)
{
    printf("%s : %s\n", pname, prog_desc) ;
    printf(
    "use : %s <in> <extension>\n"
    "\n", pname) ;
    exit(0) ;
}

static int textract_write_ext(
        char    *   in,
        int         ext)
{
    qfits_table     *   th ;
    void            **  array ;
    qfits_header    *   fh ;
    int                 array_size ;
    int                 i ;

    /* Get the table infos */
    if ((th = qfits_table_open(in, ext)) == NULL) {
        printf("cannot read extension: %d\n", ext) ;
        return -1 ;
    }
    
    /* Compute array_size */
    array_size = 0 ;
    for (i=0 ; i<th->nc ; i++) {
        switch (th->col[i].atom_type) {
            case TFITS_ASCII_TYPE_A:
            case TFITS_ASCII_TYPE_I:
            case TFITS_ASCII_TYPE_E:
            case TFITS_ASCII_TYPE_F:
            case TFITS_ASCII_TYPE_D:
            case TFITS_BIN_TYPE_A:
            case TFITS_BIN_TYPE_L:
            case TFITS_BIN_TYPE_B:
            case TFITS_BIN_TYPE_X:
                array_size += sizeof(char*) ;
                break ;
                
            case TFITS_BIN_TYPE_I:
                array_size += sizeof(short*) ;
                break ;

            case TFITS_BIN_TYPE_J:
            case TFITS_BIN_TYPE_E:
                array_size += sizeof(int*) ;
                break ;
            
            case TFITS_BIN_TYPE_C:
            case TFITS_BIN_TYPE_P:
                array_size += sizeof(float*) ;
                break ;

            case TFITS_BIN_TYPE_D:
            case TFITS_BIN_TYPE_M:
                array_size += sizeof(double*) ;
                break ;
            default:
                return -1 ;
        }
    }
    
    /* Allocate memory for array */
    array = malloc(array_size) ;

    /* Load columns in array */
    for (i=0 ; i<th->nc ; i++) {
        array[i] = qfits_query_column_data(th, i, NULL, NULL) ;
        if (array[i] == NULL) {
            printf("cannot read column %d\n", i+1) ;
        }
    }

    /* Update th : filename */
    sprintf(th->filename, "ext%d.tfits", ext) ; 

    /* Get fits primary header */
    if ((fh = qfits_header_read(in)) == NULL) {
        for (i=0 ; i<th->nc ; i++) if (array[i] != NULL) free(array[i]) ;
        qfits_table_close(th) ;
        free(array) ;
        printf("cannot read fits header\n") ;
        return -1 ; 
    }
 
    if (ext != 0) {
        /* No data in primary HDU */
        qfits_header_mod(fh, "NAXIS", "0", NULL) ;
        qfits_header_del(fh, "NAXIS1") ;
        qfits_header_del(fh, "NAXIS2") ;
    }   

    /* Write the tfits file */
    if (qfits_save_table_hdrdump((const void **)array, th, fh) == -1) {
        qfits_header_destroy(fh) ;
        for (i=0 ; i<th->nc ; i++) if (array[i] != NULL) free(array[i]) ;
        qfits_table_close(th) ;
        free(array) ;
        printf("cannot write fits table\n") ;
        return -1 ;
    }

    /* Free and return */
    qfits_header_destroy(fh) ;
    for (i=0 ; i<th->nc ; i++) if (array[i] != NULL) free(array[i]) ;
    qfits_table_close(th) ;
    free(array) ;
    return 0 ;
}

static int iextract_write_ext(
        char    *   in,
        int         ext)
{
    qfitsloader         ql ;
    qfitsdumper         qd ;
    qfits_header    *   fh ;
    char                outname[1024] ;
    FILE            *   out ;
    
    sprintf(outname, "ext%d.fits", ext) ;
    
    /* Initialize a FITS loader */
    ql.filename = in ;
    ql.xtnum    = ext ;
    ql.pnum     = 0 ;
    ql.map      = 1 ;
    ql.ptype    = PTYPE_DOUBLE ;
    if (qfitsloader_init(&ql)!=0)   return -1 ;
   
    /* Load the primary header */
    if ((fh = qfits_header_read(ql.filename)) == NULL) return -1 ;

    if (ext != 0) {
        /* No data in primary HDU */
        qfits_header_mod(fh, "NAXIS", "0", NULL) ;
        qfits_header_del(fh, "NAXIS1") ;
        qfits_header_del(fh, "NAXIS2") ;
    }
    
    /* Dump the primary header */
    if ((out = fopen(outname, "w")) == NULL) return -1 ;
    qfits_header_dump(fh, out);
    qfits_header_destroy(fh) ;

    if (ext != 0) {
        /* Load the extension header */
        if ((fh = qfits_header_readext(ql.filename, ext)) == NULL) return -1 ;
        /* Dump the extension header */
        qfits_header_dump(fh, out);
        qfits_header_destroy(fh) ;
    }
    fclose(out) ;
    
    /* Load the FITS image */
    if (qfits_loadpix(&ql)!=0) return -1 ;
    
    /* Write the FITS image */
    qd.filename  = outname ;
    qd.npix      = ql.lx * ql.ly ;
    qd.ptype     = PTYPE_DOUBLE ;
    qd.dbuf      = ql.dbuf ;
    qd.out_ptype = ql.bitpix ;
    if (qfits_pixdump(&qd)!=0) return -1 ;
    /* qfits_zeropad(outname) ; */
    free(ql.dbuf) ;
    
    return 0 ;
}
