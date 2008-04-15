/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: dtfits.c,v 1.19 2006/02/17 13:41:49 yjung Exp $
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
 * $Date: 2006/02/17 13:41:49 $
 * $Revision: 1.19 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>

#include "qfits_table.h"
#include "qfits_tools.h"

/*-----------------------------------------------------------------------------
                                   Define
 -----------------------------------------------------------------------------*/

#define ELEMENT_MAX_DISP_SIZE   50
#define DISP_SIZE_INT           5
#define DISP_SIZE_DOUBLE        8
#define DISP_SIZE_FLOAT         7
#define DISP_SIZE_CHAR          1      

/*-----------------------------------------------------------------------------
                            Function prototypes
 -----------------------------------------------------------------------------*/

static int dump_extension_bin(qfits_table *, FILE *, void **, char, int,
        int) ;
static int dump_extension_ascii(qfits_table *, FILE *, void **, char,
        int, int) ;
static int dump_extension(qfits_table *, FILE *, char, int, int) ;
static void qfits_dump(char *, char *, int, char, int);
static void usage(char * pname) ;
static char prog_desc[] = "FITS table dump" ;

/*-----------------------------------------------------------------------------
                                Main 
 -----------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    char    name_i[512] ;
    char    name_o[512] ;
    int        i ;
    int        data_only ;
    char    separator ;
    int     ext ;

    /* Initialize */
    data_only = 0 ;
    separator = '|' ;
    ext = -1 ;
    
    i=1 ;

    if (argc<2) usage(argv[0]);

    while (i<argc) {
        if (!strcmp(argv[i], "--help") ||
            !strcmp(argv[i], "-h")) {
            usage(argv[0]);
        } else if (!strcmp(argv[i], "-d")) {
            data_only=1 ;
        } else if (!strcmp(argv[i], "-s")) {
            if ((i+1)>=argc) {
                fprintf(stderr, "option -s needs an argument\n");
                return -1 ;
            }
            i++ ;
            separator = argv[i][0] ;


        } else if (!strcmp(argv[i], "-x")) {
            if ((i+1)>=argc) {
                fprintf(stderr, "option -x needs an argument\n");
                return -1 ;
            }
            i++ ;
            ext = atoi(argv[i]) ;
        } else {
            break ;
        }
        i++ ;
    }
    if ((argc-i)<1) {
        fprintf(stderr, "missing input file name\n");
        return -1 ;
    }
    strcpy(name_i, argv[i]);
    i++ ;
    if ((argc-i)<1) name_o[0] = 0 ;
    else strcpy(name_o, argv[i]);

    qfits_dump(name_i, name_o, data_only, separator, ext);
    return 0 ;
}

static void usage(char * pname)
{
    printf("%s -- %s\n", pname, prog_desc);
    printf(
    "use : %s [options] <FITS table> [out]\n"
    "options are:\n"
    "\t-d          to dump data only (no headers)\n"
    "\t-s <char>   to output data with separator <char>\n"
    "\n",
    pname);
    exit(0) ;
}

static void qfits_dump(
        char    *   name_i,
        char    *   name_o,
        int         data_only,
        char        separator,
        int         ext)
{
    qfits_table        *    tdesc ;
    FILE            *    out ;
    int                    xtnum_start, xtnum_stop ;
    int                    n_ext ;
    int                 i ;
    
    /* Set where to send the output */
    if (name_o[0]==(char)0) {
        out = stdout ;
    } else {
        if ((out = fopen(name_o, "w"))==NULL) {
            fprintf(stderr, "cannot create output file [%s]\n", name_o);
            return ;
        }
    }
    
    if (!data_only) {
        fprintf(out, "#\n");
        fprintf(out, "# file           %s\n", name_i);
    }

    /* Query number of extensions in the file */
    n_ext = qfits_query_n_ext(name_i);
    if (!data_only) {
        fprintf(out, "# extensions     %d\n", n_ext);
    }
    /* If no extension, bail out */
    if (n_ext<1) {
        if (out!=stdout) fclose(out) ;
        return ;
    }

    /* 1 extension required or all */
    if (ext < 1) {
        xtnum_start = 1 ;
        xtnum_stop = n_ext ;
    } else if (ext > n_ext) {
        fprintf(out, "# requested extension does not exist %d\n", ext) ;
        if (out!=stdout) fclose(out) ;
        return ;
    } else {
        xtnum_start = xtnum_stop = ext ;
    }

    /* Loop over all extensions */
    for (i=xtnum_start ; i<=xtnum_stop ; i++) {
        if (!data_only) {
            fprintf(out, "# --------------------------------------------\n");
            fprintf(out, "# XTENSION       %d\n", i);
        }
        if ((tdesc = qfits_table_open(name_i, i)) == NULL) {
            printf("cannot open table [%s]:[%d]\n", name_i, i);
            if (out!=stdout) fclose(out);
            return ;
        }
        dump_extension(tdesc, out, separator, data_only, 1) ;
        qfits_table_close(tdesc);
    }
    fclose(out) ;
    return ;
}
    
static int dump_extension(
        qfits_table        *   tdesc,
        FILE            *    out,
        char                separator,
        int                    data_only,
        int                    use_zero_scale)
{
    void            **      cols ;
    int                     i ;
    
    if (!data_only) {
        fprintf(out, "# Number of columns %d\n", tdesc->nc);
        fprintf(out, "#\n");
    }

    /* First read the columns in memory */
    cols = malloc(tdesc->nc * sizeof(void*)) ;
    for (i=0 ; i<tdesc->nc ; i++) {
        cols[i] = qfits_query_column_data(tdesc, i, NULL, NULL) ;
        if (cols[i] == NULL) {
            fprintf(out, "# Cannot load column nb %d\n", i+1) ;
        }
    }

    switch (tdesc->tab_t) {
        case QFITS_BINTABLE:
            dump_extension_bin(tdesc, out, cols, separator, data_only, 
                    use_zero_scale) ;
            break ;
        case QFITS_ASCIITABLE:
            dump_extension_ascii(tdesc, out, cols, separator, data_only, 
                    use_zero_scale) ;
            break ;
        default:
            fprintf(out, "Table type not recognized") ;
            break ;
    }
    
    for (i=0 ; i<tdesc->nc ; i++) if (cols[i]) free(cols[i]) ;
    free(cols) ;
    return 0 ;
}

static int dump_extension_bin(
        qfits_table        *   tdesc,
        FILE            *    out,
        void            **  cols,
        char                separator,
        int                 data_only,
        int                    use_zero_scale)
{
    int             *   col_sizes ;
    qfits_col       *   col ;
    char            *   ccol ;
    unsigned char   *   ucol ;
    double          *   dcol ;
    float           *   fcol ;
    short           *   scol ;
    int             *   icol ;
    int                 size ;
    int                 field_size ;
    char            *   str ;
    char                ctmp[512];
    int                 i, j, k ;

    /* GET THE FIELDS SIZES */
    col_sizes = calloc(tdesc->nc, sizeof(int)) ;
    for (i=0 ; i<tdesc->nc ; i++) {
		size = 0;
        col = tdesc->col + i ;
        col_sizes[i] = (int)strlen(col->tlabel) ;
        switch(col->atom_type) {
            case TFITS_BIN_TYPE_A:
                size = col->atom_size * col->atom_nb ; 
                break ;
            case TFITS_BIN_TYPE_B:
                size = col->atom_nb * (DISP_SIZE_INT + 2) ;
                break ;
            case TFITS_BIN_TYPE_D:
            case TFITS_BIN_TYPE_M:
                size = col->atom_nb * (DISP_SIZE_DOUBLE + 2) ;
                break ;
            case TFITS_BIN_TYPE_E:
            case TFITS_BIN_TYPE_C:
                size = col->atom_nb * (DISP_SIZE_FLOAT + 2) ;
                break ;
            case TFITS_BIN_TYPE_I:
                size = col->atom_nb * (DISP_SIZE_INT + 2) ;
                break ;
            case TFITS_BIN_TYPE_J:
                size = col->atom_nb * (DISP_SIZE_INT + 2) ;
                break ;
            case TFITS_BIN_TYPE_L:
                size = col->atom_nb * (DISP_SIZE_CHAR + 2) ;
                break ;
            case TFITS_BIN_TYPE_X:
                size = col->atom_nb * (DISP_SIZE_INT + 2) ;
                break ;
            case TFITS_BIN_TYPE_P:
                size = col->atom_nb * (DISP_SIZE_INT + 2) ;
                break ;
            default:
                fprintf(out, "Type not recognized") ;
                break ;
        }
        if (size > col_sizes[i]) col_sizes[i] = size ;
    }

    /* Print out the column names */
    if (!data_only) {
        for (i=0 ; i<tdesc->nc ; i++) {
            col = tdesc->col + i ;
            fprintf(out, "%*s", col_sizes[i], col->tlabel);
            if (i!=(tdesc->nc-1)) printf("%c", separator);
        }
        fprintf(out, "\n");
    }
    
    /* Get the string to write according to the type */
    for (j=0 ; j<tdesc->nr ; j++) {
        for (i=0 ; i<tdesc->nc ; i++) {
            if (cols[i] == NULL) continue ;
            col = tdesc->col + i ;
            field_size = col->atom_nb * ELEMENT_MAX_DISP_SIZE ;
            str = malloc(field_size * sizeof(char)) ;
            str[0] = (char)0 ;
            switch(col->atom_type) {
                case TFITS_BIN_TYPE_A:
                    ccol = (char*)(cols[i]) ;
                    ccol += col->atom_size * col->atom_nb * j ;
                    strncpy(ctmp, ccol, col->atom_size * col->atom_nb) ;
                    ctmp[col->atom_size*col->atom_nb] = (char)0 ;
                    strcpy(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_B:
                    ucol = (unsigned char*)(cols[i]) ;
                    ucol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%d, ", (int)ucol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp,"%d",(int)ucol[col->atom_nb-1]);
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_D:
                case TFITS_BIN_TYPE_M:
                    dcol = (double*)(cols[i]) ;
                    dcol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%g, ", dcol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp, "%g", dcol[col->atom_nb-1]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_E:
                case TFITS_BIN_TYPE_C:
                    fcol = (float*)(cols[i]) ;
                    fcol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%f, ", fcol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp, "%f", fcol[col->atom_nb-1]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_I:
                    scol = (short*)(cols[i]) ;
                    scol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%d, ", scol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp, "%d", scol[col->atom_nb-1]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_J:
                    icol = (int*)(cols[i]) ;
                    icol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%d, ", icol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp, "%d", icol[col->atom_nb-1]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_L:
                    ccol = (char*)(cols[i]) ;
                    ccol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%c, ", ccol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp, "%c", ccol[col->atom_nb-1]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_X:
                    ucol = (unsigned char*)(cols[i]) ;
                    ucol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%d, ", ucol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp, "%d", ucol[col->atom_nb-1]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_BIN_TYPE_P:
                    icol = (int*)(cols[i]) ;
                    icol += col->atom_nb * j ;
                    /* For each atom of the column */
                    for (k=0 ; k<col->atom_nb-1 ; k++) {
                        sprintf(ctmp, "%d, ", icol[k]) ;
                        strcat(str, ctmp) ;
                    }
                    /* Handle the last atom differently: no ',' */
                    sprintf(ctmp, "%d", icol[col->atom_nb-1]) ;
                    strcat(str, ctmp) ;
                    break ;
                default:
                    fprintf(out, "Type not recognized") ;
                    break ;
            }
            fprintf(out, "%*s", col_sizes[i], str);
            if (i!=(tdesc->nc-1)) printf("%c", separator);
            free(str) ;
        }
        fprintf(out, "\n");
    }
    return 0 ;
}

static int dump_extension_ascii(
        qfits_table        *   tdesc,
        FILE            *    out,
        void            **  cols,
        char                separator,
        int                 data_only,
        int                    use_zero_scale)
{
    int             *   col_sizes ;
    qfits_col       *   col ;
    char            *   ccol ;
    double          *   dcol ;
    float           *   fcol ;
    int             *   icol ;
    int                 size ;
    int                 field_size ;
    char            *   str ;
    char                ctmp[512];
    int                 i, j ;

    /* GET THE FIELDS SIZES */
    col_sizes = calloc(tdesc->nc, sizeof(int)) ;
    for (i=0 ; i<tdesc->nc ; i++) {
		size = 0;
        col = tdesc->col + i ;
        col_sizes[i] = (int)strlen(col->tlabel) ;
        switch(col->atom_type) {
            case TFITS_ASCII_TYPE_A:
                size = col->atom_nb ; 
                break ;
            case TFITS_ASCII_TYPE_I:
                size = DISP_SIZE_INT ;
                break ;
            case TFITS_ASCII_TYPE_E:
            case TFITS_ASCII_TYPE_F:
                size = DISP_SIZE_FLOAT ;
                break ;
            case TFITS_ASCII_TYPE_D:
                size = DISP_SIZE_DOUBLE ;
                break ;
            default:
                fprintf(out, "Type not recognized") ;
                break ;
        }
        if (size > col_sizes[i]) col_sizes[i] = size ;
    }

    /* Print out the column names */
    if (!data_only) {
        for (i=0 ; i<tdesc->nc ; i++) {
            col = tdesc->col + i ;
            fprintf(out, "%*s", col_sizes[i], col->tlabel);
            if (i!=(tdesc->nc-1)) printf("%c", separator);
        }
        fprintf(out, "\n");
    }
    
    /* Get the string to write according to the type */
    for (j=0 ; j<tdesc->nr ; j++) {
        for (i=0 ; i<tdesc->nc ; i++) {
            if (cols[i] == NULL) continue ;
            col = tdesc->col + i ;
            field_size = col->atom_nb * ELEMENT_MAX_DISP_SIZE ;
            str = malloc(field_size * sizeof(char)) ;
            str[0] = (char)0 ;
            switch(col->atom_type) {
                case TFITS_ASCII_TYPE_A:
                    ccol = (char*)(cols[i]) ;
                    ccol += col->atom_nb * j ;
                    strncpy(ctmp, ccol, col->atom_nb) ;
                    ctmp[col->atom_nb] = (char)0 ;
                    strcpy(str, ctmp) ;
                    break ;
                case TFITS_ASCII_TYPE_I:
                    icol = (int*)(cols[i]) ;
                    icol += j ;
                    sprintf(ctmp, "%d", icol[0]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_ASCII_TYPE_E:
                case TFITS_ASCII_TYPE_F:
                    fcol = (float*)(cols[i]) ;
                    fcol += j ;
                    sprintf(ctmp, "%f", fcol[0]) ;
                    strcat(str, ctmp) ;
                    break ;
                case TFITS_ASCII_TYPE_D:
                    dcol = (double*)(cols[i]) ;
                    dcol += j ;
                    sprintf(ctmp, "%g", dcol[0]) ;
                    strcat(str, ctmp) ;
                    break ;
                default:
                    fprintf(out, "Type not recognized") ;
                    break ;
            }
            fprintf(out, "%*s", col_sizes[i], str);
            if (i!=(tdesc->nc-1)) printf("%c", separator);
            free(str) ;
        }
        fprintf(out, "\n");
    }
 
    return 0 ;

}


