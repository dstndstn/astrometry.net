/* $Id: test_tfits.c,v 1.18 2006/04/27 13:08:43 yjung Exp $
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
 * $Date: 2006/04/27 13:08:43 $
 * $Revision: 1.18 $
 * $Name: qfits-6_2_0 $
 */

/*---------------------------------------------------------------------------
                                   Includes
 ---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "qfits_header.h"
#include "qfits_table.h"
#include "qfits_memory.h"

/*---------------------------------------------------------------------------
                                   Define
 ---------------------------------------------------------------------------*/

#define BIN_TABLE_NAME          "/bintable.tfits"
#define ASCII_TABLE_NAME        "/asciitable.tfits"
#define TMP_TABLE_NAME          "tmp_table.tfits"

/*---------------------------------------------------------------------------
                               Functions prototypes
 ---------------------------------------------------------------------------*/

static int qfits_test_table(char *, int) ;
static void say(char * fmt, ...) ;
static void fail(char * fmt, ...) ;

/*---------------------------------------------------------------------------
                                Functions
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

/* Test a bin table : read it, write it, read it, and compare the 2 reads */
static int qfits_test_table(
        char    *    name,
        int            extnum)
{
    qfits_header    *    qh ;
    qfits_table        *    qt ;
    void            **    one_column_table ;
    void            *    column_after_wr ;
    qfits_table        *    qt_tmp ;
    int                    nb_val ;
    int                    err ;
    double                ddiff ;
    int                    idiff ;

    int                    i, j ;

    /* Initialize */
    err = 0 ;    

    /* Create a default primary header */
    if ((qh = qfits_table_prim_header_default()) == NULL) {
        fail("cannot create a default primary header") ;
        err++ ;
        return err ;
    }
    
    printf("----- %s\n", name) ;
    /* Read the test table */
    if ((qt = qfits_table_open(name, extnum)) == NULL) {
        fail("cannot load infos from the input table") ;
        qfits_header_destroy(qh) ;
        err++ ;
        return err ;
    }
    
    /* For each column */
    for (i=0 ; i<qt->nc ; i++) {
        say("Test the %dth column", i+1) ;
        /* Create the data array */
        one_column_table = qfits_malloc(1*sizeof(void*)) ; 

        /* Load the ith column data */
        if ((one_column_table[0] = qfits_query_column_data(qt, 
                        i, NULL, NULL)) == NULL) {
            qfits_free(one_column_table) ;
            /* The 6th column is empty - normal if not loaded */
            if ((i == 5) && (qt->tab_t == QFITS_BINTABLE)) {
                say("Column %d is empty", i+1) ;
            } else {
                err++ ;
                fail("cannot load the data from the %dth column", i+1) ;
            }
            continue ;
        }
        
        /* Create a tmp table object */
        qt_tmp = qfits_table_new(TMP_TABLE_NAME, qt->tab_t, -1, 1, qt->nr) ;
        qfits_col_fill(qt_tmp->col, (&(qt->col)[i])->atom_nb, 
                (&(qt->col)[i])->atom_dec_nb, (&(qt->col)[i])->atom_size, 
                (&(qt->col)[i])->atom_type, (&(qt->col)[i])->tlabel, 
                (&(qt->col)[i])->tunit, (&(qt->col)[i])->nullval, 
                (&(qt->col)[i])->tdisp, 0, 0.0, 0, 1.0, 0) ;
            
        /* Write a one column table */
        if (qfits_save_table_hdrdump((const void **)one_column_table, qt_tmp, 
                    qh) == -1) {
            fail("cannot save a table to file") ;
            qfits_free(one_column_table) ;
            qfits_table_close(qt_tmp) ;
            err++ ;
            continue ;
        }
        /* Destroy the tmp table object */
        qfits_table_close(qt_tmp) ;
        
        /* Read the one column table */
        if ((qt_tmp = qfits_table_open(TMP_TABLE_NAME, 1)) == NULL) {
            fail("cannot load infos from the generated table"); 
            qfits_free(one_column_table) ;
            err++ ;
            continue ;
        }
    
        /* Load the column data */
        if ((column_after_wr = qfits_query_column_data(qt_tmp, 
                        0, NULL, NULL)) == NULL) {
            fail("cannot load the data from the generated table") ;
            qfits_free(one_column_table) ;
            qfits_table_close(qt_tmp) ;
            err++ ;
            continue ;
        }
        
        /* Destroy the tmp table object */
        remove(qt_tmp->filename) ;    
        qfits_table_close(qt_tmp) ;

        /* Test the diff between column_after_wr and one_column_table[0] */
        switch ((&(qt->col)[i])->atom_type) {
            case TFITS_BIN_TYPE_A :
            case TFITS_BIN_TYPE_L :
            case TFITS_ASCII_TYPE_A :
                /* Set the number of values in the data array */
                nb_val = qt->nr * (&(qt->col)[i])->atom_nb ;
                idiff = strncmp((char*)column_after_wr, 
                        (char*)one_column_table[0], nb_val) ;
                if (idiff == 0) say("Columns are identical...ok") ;
                else { fail("Columns are not the same") ; err++ ;}
                break ;
            case TFITS_BIN_TYPE_B :
            case TFITS_BIN_TYPE_X :
                /* Set the number of values in the data array */
                nb_val = qt->nr * (&(qt->col)[i])->atom_nb ;
                idiff = 0 ;
                for (j=0 ; j<nb_val ; j++) {
                    idiff += ((int)((unsigned char*)column_after_wr)[j] -
                        (int)((unsigned char*)one_column_table[0])[j]) ;
                }
                if (idiff == 0) say("Columns are identical...ok") ;
                else { fail("Columns are not the same") ; err++ ;}
                break ;
            case TFITS_BIN_TYPE_D :
            case TFITS_BIN_TYPE_M :
                /* Set the number of values in the data array */
                nb_val = qt->nr * (&(qt->col)[i])->atom_nb ;
                ddiff = 0.0 ;
                for (j=0 ; j<nb_val ; j++) {
                    ddiff += ((double*)column_after_wr)[j] -
                        ((double*)one_column_table[0])[j] ;
                }
                if (fabs(ddiff) < 1e-6) say("Columns are identical...ok") ;
                else { fail("Columns are not the same") ; err++ ;}
                break ;
            case TFITS_BIN_TYPE_E :
            case TFITS_BIN_TYPE_C :
                /* Set the number of values in the data array */
                nb_val = qt->nr * (&(qt->col)[i])->atom_nb ;
                ddiff = 0.0 ;
                for (j=0 ; j<nb_val ; j++) {
                    ddiff += (double)(((float*)column_after_wr)[j] -
                        ((float*)one_column_table[0])[j]) ;
                }
                if (fabs(ddiff) < 1e-6) say("Columns are identical...ok") ;
                else { fail("Columns are not the same") ; err++ ;}
                break ;
            case TFITS_BIN_TYPE_I :
                /* Set the number of values in the data array */
                nb_val = qt->nr * (&(qt->col)[i])->atom_nb ;
                idiff = 0 ;
                for (j=0 ; j<nb_val ; j++) {
                    idiff += (int)(((short*)column_after_wr)[j] -
                        ((short*)one_column_table[0])[j]) ;
                }
                if (idiff == 0) say("Columns are identical...ok") ;
                else { fail("Columns are not the same") ; err++ ;}
                break ;
            case TFITS_BIN_TYPE_P :
            case TFITS_BIN_TYPE_J :
                /* Set the number of values in the data array */
                nb_val = qt->nr * (&(qt->col)[i])->atom_nb ;
                idiff = 0 ;
                for (j=0 ; j<nb_val ; j++) {
                    idiff += ((int*)column_after_wr)[j] -
                        ((int*)one_column_table[0])[j] ;
                }
                if (idiff == 0) say("Columns are identical...ok") ;
                else { fail("Columns are not the same") ; err++ ;}
                break ;
            case TFITS_ASCII_TYPE_E :
            case TFITS_ASCII_TYPE_F :
                /* Set the number of values in the data array */
                nb_val = qt->nr ;
                ddiff = 0.0 ;
                for (j=0 ; j<nb_val ; j++) {
                    ddiff += (double)(((float*)column_after_wr)[j] -
                        ((float*)one_column_table[0])[j]) ;
                }
                say("Check the cumulated difference : %g", fabs(ddiff)) ; 
                break ;
            case TFITS_ASCII_TYPE_D :
                /* Set the number of values in the data array */
                nb_val = qt->nr ;
                ddiff = 0.0 ;
                for (j=0 ; j<nb_val ; j++) {
                    ddiff += (double)((double*)column_after_wr)[j] -
                        ((double*)one_column_table[0])[j] ;
                }
                say("Check the cumulated difference : %g", fabs(ddiff)) ; 
                break ;
            case TFITS_ASCII_TYPE_I :
                /* Set the number of values in the data array */
                nb_val = qt->nr ;
                idiff = 0 ;
                for (j=0 ; j<nb_val ; j++) {
                    idiff += ((int)((unsigned char*)column_after_wr)[j] -
                        (int)((unsigned char*)one_column_table[0])[j]) ;
                }
                if (idiff == 0) say("Columns are identical...ok") ;
                else { fail("Columns are not the same") ; err++ ;}
                break ;
            default :
                err++ ;
                fail("Column type not recognized") ;
                break ;
        }
    
        /* Free */
        qfits_free(one_column_table[0]) ;
        qfits_free(one_column_table) ;
        qfits_free(column_after_wr) ;
    }

    /* Delete the table */
    qfits_table_close(qt) ;

    /* Delete the fits header */
    qfits_header_destroy(qh) ;

    return err ;
}

/*---------------------------------------------------------------------------
                                Main
 ---------------------------------------------------------------------------*/

int main(int argc, char * argv[])
{
    int         err ;
    char        filename[512] ;
    char    *   srcdir ;

    /* Initialize */
    err=0 ;

    /* Get the src directory */
    srcdir = strdup(getenv("srcdir")) ;
    
    /* 
     *    Test on BINARY tables
     */
    say("Test the BINARY table") ;
    strcat(filename, srcdir) ;
    strcat(filename, BIN_TABLE_NAME) ;
    err += qfits_test_table(filename, 1) ;    
    
    /* 
     *    Test on ASCII tables
     */
    say("Test the ASCII table") ;
    filename[0] = (char)0 ;
    strcat(filename, srcdir) ;
    strcat(filename, ASCII_TABLE_NAME) ;
    err += qfits_test_table(filename, 1) ;    
    
    free(srcdir) ;
    fprintf(stderr, "total error(s): %d\n", err);
    return err ;
}



