/* $Id: replacekey.c,v 1.10 2006/02/17 10:26:58 yjung Exp $
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
 * $Date: 2006/02/17 10:26:58 $
 * $Revision: 1.10 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Include
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "qfits_tools.h" 
#include "qfits_card.h" 

/*-----------------------------------------------------------------------------
                                Define
 -----------------------------------------------------------------------------*/

#define NM_SIZ 512

/*-----------------------------------------------------------------------------
                                Functions prototypes
 -----------------------------------------------------------------------------*/

static char prog_desc[] = "replace keyword in a FITS header" ;
static void usage(char *) ;

/*-----------------------------------------------------------------------------
                                    Main 
 -----------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    char    *       name_in ;
    char    *        card ;
    char    *        place ;
    char    *       key ;
    char    *       key_tmp ;
    char    *       val ;
    char    *       val_tmp ;
    char    *       com ;
    char    *       com_tmp ;
    int             keep_com ;
    int             numeric ;
    char    *       stmp ;
    int             i ;
    
    char            card_tmp[NM_SIZ] ;

    if (argc<2) usage(argv[0]) ;
    
    /* Initialize */
    name_in = NULL ;
    card = NULL ;
    place = NULL ;
    key = NULL ;
    val = NULL ;
    com = NULL ;
    keep_com = 0 ;
    numeric = 0 ;

    /* Command line handling */
    i=1 ;
    while (i<argc) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
        } else if (!strcmp(argv[i], "-p")) {
            if ((i+1)>=argc) {
                fprintf(stderr, "option -p needs an argument\n");
                return -1 ;
            }
            i++ ;
            place = strdup(argv[i]);
        } else if (!strcmp(argv[i], "-k")) {
            if ((i+1)>=argc) {
                fprintf(stderr, "option -k needs an argument\n");
                return -1 ;
            }
            i++ ;
            key = strdup(argv[i]);
        } else if (!strcmp(argv[i], "-v")) {
            if ((i+1)>=argc) {
                fprintf(stderr, "option -v needs an argument\n");
                return -1 ;
            }
            i++ ;
            val = strdup(argv[i]);
        } else if (!strcmp(argv[i], "-c")) {
            if ((i+1)>=argc) {
                fprintf(stderr, "option -c needs an argument\n");
                return -1 ;
            }
            i++ ;
            com = strdup(argv[i]);
        } else if (!strcmp(argv[i], "-C")) {
            keep_com = 1 ;
        } else if (!strcmp(argv[i], "-n")) {
            numeric = 1 ;
        } else {
            break ;
        }
        i++ ;
    }
 
    /* Check options coherence */
    if ((keep_com == 1) && (com != NULL)) {
        fprintf(stderr, "options -c and -C should not be used together\n") ;
        if (place) free(place) ;
        if (key) free(key) ;
        if (val) free(val) ;
        free(com) ;
        return -1 ;
    }
    if (place == NULL) {
        fprintf(stderr, "options -p has to be used\n") ;
        if (key) free(key) ;
        if (val) free(val) ;
        if (com) free(com) ;
        return -1 ;
    }

    /* Get input file name */
    if ((argc-i)<1) {
        fprintf(stderr, "missing input file name\n");
        return -1 ;
    }
    
    /* Loop on the input files */
    while (argc-i >= 1) {
        name_in = strdup(argv[i]) ;

        /* Set keyword to write */
        key_tmp = NULL ;
        if (key==NULL) key_tmp = strdup(place) ;
        else key_tmp = strdup(key) ;
        
        /* Set value to write */
        val_tmp = NULL ;
        if (val==NULL) {
            card = qfits_query_card(name_in, place) ;
            if (card!= NULL) {
                stmp = qfits_getvalue(card) ;
                val_tmp = strdup(stmp) ;
            }
        } else val_tmp = strdup(val) ;
            
        com_tmp = NULL ;
        /* Set comment to write */
        if ((com == NULL) && (keep_com == 1)) {
            if (card == NULL) card = qfits_query_card(name_in, place) ;
            if (card != NULL) {
                stmp = qfits_getcomment(card) ;
                com_tmp = strdup(stmp) ;
            }
        } else if (com != NULL) com_tmp = strdup(com) ;
        if (card != NULL) free(card) ;
        card = NULL ;
        printf("DEBUG: %s\n", key_tmp) ;
        
        /* Create the card */
        qfits_card_build(card_tmp, key_tmp, val_tmp, com_tmp) ;
        if (key_tmp) free(key_tmp) ;
        if (val_tmp) free(val_tmp) ;
        if (com_tmp) free(com_tmp) ;
        card = strdup(card_tmp) ;
         
        /* Display what will be written where */
        printf("File %s\n", name_in) ;
        printf("\tcard  : \n\t\t%s\n", card) ;
        printf("\tplace : \n\t\t%s\n", place) ;

        /* Try to replace the first key */
        if (qfits_replace_card(name_in, place, card) == -1) {
            fprintf(stderr, "cannot replace the key %s\n", place) ;
        }
        free(name_in);
        free(card) ;
        card = NULL ;
        i++ ;
    }
    
    if (val) free(val) ;
    if (com) free(com) ;
    if (key) free(key) ;
    free(place) ;
    /* Free and return */
    return 0 ;
}

static void usage(char * pname)
{
    printf("%s : %s\n", pname, prog_desc) ;
    printf(
    "use : %s [options] <in>\n"
    "options are:\n"
    "\t-p place   gives the keyword to write over (required).\n" 
    "\t-k key     gives the new keyword name (optional).\n" 
    "\t-v val     gives the value to write (optional).\n" 
    "\t-c com     gives the comment to write (optional).\n" 
    "\t-C         flag to keep comment\n"
    "\n", pname) ;
    exit(0) ;
}

