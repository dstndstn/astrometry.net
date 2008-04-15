/* $Id: fitsort.c,v 1.6 2006/02/17 10:26:07 yjung Exp $
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
 * $Revision: 1.6 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Include
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/*-----------------------------------------------------------------------------
                                Define
 -----------------------------------------------------------------------------*/

#define MAX_STRING  128        
#define MAX_KEY     512       
#define FMT_STRING  "%%-%ds\t"

/*-----------------------------------------------------------------------------
                                New types   
 -----------------------------------------------------------------------------*/

/* This holds a keyword value and a flag to indicate its presence   */
typedef struct _KEYWORD_ {
    char    value[MAX_STRING] ;
    int     present ;
} keyword ;

/* Each detected file in input has such an associated structure */
typedef struct _RECORD_ {
    char            filename[MAX_STRING] ;
    keyword         listkw[MAX_KEY] ;
} record ;

/*-----------------------------------------------------------------------------
                            Function prototypes
 -----------------------------------------------------------------------------*/

static int isfilename(char *string) ;
static void getfilename(char *line, char *word) ;
static char * expand_hierarch_keyword(char *, char *) ;
static int isdetectedkeyword(char *line, char *keywords[], int nkeys) ;
static void getkeywordvalue(char *line, char *word) ;

/*-----------------------------------------------------------------------------
                                Main
 -----------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    char    curline[MAX_STRING] ;
    char    word[MAX_STRING] ;
    int     i, j ;
    int     nfiles ;
    record  *allrecords ;
    int     kwnum ;
    int        len ;
    int        max_width[MAX_KEY] ;
    int        max_filnam ;
    char    fmt[8] ;
    int        flag ;
    int        printnames ;
    int        print_hdr ;

    if (argc<2) {
        printf("\n\nuse : %s [-d] KEY1 KEY2 ... KEYn\n", argv[0]) ;
        printf("Input data is received from stdin\n") ;
        printf("See man page for more details and examples\n\n") ;
        return 0 ;
    }

    /* Initialize */
    printnames = 0 ;
    print_hdr  = 1 ;
    nfiles = 0 ;
    allrecords = (record*)calloc(1, sizeof(record));
    if (!strcmp(argv[1], "-d")) {
        print_hdr = 0;
        argv++ ;
        argc-- ;
    }
    argv++ ;

    /* Uppercase all inputs */
    for (i=0 ; i<(argc-1) ; i++) {
        j=0 ;
        while (argv[i][j]!=0) {
            argv[i][j] = toupper(argv[i][j]);
            j++ ;
        }
    }

    while (fgets(curline, MAX_STRING, stdin) != (char*)NULL) {
        flag=isfilename(curline) ;
        if (flag == 1) {
            /* New file name is detected, get the new file name */
            printnames = 1 ;
            getfilename(curline, allrecords[nfiles].filename) ;
            nfiles++ ;
            
            /* Initialize a new record structure to store data for this file. */
            allrecords = (record*)realloc(allrecords,(nfiles+1)*sizeof(record));
            for (i=0 ; i<MAX_KEY ; i++) allrecords[nfiles].listkw[i].present=0;
        } else if (flag==0) {
            /* Is not a file name, is it a searched keyword?    */
            if ((kwnum = isdetectedkeyword(    curline, argv, argc-1)) != -1) {
                /* Is there anything allocated yet to store this? */
                if (nfiles>0) {
                    /* It has been detected as a searched keyword.  */
                    /* Get its value, store it, present flag up     */
                    getkeywordvalue(curline, word) ;
                    strcpy(allrecords[nfiles-1].listkw[kwnum].value, word) ;
                    allrecords[nfiles-1].listkw[kwnum].present ++ ;
                }
            }
        }
    }
    for (i=0 ; i<argc-1 ; i++) max_width[i] = (int)strlen(argv[i]) ;

    /* Record the maximum width for each column */
    max_filnam = 0 ;
    for (i=0 ; i<nfiles ; i++) {
        len = (int)strlen(allrecords[i].filename) ;
        if (len>max_filnam) max_filnam=len ;
        for (kwnum=0 ; kwnum<argc-1 ; kwnum++) {
            if (allrecords[i].listkw[kwnum].present) {
                len = (int)strlen(allrecords[i].listkw[kwnum].value) ;
            } else {
                len = 0 ;
            }
            if (len>max_width[kwnum]) max_width[kwnum] = len ;
        }
    }

    /* Print out header line */
    if (print_hdr) {
        sprintf(fmt, FMT_STRING, max_filnam) ;
        if (printnames) printf(fmt, "FILE");
        for (i=0 ; i<argc-1 ; i++) {
            sprintf(fmt, FMT_STRING, max_width[i]) ;
            printf(fmt, argv[i]) ;
        }
        printf("\n") ;
    }

    /* Now print out stored data    */
    if (nfiles<1) {
        printf("*** error: no input data corresponding to dfits output\n");
        return -1 ;
    }
    for (i=0 ; i<nfiles ; i++) {
        if (printnames) {
            sprintf(fmt, FMT_STRING, max_filnam) ;
            printf(fmt, allrecords[i].filename) ;
        }
        for (kwnum=0 ; kwnum<argc-1 ; kwnum++) {
            sprintf(fmt, FMT_STRING, max_width[kwnum]);
            if (allrecords[i].listkw[kwnum].present)
                printf(fmt, allrecords[i].listkw[kwnum].value) ;
            else printf(fmt, " ");
        }
        printf("\n") ;
    }
    free(allrecords) ;
    return 0 ;
}

/*-----------------------------------------------------------------------------
                              Functions code
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    find out if an input line contains a file name or a FITS magic nb.
  @param    string  dfits output line
  @return   int (see below)
  Filename recognition is based on 'dfits' output. The function returns 1 if 
  the line contains a valid file name as produced by dfits, 2 if it is for an
  extension, 0 otherwise.
 */
/*----------------------------------------------------------------------------*/
static int isfilename(char * string)
{
    if (!strncmp(string, "====>", 5)) return 1 ;
    if (!strncmp(string, "===>", 4)) return 2 ;
    return 0 ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    returns a file name from a dfits output line
  @param    line    dfits output line 
  @param    word    file name
  @return   a file name from a dfits output line
  This is dfits dependent.
 */
/*----------------------------------------------------------------------------*/
static void getfilename(char * line, char * word)
{
    /* get filename from a dfits output */
    sscanf(line, "%*s %*s %s", word) ;
    return ;
}
    
/*----------------------------------------------------------------------------*/
/**
  @brief    detects a if a keyword is present in a FITS line
  @param    line        FITS line
  @param    keywords    set of keywords
  @param    nkeys       number of kw in the set 
  @return   keyword rank, -1 if unidentified
  Feed this function a FITS line, a set of keywords in the *argv[] fashion 
  (*keywords[]). If the provided line appears to contain one of the keywords
  registered in the list, the rank of the keyword in the list is returned, 
  otherwise, -1 is returned.
 */
/*----------------------------------------------------------------------------*/
static int isdetectedkeyword(
        char    *   line, 
        char    *   keywords[], 
        int         nkeys)
{
    char    kw[MAX_STRING] ;
    char    esokw[MAX_STRING] ;
    int     i ;

    /* The keyword is up to the equal character, with trailing blanks removed */
    strcpy(kw, line) ;
    strtok(kw, "=") ;
    /* Now remove all trailing blanks (if any) */
    i = (int)strlen(kw) -1 ;
    while (kw[i] == ' ') i -- ;
    kw[i+1] = (char)0 ;
    
    /* Now compare what we got with what's available */
    for (i=0 ; i<nkeys ; i++) {
        if (strstr(keywords[i], ".")!=NULL) {
            /*
             * keyword contains a dot, it is a hierarchical keyword that
             * must be expanded. Pattern is:
             * A.B.C... becomes HIERARCH ESO A B C ...
             */
            expand_hierarch_keyword(keywords[i], esokw) ;
            if (!strcmp(kw, esokw)) {
                    return i ;
            }
        } else if (!strcmp(kw, keywords[i])) {
            return i ;
        }
    }
    /* Keyword not found    */
    return -1 ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Expand a HIERARCH keyword in format A.B.C to  HIERARCH ESO A B C
  @param    dotkey      Keyword
  @param    hierarchy   HIERARCH
  @return   char *, pointer to second input string (modified)
 */
/*----------------------------------------------------------------------------*/
static char * expand_hierarch_keyword(
        char    *    dotkey,
        char    *    hierarchy)
{
    char *  token ;
    char    ws[MAX_STRING] ;

    sprintf(hierarchy, "HIERARCH ESO");
    strcpy(ws, dotkey) ;
    token = strtok(ws, ".") ;
    while (token!=NULL) {
        strcat(hierarchy, " ") ;
        strcat(hierarchy, token) ;
        token = strtok(NULL, ".");
    }
    return hierarchy ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Get a keyword value within a FITS line
  @param    line    FITS line to process
  @param    word    char string to return result
  @return   void, result returned in word
  No complex value is recognized
 */
/*----------------------------------------------------------------------------*/
static void getkeywordvalue(
        char    *   line, 
        char    *   word)
{
    int     c, w ;
    char    tmp[MAX_STRING] ;
    char    *begin, *end ;
    int     length ;
    int     quote = 0 ;
    int     search = 1 ;

    memset(tmp, (char)0, MAX_STRING) ;
    memset(word, (char)0, MAX_STRING) ;
    c = w = 0;

    /* Parse the line till the equal '=' sign is found  */
    while (line[c] != '=') c++ ;
    c++ ;

    /* Copy the line till the slash '/' sign or the end of data is found.  */
    while (search == 1) {
        if (c>=80) search = 0 ;
        else if ((line[c] == '/') && (quote == 0)) search = 0 ;
        if (line[c] == '\'') quote = !quote ;
        tmp[w++] = line[c++] ;
    }
    
    /* NULL termination of the string   */
    tmp[--w] = (char)0 ;

    /* Return the keyword only : a diff is made between text fields and nbs. */
    if ((begin = strchr(tmp, '\'')) != (char*)NULL) {
        /* A quote has been found: it is a string value */
        begin++ ;
        end = strrchr(tmp, '\'') ;
        length = (int)strlen(begin) - (int)strlen(end) ;
        strncpy(word, begin, length) ;
    } else {
        /* No quote, just get the value (only one, no complex supported) */
        sscanf(tmp, "%s", word) ;
    }
        
    return ;
}

