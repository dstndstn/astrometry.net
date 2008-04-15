/* $Id: frameq.c,v 1.12 2006/02/17 13:41:49 yjung Exp $
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
 * $Revision: 1.12 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <glob.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "qfits_tools.h"
#include "qfits_filename.h"

/*-----------------------------------------------------------------------------
                                   New types
 -----------------------------------------------------------------------------*/

/* Frame informations needed to classify it */
typedef struct _framei {
    char    *   name ;
    char    *   tplid ;
    char    *   origfile ;
    int         expno ;
    int         nexp ;

    struct _framei * next ;
} framei ;

/* Frame queue: is mostly a pointer to a list of frame information objects */
typedef struct _frameq {
    framei * first ;
    int      n ;
} frameq ;

/*-----------------------------------------------------------------------------
                               Global variables    
 -----------------------------------------------------------------------------*/

/* List of strings to identify unwanted templates. */
char * tpl_filter[] = {
    "_acq_",
    "_CheckAoCorrection",
    NULL
};

/*-----------------------------------------------------------------------------
                              Functions code
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Build a new framei object
  @param    filename    FITS file name
  @return   A newly allocated framei obj. NULL in error case
 */
/*----------------------------------------------------------------------------*/
static framei * framei_new(char * filename)
{
    framei  *   fi ;
    char    *   sval ;
    char    *   sval2 ;
    int         i ;

    fi = malloc(sizeof(framei));
    fi->name = strdup(filename);
    sval = qfits_query_hdr(filename, "tpl.id");
    if (sval!=NULL) {
        /* Filter out unwanted template IDs */
        i=0 ;
        while (tpl_filter[i]!=NULL) {
            if (strstr(sval, tpl_filter[i])!=NULL) return NULL ;
            i++ ;
        }
        fi->tplid = strdup(qfits_pretty_string(sval)) ;
    } else {
        fi->tplid = NULL ;
    }
    sval = qfits_query_hdr(filename, "origfile");
    if (sval!=NULL) {
        sval2 = qfits_pretty_string(sval) ;
        fi->origfile = strdup(qfits_get_root_name(sval2)) ;
    } else {
        fi->origfile = NULL ;
    }
    sval = qfits_query_hdr(filename, "tpl.expno");
    if (sval!=NULL) fi->expno = (int)atoi(sval);
    else fi->expno = -1 ;
    sval = qfits_query_hdr(filename, "tpl.nexp");
    if (sval!=NULL) fi->nexp = (int)atoi(sval);
    else fi->nexp = -1 ;

    fi->next = NULL ;
    return fi ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Delete a framei object
  @param    fi  Object to delete
 */
/*----------------------------------------------------------------------------*/
static void framei_del(framei * fi)
{
    if (fi==NULL) return ;
    if (fi->name!=NULL)
        free(fi->name);
    if (fi->tplid!=NULL) {
        free(fi->tplid);
    }
    if (fi->origfile!=NULL) {
        free(fi->origfile);
    }
    free(fi);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Create a frameq object
  @return   A newly allocated empty frameq object
 */
/*----------------------------------------------------------------------------*/
frameq * frameq_new(void)
{
    frameq  *   fq ;

    fq = malloc(sizeof(frameq));
    fq->first = NULL ;
    fq->n = 0 ;
    return fq ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Delete a frameq object
  @param    fq  The object to delete
 */
/*----------------------------------------------------------------------------*/
void frameq_del(frameq * fq)
{
    framei  *   fi ;
    framei  *   fin ;

    if (fq==NULL) return ;

    fi = fq->first ;
    while (fi!=NULL) {
        fin = fi->next ;
        framei_del(fi);
        fi = fin ;
    }
    free(fq);
    return ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Append a new frame to an existing frame queue
  @param    fq          The existing frame queue
  @param    filename    The file to append name
 */
/*----------------------------------------------------------------------------*/
void frameq_append(frameq * fq, char * filename)
{
    framei  *   fi ;
    framei  *   fn ;

    if (fq==NULL || filename==NULL) return ;
    fi = framei_new(filename);
    if (fi==NULL)
        return ;
    if (fq->n==0) {
        fq->first = fi ;
        fq->n = 1 ;
        return ;
    }

    fn = fq->first ;
    while (fn->next!=NULL)
        fn = fn->next ;
    
    fn->next = fi ;
    fq->n ++ ;
    return ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Pop a frame out of the queue
  @param    fq  The frame queue
 */
/*----------------------------------------------------------------------------*/
void frameq_pop(frameq * fq)
{
    framei * first ;

    first = fq->first->next ;
    framei_del(fq->first);
    fq->first = first ;
    fq->n -- ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Dump a frame queue in a file
  @param    fq  Frame queue to dump
  @param    out File where the queue is dumped
 */
/*----------------------------------------------------------------------------*/
void frameq_dump(frameq * fq, FILE * out)
{
    int      i ;
    framei * fi ;

    fi = fq->first ;
    for (i=0 ; i<fq->n ; i++) {
        fprintf(out,
                "%s %s %02d/%02d\n",
                fi->name,
                fi->tplid,
                fi->expno,
                fi->nexp);
        fi = fi->next ;
    }
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Create a frame queue from a directory
  @param    dirname     Directory name
  @return   A newly allocated frame queue
 */
/*----------------------------------------------------------------------------*/
frameq * frameq_load(char * dirname)
{
    frameq * fq ;
    int      i ;
    glob_t   pglob ;
    char     filename[512];

    /* Look for *.fits *.FITS *.fits.gz *.FITS.gz */
    sprintf(filename, "%s/*.fits", dirname);
    glob(filename, GLOB_MARK, NULL, &pglob);
    sprintf(filename, "%s/*.FITS", dirname);
    glob(filename, GLOB_APPEND, NULL, &pglob);
    sprintf(filename, "%s/*.fits.gz", dirname);
    glob(filename, GLOB_APPEND, NULL, &pglob);
    sprintf(filename, "%s/*.FITS.gz", dirname);
    glob(filename, GLOB_APPEND, NULL, &pglob);
    if (pglob.gl_pathc<1) {
        printf("found no frame\n");
        return NULL ;
    }

    /* Build frame queue */
    fq = frameq_new();
    for (i=0 ; i<pglob.gl_pathc ; i++) {
        printf("\radding %d of %u", i+1, (unsigned)pglob.gl_pathc);
        fflush(stdout);
        frameq_append(fq, pglob.gl_pathv[i]);
    }
    printf("\n");
    globfree(&pglob);
    return fq ;
}

static int stringsort(const void * e1, const void * e2)
{
    return strcmp(*(char**)e1, *(char**)e2);
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Get TPL.ID keywords from a frame queue
  @param    fq      The frame queue
  @param    n       Number of values in output 
  @return   List of n strings
 */
/*----------------------------------------------------------------------------*/
char ** frameq_get_tplid(frameq * fq, int * n)
{
    framei  *   fn ;
    char    **  tplid_all ;
    char    **  tplid ;
    int         i, j ;
    int         ntplid ;

    /* Get all possible values for tplid */
    tplid_all = malloc(fq->n * sizeof(char*));
    fn = fq->first ;
    for (i=0 ; i<fq->n ; i++) {
        if (fn->tplid==NULL) {
            tplid_all[i] = strdup("none");
        } else {
            tplid_all[i] = strdup(fn->tplid);
        }
        fn = fn->next ;
    }
    /* Sort all tplid's */
    qsort(tplid_all, fq->n, sizeof(char*), stringsort);

    /* Compute how many different tplid's can be found */
    ntplid=1 ;
    for (i=1 ; i<fq->n ; i++) { 
        if (strcmp(tplid_all[i], tplid_all[i-1])) {
            ntplid++ ;
        }
    }

    tplid = malloc(ntplid * sizeof(char*));
    tplid[0] = tplid_all[0] ;
    tplid_all[0] = NULL ;
    j=0 ;
    for (i=1 ; i<fq->n ; i++) { 
        if (strcmp(tplid_all[i], tplid[j])) {
            j++ ;
            tplid[j] = tplid_all[i] ;
        } else {
            free(tplid_all[i]);
        }
        tplid_all[i] = NULL ;
    }
    free(tplid_all);

    *n = ntplid ;
    return tplid ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Get the setting number
  @param    dirname     Directory name
  @return   Setting number
 */
/*----------------------------------------------------------------------------*/
int frameq_getsetnum(char * dirname)
{
    char    pattern[512];
    glob_t  pglob ;
    int     i ;
    int     max ;
    int     num ;

    sprintf(pattern, "%s/set*", dirname);
    glob(pattern, GLOB_MARK, NULL, &pglob);
    if (pglob.gl_pathc<1) {
        max=0 ;
    } else {
        sprintf(pattern, "%s/set%%02d", dirname);
        max=0 ;
        for (i=0 ; i<pglob.gl_pathc ; i++) {
            sscanf(pglob.gl_pathv[i], pattern, &num);
            if (num>max)
                max=num ;
        }
    }
    globfree(&pglob);
    return max+1 ;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Classify the frames
  @param    fq  Frame queue
 */
/*----------------------------------------------------------------------------*/
void frameq_makelists(frameq * fq)
{
    FILE    *   list ;
    char        filename[512];
    framei  *   fi ;
    int         setnum ;
    int         count ;
    int         batches ;

    /* Count # of batches in input */
    fi = fq->first ;
    batches=0 ;
    while (fi!=NULL) {
        if (fi->expno==1)
            batches++ ;
        fi = fi->next ;
    }

    fi = fq->first ;
    count=0 ;
    list=NULL ;
    while (fi!=NULL) {
        printf("\rclassifying batches: %d of %d", count, batches);
        fflush(stdout);
        if (fi->expno<0) {
            fi=fi->next ;
            continue ;
        }
        if (fi->expno==1) {
            count++ ;
            if (list!=NULL) {
                fclose(list);
            }
            if (fi->tplid == NULL) {
                printf("No TPL ID - abort\n") ;
                return ;
            }
            if (fi->origfile == NULL) {
                printf("No ORIGFILE - abort\n") ;
                return ;
            }
            mkdir(fi->tplid, 0755);
            setnum = frameq_getsetnum(fi->tplid);
            sprintf(filename, "%s/%s_%02d", fi->tplid, fi->origfile, fi->nexp);
            mkdir(filename, 0755);
            sprintf(filename, "%s/%s_%02d/IN", fi->tplid,fi->origfile,fi->nexp);
            list = fopen(filename, "w");
            fprintf(list,"# TPL.ID= %s\n", fi->tplid);
            fprintf(list,"# NEXP  = %02d\n", fi->nexp);
        }
        if (list) fprintf(list, "%s\n", fi->name);
        fi = fi->next ;
    }
    printf("\n");
    return ;
}

/*-----------------------------------------------------------------------------
                                   Main    
 -----------------------------------------------------------------------------*/
int main(int argc, char * argv[])
{
    frameq  *   fq ;

    if (argc<2) {
        printf("use: %s <dirname>\n", argv[0]);
        return 1 ;
    }
    
    printf("loading frames from %s...\n", argv[1]);
    fq = frameq_load(argv[1]);
    printf("processing lists...\n");
    frameq_makelists(fq);
    frameq_del(fq);
    printf("done\n");

    return 0 ;
}

