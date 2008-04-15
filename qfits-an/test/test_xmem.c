/* $Id: test_xmem.c,v 1.4 2006/02/23 11:33:15 yjung Exp $
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
 * $Date: 2006/02/23 11:33:15 $
 * $Revision: 1.4 $
 * $Name: qfits-6_2_0 $
 */

/*---------------------------------------------------------------------------
                                Includes
 ---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "qfits_memory.h"

/*---------------------------------------------------------------------------
                                Define
 ---------------------------------------------------------------------------*/

#define PTRS_TOT_NB 20000
#define PTRS_SIZE 10
#define PTRS_TMP_NB 5
#define ITERATIONS_NB 5000

/*---------------------------------------------------------------------------
                                Main
 ---------------------------------------------------------------------------*/

int main(void)
{
    int     *   array[PTRS_TOT_NB] ;
    int     *   ptr[PTRS_TMP_NB] ;
    clock_t     chrono ;
    double      elapsed_time ;
    int         i, j ;

    /* Start chrono */
    chrono = clock() ;
    
    /* Allocate pointers */
    printf("----- RAM allocation\n");
    for (i=0 ; i<PTRS_TOT_NB ; i++) {
        array[i] = qfits_malloc(PTRS_SIZE*sizeof(int));
        array[i][0] = i ; 
    }

    /* Stop chrono */
    elapsed_time = (double)(clock() - chrono) / (double)CLOCKS_PER_SEC ;
    printf("\tFor allocation: %4.2f sec\n", elapsed_time) ;
    
    /* Some prints */
    for (i=0 ; i<3 ; i++) {
        printf("a[%d][0]=%d\n", i, array[i][0]);
    }
    printf("...\n");
    for (i=PTRS_TOT_NB-3 ; i<PTRS_TOT_NB ; i++) {
        printf("a[%d][0]=%d\n", i, array[i][0]);
    }
    
    /* Start chrono */
    chrono = clock() ;

    /* Test improvement of allocate/deallocate loop  */
    for (i=0 ; i<ITERATIONS_NB ; i++) {
        for (j=0 ; j<PTRS_TMP_NB ; j++) {
            ptr[j] = qfits_malloc(PTRS_SIZE*sizeof(int));
        }
        for (j=0 ; j<PTRS_TMP_NB ; j++) {
            qfits_free(ptr[j]) ;
        }
    }
    
    /* Stop chrono */
    elapsed_time = (double)(clock() - chrono) / (double)CLOCKS_PER_SEC ;
    printf("\tFor the alloc/dealloc loop: %4.2f sec\n", elapsed_time) ;
    
    /* Start chrono */
    chrono = clock() ;
    
    /* Deallocate pointers */
    printf("----- RAM deallocation\n");
    for (i=0 ; i<PTRS_TOT_NB ; i++) qfits_free(array[i]) ;
    
    /* Stop chrono */
    elapsed_time = (double)(clock() - chrono) / (double)CLOCKS_PER_SEC ;
    printf("\tFor deallocation: %4.2f sec\n", elapsed_time) ;

    qfits_memory_status() ;
    return 0 ;
}
