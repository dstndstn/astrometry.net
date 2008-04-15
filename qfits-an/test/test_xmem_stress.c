/* $Id: test_xmem_stress.c,v 1.5 2006/02/23 11:33:15 yjung Exp $
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
 * $Revision: 1.5 $
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
#include <sys/resource.h>

#include "qfits_memory.h"

/*---------------------------------------------------------------------------
                                Define
 ---------------------------------------------------------------------------*/

#define BIGNUM  100

/*---------------------------------------------------------------------------
                                Functions
 ---------------------------------------------------------------------------*/

/* Clocking variables and routines */
static clock_t  chrono ;
static void c_start(void)
{
    chrono = clock();
}
static void c_stop(void)
{
    double elapsed = (double)(clock()-chrono) / (double)CLOCKS_PER_SEC ;
    printf("elapsed: %5.3f sec\n", elapsed);
}


/*
 * First stress test: allocate a large number of small memory blocks and
 * free them all in reverse order.
 */
void stress1(void)
{
    int     i ;
    int *   ip[BIGNUM] ;

    for (i=0 ; i<BIGNUM ; i++) {
        ip[i] = qfits_malloc(256*1024*sizeof(int));
        printf("\rallocated %d", i);
        fflush(stdout);
    }
    qfits_memory_status();
    sleep(1);
    for (i=(BIGNUM-1) ; i>=0 ; i--) {
        qfits_free(ip[i]);
    }
}

void stress2(void)
{
    int     i ;
    int *   ip[BIGNUM] ;

    for (i=0 ; i<BIGNUM ; i++) {
        ip[i] = qfits_malloc(sizeof(int));
    }
    for (i=0 ; i<BIGNUM/2 ; i++) {
        qfits_free(ip[i]);
    }
    for (i=0 ; i<BIGNUM/2 ; i++) {
        ip[i] = qfits_malloc(sizeof(int));
    }
    for (i=0 ; i<BIGNUM ; i++) {
        qfits_free(ip[i]);
    }

}

int main(void)
{
    printf("Allocating/deallocating %d pointers\n", BIGNUM);
    c_start();
    stress1();
    c_stop();

    printf("Allocating/deallocating 1 pointer %d times\n", BIGNUM);
    c_start();
    stress2();
    c_stop();

    qfits_memory_status();
    return 0 ;
}
