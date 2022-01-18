/* $Id: qfits_byteswap.c,v 1.5 2006/02/17 10:24:52 yjung Exp $
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
 * $Date: 2006/02/17 10:24:52 $
 * $Revision: 1.5 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include "qfits_config.h"
#include "qfits_byteswap.h"

// does qfits think this platform is big-endian?
int qfits_is_platform_big_endian() {
#ifdef WORDS_BIGENDIAN
    return 1;
#else
    return 0;
#endif
}


/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_byteswap  Low-level byte-swapping routines
 *
 *  This module offers access to byte-swapping routines.
 *  Generic routines are offered that should work everywhere.
 *  Assembler is also included for x86 architectures, and dedicated
 *  assembler calls for processors > 386.
 *
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                              Function codes
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Swap a 16-bit number
  @param    w A 16-bit (short) number to byte-swap.
  @return   The swapped version of w, w is untouched.

  This function swaps a 16-bit number, returned the swapped value without
  modifying the passed argument. Assembler included for x86 architectures.
 */
/*----------------------------------------------------------------------------*/
unsigned short qfits_swap_bytes_16(unsigned short w)
{
#ifdef CPU_X86
    __asm("xchgb %b0,%h0" :
            "=q" (w) :
            "0" (w));
    return w;
#else
    return (((w) & 0x00ff) << 8 | ((w) & 0xff00) >> 8);
#endif
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Swap a 32-bit number
  @param    dw A 32-bit (long) number to byte-swap.
  @return   The swapped version of dw, dw is untouched.

  This function swaps a 32-bit number, returned the swapped value without
  modifying the passed argument. Assembler included for x86 architectures
  and optimized for processors above 386.
 */
/*----------------------------------------------------------------------------*/
unsigned int qfits_swap_bytes_32(unsigned int dw)
{
#ifdef CPU_X86
#if CPU_X86 > 386
    __asm("bswap   %0":
            "=r" (dw)   :
#else
    __asm("xchgb   %b0,%h0\n"
        " rorl    $16,%0\n"
        " xchgb   %b0,%h0":
        "=q" (dw)      :
#endif
        "0" (dw));
    return dw;
#else
    return ((((dw) & 0xff000000) >> 24) | (((dw) & 0x00ff0000) >>  8) |
            (((dw) & 0x0000ff00) <<  8) | (((dw) & 0x000000ff) << 24));
#endif
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Swaps bytes in a variable of given size
  @param    p pointer to void (generic pointer)
  @param    s size of the element to swap, pointed to by p
  @return    void

  This byte-swapper is portable and works for any even variable size.
  It is not truly the most efficient ever, but does its job fine
  everywhere this file compiles.
 */
/*----------------------------------------------------------------------------*/
void qfits_swap_bytes(void * p, int s)
{
    unsigned char tmp, *a, *b;

    a = (unsigned char*)p;
    b = a + s;

    while (a<b) {
        tmp = *a;
        *a++ = *--b;
        *b = tmp;
    }
}

/**@}*/
