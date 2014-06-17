/* $Id: qfits_byteswap.h,v 1.4 2006/02/17 10:24:52 yjung Exp $
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
 * $Date: 2006/02/17 10:24:52 $
 * $Revision: 1.4 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_BYTESWAP_H
#define QFITS_BYTESWAP_H

/*-----------------------------------------------------------------------------
                                Includes
 -----------------------------------------------------------------------------*/

#include <stdlib.h>

/*-----------------------------------------------------------------------------
                        Function ANSI C prototypes
 -----------------------------------------------------------------------------*/

unsigned short qfits_swap_bytes_16(unsigned short w);
unsigned int qfits_swap_bytes_32(unsigned int dw);
void qfits_swap_bytes(void * p, int s);

#endif
