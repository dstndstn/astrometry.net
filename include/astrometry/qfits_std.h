/* $Id: qfits_std.h,v 1.5 2006/02/17 13:51:52 yjung Exp $
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
 * $Date: 2006/02/17 13:51:52 $
 * $Revision: 1.5 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_STD_H
#define QFITS_STD_H

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

// does qfits think this platform is big-endian?
int qfits_is_platform_big_endian(void);

// Not POSIX; doesn't exist in Solaris 10
#include <sys/param.h>
#ifndef MIN
#define	MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
#define	MAX(a,b) (((a)>(b))?(a):(b))
#endif




#define QFITS_THREAD_UNSAFE {}

/* FITS header constants */

/** FITS block size */
#define FITS_BLOCK_SIZE     (2880)
/** FITS number of cards per block */
#define FITS_NCARDS         (36)
/** FITS size of each line in bytes */
#define FITS_LINESZ         (80)

#endif
