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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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

/* FITS header constants */

/** FITS block size */
#define FITS_BLOCK_SIZE     (2880)
/** FITS number of cards per block */
#define FITS_NCARDS         (36)
/** FITS size of each line in bytes */
#define FITS_LINESZ         (80)

#endif
