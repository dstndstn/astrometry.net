/* $Id: qfits_rw.h,v 1.9 2006/02/20 09:45:25 yjung Exp $
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
 * $Date: 2006/02/20 09:45:25 $
 * $Revision: 1.9 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_RW_H
#define QFITS_RW_H

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include "astrometry/qfits_header.h"

/*-----------------------------------------------------------------------------
                        Function ANSI prototypes
 -----------------------------------------------------------------------------*/

qfits_header * qfits_header_read_hdr_string(const unsigned char *, int);
void qfits_zeropad(const char *);
int qfits_is_fits(const char *);
int is_blank_line(const char * s);

#endif
