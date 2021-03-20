/* $Id: qfits_tools.h,v 1.7 2006/02/20 09:45:25 yjung Exp $
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
 * $Revision: 1.7 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_TOOLS_H
#define QFITS_TOOLS_H

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

/* Unknown type for FITS value */
#define QFITS_UNKNOWN       0
/* Boolean type for FITS value */
#define QFITS_BOOLEAN       1
/* Int type for FITS value */
#define    QFITS_INT        2
/* Float type for FITS value */
#define QFITS_FLOAT         3
/* Complex type for FITS value */
#define QFITS_COMPLEX       4
/* String type for FITS value */
#define QFITS_STRING        5

#include "astrometry/qfits_std.h"

size_t qfits_blocks_needed(size_t size);

/*-----------------------------------------------------------------------------
                              Function codes
 -----------------------------------------------------------------------------*/

char* qfits_pretty_string_r(const char* in, char* out);
int qfits_is_boolean(const char *);
int qfits_is_int(const char *);
int qfits_is_float(const char *);
int qfits_is_complex(const char *);
int qfits_is_string(const char *);
int qfits_get_type(const char *);
const char * qfits_version(void);

#endif
