/* $Id: qfits_cache.h,v 1.4 2006/02/20 09:45:25 yjung Exp $
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
 * $Date: 2006/02/20 09:45:25 $
 * $Revision: 1.4 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_CACHE_H
#define QFITS_CACHE_H

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

/** Query the number of extensions */
#define QFITS_QUERY_N_EXT        (1<<30)
/** Query the offset to header start */
#define QFITS_QUERY_HDR_START    (1<<29)
/** Query the offset to data start */
#define QFITS_QUERY_DAT_START    (1<<28)
/** Query header size in bytes */
#define QFITS_QUERY_HDR_SIZE    (1<<27)
/** Query data size in bytes */
#define QFITS_QUERY_DAT_SIZE    (1<<26)

void qfits_cache_purge(void) ;
int qfits_query(const char *, int) ;

#endif
