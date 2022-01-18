/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_memory.h,v 1.3 2006/02/23 14:15:13 yjung Exp $
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
 * $Date: 2006/02/23 14:15:13 $
 * $Revision: 1.3 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_MEMORY_H
#define QFITS_MEMORY_H

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

#define qfits_malloc(s)         qfits_memory_malloc(s,      __FILE__,__LINE__)
#define qfits_calloc(n,s)       qfits_memory_calloc(n,s,    __FILE__,__LINE__)
#define qfits_realloc(p,s)      qfits_memory_realloc(p,s,   __FILE__,__LINE__)
#define qfits_free(p)           qfits_memory_free(p,        __FILE__,__LINE__)
#define qfits_strdup(s)         qfits_memory_strdup(s,      __FILE__,__LINE__)
#define qfits_falloc(f,o,s)     qfits_memory_falloc(f,o,s,  __FILE__,__LINE__)
#define qfits_fdealloc(f,o,s)   qfits_memory_fdealloc(f,o,s,__FILE__,__LINE__)

#define qfits_falloc2(f,o,s,fa,fs)     qfits_memory_falloc2(f,o,s,fa,fs,  __FILE__,__LINE__)
#define qfits_fdealloc2(p,s)   qfits_memory_fdealloc2(p,s,__FILE__,__LINE__)


/*-----------------------------------------------------------------------------
                               Function prototypes
 -----------------------------------------------------------------------------*/

/* *********************************************************************** */
/* These functions have to be called by the assiciated macro defined above */
void * qfits_memory_malloc(size_t, const char *, int);
void * qfits_memory_calloc(size_t, size_t, const char *, int);
void * qfits_memory_realloc(void *, size_t, const char *, int);
void   qfits_memory_free(void *, const char *, int);
char * qfits_memory_strdup(const char *, const char *, int);
char * qfits_memory_falloc(const char *, size_t, size_t *, const char *, int);
void qfits_memory_fdealloc(void *, size_t, size_t, const char *, int);
/* *********************************************************************** */

void* qfits_memory_falloc2(
	const char* name,
	size_t      offs,
	size_t      size,
	char** freeaddr,
	size_t* freesize,
	const char  *   srcname,
	int             srclin);
void qfits_memory_fdealloc2(
        void        *   ptr, 
		size_t len,
        const char  *   filename, 
        int             lineno);


void qfits_memory_status(void);
int qfits_memory_is_empty(void);

#endif
