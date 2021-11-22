/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_header.h,v 1.8 2006/11/22 13:33:42 yjung Exp $
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
 * $Date: 2006/11/22 13:33:42 $
 * $Revision: 1.8 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_HEADER_H
#define QFITS_HEADER_H

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*-----------------------------------------------------------------------------
                                   New types
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    FITS header object

  This structure represents a FITS header in memory. It is actually no
  more than a thin layer on top of the keytuple object. No field in this
  structure should be directly modifiable by the user, only through
  accessor functions.
 */
/*----------------------------------------------------------------------------*/

struct qfits_header;
typedef struct qfits_header qfits_header;

/*-----------------------------------------------------------------------------
                        Function ANSI prototypes
 -----------------------------------------------------------------------------*/

void qfits_header_debug_dump(const qfits_header*);

int qfits_header_list(const qfits_header* hdr, FILE* out);
                      


qfits_header * qfits_header_new(void);
qfits_header * qfits_header_default(void);
int qfits_header_n(const qfits_header*);
void qfits_header_add(qfits_header *, const char *, const char *, const char *,
        const char *);
void qfits_header_add_after(qfits_header *, const char *, const char *, 
        const char *, const char *, const char *);
void qfits_header_append(qfits_header *, const char *, const char *,
        const char *, const char *);
void qfits_header_del(qfits_header *, const char *);
int qfits_header_sort(qfits_header **);
qfits_header * qfits_header_copy(const qfits_header *);
void qfits_header_mod(qfits_header *, const char *, const char *, const char *);
void qfits_header_destroy(qfits_header *);

char* qfits_header_getstr(const qfits_header *, const char *);

int qfits_header_getstr_pretty(const qfits_header* hdr, const char* key, char* pretty, const char* default_val);

int qfits_header_getitem(const qfits_header *, int, char *, char *, char *, 
        char *); 

/*
  Note, the "key", "val", "comment", args are copied with "strdup", while "line" is
  copied with "memcpy(dest, line, 80)", so you must ensure that the string you pass in
  has at least 80 chars.
 */
int qfits_header_setitem(qfits_header *, int, char* key, char* val, char* comment,
                         char* line);

char * qfits_header_getcom(const qfits_header *, const char *);
int qfits_header_getint(const qfits_header *, const char *, int);
double qfits_header_getdouble(const qfits_header *, const char *, double);
int qfits_header_getboolean(const qfits_header *, const char *, int);
int qfits_header_dump(const qfits_header *, FILE *);
char * qfits_header_findmatch(const qfits_header * hdr, const char * key);

int qfits_header_write_line(const qfits_header* hdr, int line, char* result);

#endif
