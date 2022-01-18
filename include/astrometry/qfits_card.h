/* $Id: qfits_card.h,v 1.6 2006/02/20 09:45:25 yjung Exp $
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
 * $Revision: 1.6 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_CARD_H
#define QFITS_CARD_H

/*-----------------------------------------------------------------------------
                        Function ANSI C prototypes
 -----------------------------------------------------------------------------*/

void qfits_card_build(char *, const char *, const char *, const char *);

// NOT THREAD-SAFE
char* qfits_getvalue(const char*);
char* qfits_getkey(const char*);
char* qfits_expand_keyword(const char*);
char* qfits_getcomment(const char*);

// Thread-safe versions:
char* qfits_getvalue_r(const char *line, char* value);
char* qfits_getkey_r(const char *line, char* key);
char* qfits_expand_keyword_r(const char* keyword, char* expanded);
char* qfits_getcomment_r(const char * line, char* comment);



#endif
