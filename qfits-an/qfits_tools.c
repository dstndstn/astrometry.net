/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_tools.c,v 1.12 2006/02/23 11:19:56 yjung Exp $
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
 * $Date: 2006/02/23 11:19:56 $
 * $Revision: 1.12 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <regex.h>

#include "qfits_config.h"
#include "qfits_tools.h"
#include "qfits_card.h"
#include "qfits_rw.h"
#include "qfits_std.h"
#include "qfits_error.h"
#include "qfits_memory.h"

/*-----------------------------------------------------------------------------
                            Global variables
 -----------------------------------------------------------------------------*/

size_t qfits_blocks_needed(size_t size) {
	return (size + FITS_BLOCK_SIZE - 1) / FITS_BLOCK_SIZE;
}

/*
 * The following global variables are only used for regular expression
 * matching of integers and floats. These definitions are private to
 * this module.
 */
/** A regular expression matching a floating-point number */
static const char* regex_float =
    "^[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eEdD][+-]?[0-9]+)?$";

/** A regular expression matching an integer */
static const char* regex_int = "^[+-]?[0-9]+$";

/** A regular expression matching a complex number (int or float) */
static const char* regex_cmp =
"^[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eEdD][+-]?[0-9]+)?[ ]+[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eEdD][+-]?[0-9]+)?$";

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_tools Simple FITS access routines 
 *
 *  This module offers a number of very basic low-level FITS access
 *  routines.
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                            Function codes
 -----------------------------------------------------------------------------*/

char* qfits_pretty_string_r(const char * s, char* pretty) {
    int             i,j;
	int slen;
    pretty[0] = '\0';
	if (!s) {
        return NULL;
    }
    if (s[0] != '\'') {
		strcpy(pretty, s);
		return pretty;
	}
	slen = strlen(s);

    /* skip first quote */
    i=1;
    j=0;
    /* trim left-side blanks */
    while (s[i]==' ') {
        if (i==slen) break;
        i++;
    }
    if (i >= (slen-1)) {
        return pretty;
    }
    /* copy string, changing double quotes to single ones */
    while (i<slen) {
        if (s[i]=='\'') {
            i++;
        }
        pretty[j]=s[i];
        i++;
        j++;
    }
    /* NULL-terminate the pretty string */
    pretty[j+1]='\0';
    /* trim right-side blanks */
    j = (int)strlen(pretty)-1;
    while (pretty[j]==' ') j--;
    pretty[j+1]=(char)0;
    return pretty;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Identify if a FITS value is boolean
  @param    s FITS value as a string
  @return   int 0 or 1

  Identifies if a FITS value is boolean.
 */
/*----------------------------------------------------------------------------*/
int qfits_is_boolean(const char * s)
{
    if (s==NULL) return 0;
    if (s[0]==0) return 0;
    if ((int)strlen(s)>1) return 0;
    if (s[0]=='T' || s[0]=='F') return 1;
    return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Identify if a FITS value is an int.
  @param    s FITS value as a string
  @return   int 0 or 1

  Identifies if a FITS value is an integer.
 */
/*----------------------------------------------------------------------------*/
int qfits_is_int(const char * s)
{
    regex_t re_int;
    int     status;

    if (s==NULL) return 0;
    if (s[0]==0) return 0;
    if (regcomp(&re_int, regex_int, REG_EXTENDED|REG_NOSUB)!=0) {
        qfits_error("internal error: compiling int rule");
        exit(-1);
    }
    status = regexec(&re_int, s, 0, NULL, 0);
    regfree(&re_int); 
    return (status) ? 0 : 1;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Identify if a FITS value is float.
  @param    s FITS value as a string
  @return   int 0 or 1

  Identifies if a FITS value is float.
 */
/*----------------------------------------------------------------------------*/
int qfits_is_float(const char * s)
{
    regex_t re_float;
    int     status;

    if (s==NULL) return 0;
    if (s[0]==0) return 0;
    if (regcomp(&re_float, regex_float, REG_EXTENDED|REG_NOSUB)!=0) {
        qfits_error("internal error: compiling float rule");
        exit(-1);
    }
    status = regexec(&re_float, s, 0, NULL, 0);
    regfree(&re_float); 
    return (status) ? 0 : 1;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Identify if a FITS value is complex.
  @param    s FITS value as a string
  @return   int 0 or 1

  Identifies if a FITS value is complex.
 */
/*----------------------------------------------------------------------------*/
int qfits_is_complex(const char * s)
{
    regex_t re_cmp;
    int     status;

    if (s==NULL) return 0;
    if (s[0]==0) return 0;
    if (regcomp(&re_cmp, regex_cmp, REG_EXTENDED|REG_NOSUB)!=0) {
        qfits_error("internal error: compiling complex rule");
        exit(-1);
    }
    status = regexec(&re_cmp, s, 0, NULL, 0);
    regfree(&re_cmp); 
    return (status) ? 0 : 1;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Identify if a FITS value is string.
  @param    s FITS value as a string
  @return   int 0 or 1

  Identifies if a FITS value is a string.
 */
/*----------------------------------------------------------------------------*/
int qfits_is_string(const char * s)
{
    if (s==NULL) return 0;
    if (s[0]=='\'') return 1;
    return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Identify the type of a FITS value given as a string.
  @param    s FITS value as a string
  @return   integer naming the FITS type

  Returns the following value:

  - QFITS_UNKNOWN (0) for an unknown type.
  - QFITS_BOOLEAN (1) for a boolean type.
  - QFITS_INT (2) for an integer type.
  - QFITS_FLOAT (3) for a floating-point type.
  - QFITS_COMPLEX (4) for a complex number.
  - QFITS_STRING (5) for a FITS string.
 */
/*----------------------------------------------------------------------------*/
int qfits_get_type(const char * s)
{
    if (s==NULL) return QFITS_UNKNOWN;
    if (qfits_is_boolean(s)) return QFITS_BOOLEAN;
    if (qfits_is_int(s)) return QFITS_INT;
    if (qfits_is_float(s)) return QFITS_FLOAT;
    if (qfits_is_complex(s)) return QFITS_COMPLEX;
    return QFITS_STRING;
}


/*----------------------------------------------------------------------------*/
/**
  @brief    Return the current QFITS version
  @return   the QFITS version
 */
/*----------------------------------------------------------------------------*/
const char * qfits_version(void)
{
    return (const char *)QFITS_VERSION;
}

/**@}*/
