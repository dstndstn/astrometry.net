/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_card.c,v 1.8 2006/02/20 09:45:25 yjung Exp $
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
 * $Revision: 1.8 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#include "qfits_std.h"
#include "qfits_card.h"
#include "qfits_tools.h"
#include "qfits_error.h"

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

/* Define the following to get zillions of debug messages */
/* #define DEBUG_FITSHEADER */

/*-----------------------------------------------------------------------------
                              Static functions
 -----------------------------------------------------------------------------*/

static char* expkey_strupc(const char *, char* buf);

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_card   Card handling functions
 *
 * This module contains various routines to help parsing a single FITS
 * card into its components: key, value, comment.
 *
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                              Function codes
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Write out a card to a string on 80 chars.
  @param    line    Allocated output character buffer.
  @param    key     Key to write.
  @param    val     Value to write.
  @param    com     Comment to write.
  @return   void

  Write out a key, value and comment into an allocated character buffer.
  The buffer must be at least 80 chars to receive the information.
  Formatting is done according to FITS standard.
 */
/*----------------------------------------------------------------------------*/
void qfits_card_build(
        char        *   line,
        const char  *   key,
        const char  *   val,
        const char  *   com)
{
    int     len;
    int     hierarch = 0;
    char    cval[81];
    char    cval2[81];
    char    cval_q[81];
    char    ccom[81];
    char    safe_line[512];
    int     i, j;

    if (line==NULL || key==NULL) return;

    /* Set the line with zeroes */
    memset(line, ' ', 80);
    if (key==NULL) return;

    /* END keyword*/
    if (!strcmp(key, "END")) {
        /* Write key and return */
        sprintf(line, "END");
        return;
    }
    /* HISTORY, COMMENT and blank keywords */
    if (!strcmp(key, "HISTORY") ||
        !strcmp(key, "COMMENT") ||
        !strcmp(key, "CONTINUE") ||
        !strncmp(key, "        ", 8)) {
        /* Write key */
        sprintf(line, "%s ", key);
        if (val==NULL) return;

        /* There is a value to write, copy it correctly */
        len = strlen(val);
        /* 72 is 80 (FITS line size) - 8 (sizeof COMMENT or HISTORY) */
        if (len>72) len=72;
        strncpy(line+8, val, len);
        return;
    }

    /* Check for NULL values */
    if (val==NULL) cval[0]='\0';
    else if (strlen(val)<1) cval[0]='\0';
    else strcpy(cval, val);

    /* Check for NULL comments */
    if (com==NULL) strcpy(ccom, "no comment");
    else strcpy(ccom, com);

    /* Set hierarch flag */
    if (!strncmp(key, "HIERARCH", 8)) hierarch ++;

    /* Boolean, int, float or complex */
    if (qfits_is_int(cval) ||
            qfits_is_float(cval) ||
            qfits_is_boolean(cval) ||
            qfits_is_complex(cval)) {
        if (hierarch) sprintf(safe_line, "%-29s= %s / %s", key, cval, ccom);
        else sprintf(safe_line, "%-8.8s= %20s / %-48s", key, cval, ccom);
        strncpy(line, safe_line, 80);
        line[80]='\0';
        return;
    }

    /* Blank or NULL values */
    if (cval[0]==0) {
        if (hierarch) {
            sprintf(safe_line, "%-29s=                    / %s", key, ccom);
        } else {
        sprintf(safe_line, "%-8.8s=                      / %-48s", key, ccom);
        }
        strncpy(line, safe_line, 80);
        line[80]='\0';
        return;
    }

    /* Can only be a string - Make simple quotes ['] as double [''] */
    memset(cval_q, 0, 81);
    qfits_pretty_string_r(cval, cval2);
    j=0;
    i=0;
    while (cval2[i] != '\0') {
        if (cval2[i]=='\'') {
            cval_q[j]='\'';
            j++;
            cval_q[j]='\'';
        } else {
            cval_q[j] = cval2[i];
        }
        i++;
        j++;
    }

    if (hierarch) {
        sprintf(safe_line, "%-29s= '%s' / %s", key, cval_q, ccom);
        if (strlen(key) + strlen(cval_q) + 3 >= 80)
            safe_line[79] = '\'';
    } else {
        sprintf(safe_line, "%-8.8s= '%-8s' / %s", key, cval_q, ccom);
    }
    strncpy(line, safe_line, 80);

    /* Null-terminate in any case */
    line[80]='\0';
    return;
}

// Thread-safe version.
char* qfits_getkey_r(const char* line, char* key)
{
    int                i;

    if (line==NULL) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getkey: NULL input line\n");
#endif
        return NULL;
    }

    /* Special case: blank keyword */
    if (!strncmp(line, "        ", 8)) {
        strcpy(key, "        ");
        return key;
    }
    /* Sort out special cases: HISTORY, COMMENT, END do not have = in line */
    if (!strncmp(line, "HISTORY ", 8)) {
        strcpy(key, "HISTORY");
        return key;
    }
    if (!strncmp(line, "COMMENT ", 8)) {
        strcpy(key, "COMMENT");
        return key;
    }
    if (!strncmp(line, "END ", 4)) {
        strcpy(key, "END");
        return key;
    }
	/* Neither does CONTINUE. */
    if (!strncmp(line, "CONTINUE ", 9)) {
        strcpy(key, "CONTINUE");
        return key;
    }

    memset(key, 0, 81);
    /* General case: look for the first equal sign */
    i=0;
    while (line[i]!='=' && i<80) i++;
    if (i>=80) {
        qfits_error("qfits_getkey: cannot find equal sign in line: \"%.80s\"\n", line);
        return NULL;
    }
    i--;
    /* Equal sign found, now backtrack on blanks */
    while (line[i]==' ' && i>=0) i--;
    if (i<0) {
        qfits_error("qfits_getkey: error backtracking on blanks in line: \"%s\"\n", line);
        return NULL;
    }
    i++;

    /* Copy relevant characters into output buffer */
    strncpy(key, line, i);
    /* Null-terminate the string */
    key[i+1] = '\0';
    return key;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Find the keyword in a key card (80 chars)    
  @param    line allocated 80-char line from a FITS header
  @return    statically allocated char *

  Find out the part of a FITS line corresponding to the keyword.
  Returns NULL in case of error. The returned pointer is statically
  allocated in this function, so do not modify or try to free it.
 */
/*----------------------------------------------------------------------------*/
char * qfits_getkey(const char * line) {
    static char     key[81];
	return qfits_getkey_r(line, key);
}

// Thread-safe version of the below.
char* qfits_getvalue_r(const char* line, char* value) {
    int     i;
    int     from, to;
    int     inq;

    if (line==NULL) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getvalue: NULL input line\n");
#endif
        return NULL;
    }

    /* Special cases */

    /* END has no associated value */
    if (!strncmp(line, "END ", 4)) {
        return NULL;
    }
    /*
     * HISTORY has for value everything else on the line.
     */
    memset(value, 0, 81);

    if (!strncmp(line, "HISTORY ", 8) ||
        !strncmp(line, "        ", 8) ||
        !strncmp(line, "COMMENT ", 8) ||
        !strncmp(line, "CONTINUE", 8)) {
        strncpy(value, line+8, 80-8);
        return value;
    }
    /* General case - Get past the keyword */
    i=0;
    while (i<80 && line[i]!='=') i++;
    if (i>80) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getvalue: no equal sign found on line\n");
#endif
        return NULL;
    }
    i++;
    while (i<80 && line[i]==' ') i++;
    if (i>80) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getvalue: no value past the equal sign\n");
#endif
        return NULL;
    }
    from=i;

    /* Now value section: Look for the first slash '/' outside a string */
    inq = 0;
    while (i<80) {
        if (line[i]=='\'')
            inq=!inq;
        if (line[i]=='/')
            if (!inq)
                break;
        i++;
    }
    i--;

    /* Backtrack on blanks */
    while (i>=0 && line[i]==' ') i--;
    if (i<0) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getvalue: error backtracking on blanks\n");
#endif
        return NULL;
    }
    to=i;

    if (to<from) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getvalue: from>to?\n");
        printf("line=[%s]\n", line);
#endif
        return NULL;
    }
    /* Copy relevant characters into output buffer */
    strncpy(value, line+from, to-from+1);
    /* Null-terminate the string */
    value[to-from+1] = '\0';
    return value;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Find the value in a key card (80 chars)    
  @param    line allocated 80-char line from a FITS header
  @return    statically allocated char *

  Find out the part of a FITS line corresponding to the value.
  Returns NULL in case of error, or if no value can be found. The
  returned pointer is statically allocated in this function, so do not
  modify or try to free it.
 */
/*----------------------------------------------------------------------------*/
char* qfits_getvalue(const char* line) {
    static char value[81];
	return qfits_getvalue_r(line, value);
}

char* qfits_getcomment_r(const char* line, char* comment) {
    int    i;
    int    from, to;
    int    inq;

    if (line==NULL) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getcomment: null line in input\n");
#endif
        return NULL;
    }

    /* Special cases: END, HISTORY, COMMENT and blank have no comment */
    if (!strncmp(line, "END ", 4)) return NULL;
    if (!strncmp(line, "HISTORY ", 8)) return NULL;
    if (!strncmp(line, "COMMENT ", 8)) return NULL;
    if (!strncmp(line, "        ", 8)) return NULL;

    memset(comment, 0, 81);
    /* Get past the keyword */
    i=0;
    while (i<80 && line[i]!='=') i++;
    if (i>=80) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getcomment: no equal sign on line\n");
#endif
        return NULL;
    }
    i++;
    
    /* Get past the value until the slash */
    inq = 0;
    while (i<80) {
        if (line[i]=='\'')
            inq = !inq;
        if (line[i]=='/')
            if (!inq)
                break;
        i++;
    }
    if (i>=80) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getcomment: no slash found on line\n");
#endif
        return NULL;
    }
    i++;
    /* Get past the first blanks */
    while (line[i]==' ') i++;
    from=i;

    /* Now backtrack from the end of the line to the first non-blank char */
    to=79;
    while (line[to]==' ') to--;

    if (to<from) {
#ifdef DEBUG_FITSHEADER
        printf("qfits_getcomment: from>to?\n");
#endif
        return NULL;
    }
    /* Copy relevant characters into output buffer */
    strncpy(comment, line+from, to-from+1);
    /* Null-terminate the string */
    comment[to-from+1] = '\0';
    return comment;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Find the comment in a key card (80 chars)    
  @param    line allocated 80-char line from a FITS header
  @return    statically allocated char *

  Find out the part of a FITS line corresponding to the comment.
  Returns NULL in case of error, or if no comment can be found. The
  returned pointer is statically allocated in this function, so do not
  modify or try to free it.
 */
/*----------------------------------------------------------------------------*/
char * qfits_getcomment(const char * line) {
    static char comment[81];
	return qfits_getcomment_r(line, comment);
}

/*----------------------------------------------------------------------------*/
/**
   Thread-safe version of qfits_expand_keyword.
 */
char* qfits_expand_keyword_r(const char * keyword, char* expanded) {
    char        ws[81];
    char    *    token;

    /* Bulletproof entries */
    if (keyword==NULL) return NULL;
    /* If regular keyword, copy the uppercased input and return */
    if (strstr(keyword, ".")==NULL) {
        expkey_strupc(keyword, expanded);
        return expanded;
    }
    /* Regular shortFITS keyword */
    sprintf(expanded, "HIERARCH ESO");
    expkey_strupc(keyword, ws);
    token = strtok(ws, ".");
    while (token!=NULL) {
        strcat(expanded, " ");
        strcat(expanded, token);
        token = strtok(NULL, ".");
    }
    return expanded;
}


/*----------------------------------------------------------------------------*/
/**
  @brief    Expand a keyword from shortFITS to HIERARCH notation.
  @param    keyword        Keyword to expand.
  @return    1 pointer to statically allocated string.

  This function expands a given keyword from shortFITS to HIERARCH
  notation, bringing it to uppercase at the same time.

  Examples:

  @verbatim
  det.dit          expands to     HIERARCH ESO DET DIT
  ins.filt1.id     expands to     HIERARCH ESO INS FILT1 ID
  @endverbatim

  If the input keyword is a regular FITS keyword (i.e. it contains
  not dots '.') the result is identical to the input.
 */
/*----------------------------------------------------------------------------*/
char * qfits_expand_keyword(const char * keyword) {
    static char expanded[81];
    QFITS_THREAD_UNSAFE;
    qfits_expand_keyword_r(keyword, expanded);
    return expanded;
}
/**@}*/


/*----------------------------------------------------------------------------*/
/**
  @brief    Uppercase a string
  @param    s   string
  @return   string
 */
/*----------------------------------------------------------------------------*/
static char * expkey_strupc(const char * s, char* l) {
    int i;
    if (s==NULL) return NULL;
    i=0;
    while (s[i]) {
        l[i] = (char)toupper((int)s[i]);
        i++;
    }
    l[i] = '\0';
    return l;
}

