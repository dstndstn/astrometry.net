/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <stdio.h>
#include <string.h>

#include "svn.h"

static char date_rtnval[256];
static char url_rtnval[256];

/* Ditto for "headurlstr". */
static const char* date = "$Date$";
static const char* url  = "$HeadURL$";
static const char* rev  = "$Revision$";

const char* svn_date() {
	// (trim off the first seven and last two characters.)
    strncpy(date_rtnval, date + 7, strlen(date) - 9);
    return date_rtnval;
}

int svn_revision() {
	int revnum;
	// rev+1 to avoid having "$" in the format string - otherwise svn seems to
	// consider it close enough to the Revision keyword anchor to do replacement!
    if (sscanf(rev + 1, "Revision: %i $", &revnum) != 1)
        return -1;
    return revnum;
}

const char* svn_url() {
    char* cptr;
    char* str = (char*)url + 10;
    cptr = str + strlen(str) - 1;
    // chomp off the filename...
    while (cptr > str && *cptr != '/') cptr--;
    strncpy(url_rtnval, str, cptr - str + 1);
    return url_rtnval;
}

// The Makefile automatically appends a blank comment line to the end
// of this file every time libanutils.a gets built.
//


//
//
//
