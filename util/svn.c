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
#include <strings.h>

#include "svn.h"
#include "an-thread.h"

/***
FIXME - consider using this recipe from the SVN book instead.

##
## on every build, record the working copy revision string
##
svn_version.c: FORCE
    echo -n 'const char* svn_version(void) { const char* SVN_Version = "' \
                                       > svn_version.c
    svnversion -n .                   >> svn_version.c
    echo '"; return SVN_Version; }'   >> svn_version.c

##
## Then any executable that links in svn_version.o will be able
## to call the function svn_version() to get a string that
## describes exactly what revision was built.
##

I would do instead of 'svnversion':
  svn info | tail -r | awk '/URL|Revision/{printf("%s ", $0)}'

 ***/

static char date_rtnval[256] = "";
static char url_rtnval[256] = "";
static int  rev_rtnval;

/* Ditto for "headurlstr". */
static const char* date = "$Date$";
static const char* url  = "$HeadURL$";
static const char* rev  = "$Revision$";

AN_THREAD_DECLARE_STATIC_ONCE(svn_once);

static void runonce(void) {
    const char* cptr;
    const char* str;

	// trim off the first seven and last two characters: "$" + "Date: " + DATE STRING + " $"
	// for non-svn (eg git-to-svn), DATE STRING (and probably the trailing space) are not there
	if (strlen(date) > 9)
		strncpy(date_rtnval, date + 7, strlen(date) - 9);

	// rev+1 to avoid having "$" in the format string - otherwise svn seems to
	// consider it close enough to the Revision keyword anchor to do replacement!
    if (sscanf(rev + 1, "Revision: %i $", &rev_rtnval) != 1)
        rev_rtnval = -1;

	// trim off the first ten characters: "$" + "HeadURL: " + URL + "svn.c $"
	if (strlen(url) > 10) {
		str = url + 10;
		// chomp off the filename by looking for the last '/'
		cptr = rindex(str, '/');
		if (cptr && (cptr > str+1))
			strncpy(url_rtnval, str, cptr - str + 1);
	}
}


const char* svn_date() {
	AN_THREAD_CALL_ONCE(svn_once, runonce);
    return date_rtnval;
}

int svn_revision() {
	AN_THREAD_CALL_ONCE(svn_once, runonce);
	return rev_rtnval;
}

const char* svn_url() {
	AN_THREAD_CALL_ONCE(svn_once, runonce);
    return url_rtnval;
}

// The Makefile automatically appends a blank comment line to the end
// of this file every time libanutils.a gets built.
//

//
//
