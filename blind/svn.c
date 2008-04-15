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

const char* svn_date() {
	/* Through the magic of Subversion, the date string on the following line will
	   be replaced by the correct, updated string whenever you run "svn up".  All hail
	   subversion! */
	/* That's not quite right - it seems you actually have to modify 
		the file.  Like this. Or this. Ok, this is getting silly. Very.
		Nuts, really. Absurd. Wacky. Ridiculous. Preposterous. Spectacular.
		Bizarre. Right out there. Loco. Homer-licking-a-hallucinogenic-toad-ish.
		Like-we-should-call-it-svnlsd.  (That last one was an inside joke.)
        I saw a poster once - it had a dog with a thought bubble that said,
        "Whoever said that dogs shouldn't take LSD definitely wasn't a dog on LSD."
        Hahahaha.  It was a dopey-looking golden retriever.  But then, aren't all
        golden retrievers fairly dopey looking?  But did you ever wonder whether
        golden retrievers think we're fairly dopey looking?  Are you starting to
        wonder how long this is going to go on?
	*/
	const char* datestr = "$Date$";
	// (I want to trim off the first seven and last two characters.)
	strncpy(date_rtnval, datestr + 7, strlen(datestr) - 9);
	return date_rtnval;
}

int svn_revision() {
	int rev;
	/* See the comment above; the same thing is true of "revstr". Huzzah! */
	const char* revstr = "$Revision$";
	// revstr+1 to avoid having "$" in the format string - otherwise svn seems to
	// consider it close enough to the Revision keyword anchor to do replacement!
	if (sscanf(revstr + 1, "Revision: %i $", &rev) != 1)
		return -1;
	return rev;
}

const char* svn_url() {
	/* Ditto for "headurlstr". */
	const char* headurlstr = "$HeadURL$";
	char* cptr;
	char* str = (char*)headurlstr + 10;
	cptr = str + strlen(str) - 1;
	while (cptr > str && *cptr != '/') cptr--;
	strncpy(url_rtnval, str, cptr - str + 1);
	return url_rtnval;
}
