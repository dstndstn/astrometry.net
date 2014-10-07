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
#include "git.h"

#define GIT_URL "https://github.com/dstndstn/astrometry.net"
#define GIT_CMD_VERSION "git describe"
#define GIT_CMD_DATE "git log -1 --format=\"%cd\""

static const char* git_query(const char *cmd)
{
    FILE *fp;
    static char *ret = "";
    fp = popen(cmd, "r");
    if (fp == NULL) {
	perror("popen");
	return ret;
    }
    char line[256];
    while (fgets(line, sizeof(line)-1, fp) != NULL) {
	ret = strdup(line);
    }
    if (strlen(ret) > 0 && (ret[strlen(ret)-1] == '\n' ||
			    ret[strlen(ret)-1] == '\r'))
	ret[strlen(ret)-1] = '\0';
    pclose(fp);

    return ret;
}

const char* git_url()
{
    return GIT_URL;
}

const char *git_revision()
{
    return git_query(GIT_CMD_VERSION);
}

const char *git_date()
{
    return git_query(GIT_CMD_DATE);
}

