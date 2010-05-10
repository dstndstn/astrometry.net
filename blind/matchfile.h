/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef MATCHFILE_H
#define MATCHFILE_H

#include <stdio.h>

#include "matchobj.h"
#include "starutil.h"
#include "qfits.h"
#include "fitstable.h"
#include "bl.h"

#define AN_FILETYPE_MATCH "MATCH"

typedef fitstable_t matchfile;

pl* matchfile_get_matches_for_field(matchfile* m, int field);

matchfile* matchfile_open_for_writing(char* fn);

int matchfile_write_headers(matchfile* m);

int matchfile_fix_headers(matchfile* m);

int matchfile_write_match(matchfile* m, MatchObj* mo);

matchfile* matchfile_open(const char* fn);

int matchfile_read_matches(matchfile* m, MatchObj* mo, int offset, int n);

MatchObj* matchfile_read_match(matchfile* m);

int matchfile_pushback_match(matchfile* m);

int matchfile_count(matchfile* m);

int matchfile_close(matchfile* m);

#endif
