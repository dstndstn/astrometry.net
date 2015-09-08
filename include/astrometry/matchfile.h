/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef MATCHFILE_H
#define MATCHFILE_H

#include <stdio.h>

#include "astrometry/matchobj.h"
#include "astrometry/starutil.h"
#include "astrometry/fitstable.h"
#include "astrometry/bl.h"

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
