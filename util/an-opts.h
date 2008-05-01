/*
 This file is part of the Astrometry.net suite.
 Copyright 2008 Dustin Lang.

 The Astrometry.net suite is free software; you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA	 02110-1301 USA
 */

#ifndef AN_OPTS_H
#define AN_OPTS_H

#include <stdio.h>

#include "bl.h"

struct anoption {
    // don't change the order of these fields!
    // static initializers depend on the ordering.
    char shortopt;
    const char *name;
    int has_arg;
    const char* argname;
    const char* help;
};
typedef struct anoption an_option_t;

void opts_print_help(bl* list_of_opts, FILE* fid,
                     void (*special_case)(an_option_t* opt, bl* allopts, int index,
                                          FILE* fid, void* extra), void* extra);

int opts_getopt(bl* list_of_opts, int argc, char** argv);

bl* opts_from_array(const an_option_t* opts, int N, bl* lst);

#endif
