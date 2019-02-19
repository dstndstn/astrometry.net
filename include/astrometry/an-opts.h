/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef AN_OPTS_H
#define AN_OPTS_H

#include <stdio.h>

#include "astrometry/bl.h"

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
