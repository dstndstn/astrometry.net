/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef ALLQUADS_H
#define ALLQUADS_H

#include "starkd.h"
#include "quadfile.h"
#include "codefile.h"

struct allquads {
    int dimquads;
    int dimcodes;
    int id;

    char *quadfn;
    char *codefn;
    char *skdtfn;

    startree_t* starkd;
    quadfile_t* quads;
    codefile_t* codes;

    double quad_d2_lower;
    double quad_d2_upper;
    anbool use_d2_lower;
    anbool use_d2_upper;

    int starA;
};
typedef struct allquads allquads_t;

allquads_t* allquads_init();
int allquads_open_outputs(allquads_t* aq);
int allquads_create_quads(allquads_t* aq);
int allquads_close(allquads_t* aq);
void allquads_free(allquads_t* aq);


#endif

