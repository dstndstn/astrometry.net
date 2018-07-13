/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef CODETREE_H
#define CODETREE_H

#include "astrometry/codekd.h"
#include "astrometry/codefile.h"
#include "astrometry/fitstable.h"

/**
 */
codetree_t* codetree_build(codefile_t* codes,
                           int Nleaf, int datatype, int treetype,
                           int buildopts,
                           char** args, int argc);

int codetree_files(const char* codefn, const char* ckdtfn,
                   int Nleaf, int datatype, int treetype,
                   int buildopts,
                   char** args, int argc);

#endif
