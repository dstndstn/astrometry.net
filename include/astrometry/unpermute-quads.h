/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef UNPERMUTE_QUADS_H
#define UNPERMUTE_QUADS_H

#include "astrometry/quadfile.h"
#include "astrometry/codekd.h"

/**
 \file Applies a code kdtree permutation array to the corresponding
 .quad file to produce new .quad and .ckdt files that are
 consistent and don't require permutation.

 In:  .quad, .ckdt
 Out: .quad, .ckdt

 Original author: dstn
 */
int unpermute_quads_files(const char* quadinfn, const char* ckdtinfn,
                          const char* quadoutfn, const char* ckdtoutfn,
                          char** args, int argc);

int unpermute_quads(quadfile_t* quadin, codetree_t* ckdtin,
                    quadfile_t* quadout, codetree_t** ckdtout,
                    char** args, int argc);

#endif
