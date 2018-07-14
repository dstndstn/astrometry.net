/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef UNPERMUTE_STARS_H
#define UNPERMUTE_STARS_H

#include "astrometry/starkd.h"
#include "astrometry/quadfile.h"
#include "astrometry/fitstable.h"

/**
 \file Applies a star kdtree permutation array to all files that depend on
 the ordering of the stars:   .quad and .skdt .
 The new files are consistent and don't require the star kdtree to have a
 permutation array.

 In:  .quad, .skdt
 Out: .quad, .skdt

 Original author: dstn
 */
int unpermute_stars_files(const char* skdtinfn, const char* quadinfn,
                          const char* skdtoutfn, const char* quadoutfn,
                          anbool sweep, anbool check,
                          char** args, int argc);

int unpermute_stars(startree_t* starkdin, quadfile_t* quadin,
                    startree_t** starkdout, quadfile_t* quadout,
                    anbool sweep, anbool check,
                    char** args, int argc);

int unpermute_stars_tagalong(startree_t* starkdin,
                             fitstable_t* tagalong_out);

#endif
