/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef MERGE_INDEX_H
#define MERGE_INDEX_H

#include "astrometry/quadfile.h"
#include "astrometry/codekd.h"
#include "astrometry/starkd.h"

/**
 Merges .quad, .ckdt, and .skdt files to produce a .index file.
 */
int merge_index_files(const char* quadfn, const char* ckdtfn, const char* skdtfn,
                      const char* indexfn);

int merge_index_open_files(const char* quadfn, const char* ckdtfn, const char* skdtfn,
                           quadfile_t** quad, codetree_t** code, startree_t** star);

int merge_index(quadfile_t* quads, codetree_t* codekd, startree_t* starkd,
                const char* indexfn);

#endif
