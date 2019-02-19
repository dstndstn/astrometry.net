/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef CODE_KD_H
#define CODE_KD_H

#include "astrometry/kdtree.h"
#include "astrometry/qfits_header.h"
#include "astrometry/anqfits.h"

#define AN_FILETYPE_CODETREE "CKDT"

#define CODETREE_NAME "codes"

typedef struct {
    kdtree_t* tree;
    qfits_header* header;
    int* inverse_perm;
} codetree_t;

codetree_t* codetree_open(const char* fn);

codetree_t* codetree_open_fits(anqfits_t* fits);

int codetree_get(codetree_t* s, unsigned int codeid, double* code);

int codetree_N(codetree_t* s);

int codetree_nodes(codetree_t* s);

int codetree_D(codetree_t* s);

int codetree_get_permuted(codetree_t* s, int index);

qfits_header* codetree_header(codetree_t* s);

int codetree_close(codetree_t* s);

// for writing
codetree_t* codetree_new(void);

int codetree_append_to(codetree_t* s, FILE* fid);

int codetree_write_to_file(codetree_t* s, const char* fn);

int codetree_write_to_file_flipped(codetree_t* s, const char* fn);

#endif
