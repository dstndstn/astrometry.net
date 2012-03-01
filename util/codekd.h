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

#ifndef CODE_KD_H
#define CODE_KD_H

#include "kdtree.h"
#include "qfits.h"
#include "anqfits.h"

#define AN_FILETYPE_CODETREE "CKDT"

#define CODETREE_NAME "codes"

struct codetree {
	kdtree_t* tree;
	qfits_header* header;
	int* inverse_perm;
};
typedef struct codetree codetree;


codetree* codetree_open(const char* fn);

codetree* codetree_open_fits(anqfits_t* fits);

int codetree_get(codetree* s, unsigned int codeid, double* code);

int codetree_N(codetree* s);

int codetree_nodes(codetree* s);

int codetree_D(codetree* s);

int codetree_get_permuted(codetree* s, int index);

qfits_header* codetree_header(codetree* s);

int codetree_close(codetree* s);

// for writing
codetree* codetree_new(void);

int codetree_append_to(codetree* s, FILE* fid);

int codetree_write_to_file(codetree* s, const char* fn);

int codetree_write_to_file_flipped(codetree* s, const char* fn);

#endif
