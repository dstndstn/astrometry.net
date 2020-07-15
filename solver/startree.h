/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef STARTREE2_H
#define STARTREE2_H

#include "astrometry/starkd.h"
#include "astrometry/fitstable.h"
#include "astrometry/an-bool.h"

/**
 Given a FITS BINTABLE, pulls out RA,Dec columns and builds a kd-tree
 out of them.
 */
startree_t* startree_build(fitstable_t* intable,
						   const char* racol, const char* deccol,
						   // KDT_DATA_*, KDT_TREE_*
						   int datatype, int treetype,
						   // KD_BUILD_*
						   int buildopts,
						   int Nleaf,
						   char** args, int argc);

anbool startree_has_tagalong_data(const fitstable_t* intab);

int startree_write_tagalong_table(fitstable_t* intable, fitstable_t* outtable,
                                  const char* racol, const char* deccol,
                                  int* indices,
                                  anbool remove_radec_columns);

#endif
