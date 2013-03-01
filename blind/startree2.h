/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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

#ifndef STARTREE2_H
#define STARTREE2_H

#include "starkd.h"
#include "fitstable.h"
#include "an-bool.h"

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
								  const char* racol, const char* deccol);

#endif
