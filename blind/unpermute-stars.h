/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
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

#ifndef UNPERMUTE_STARS_H
#define UNPERMUTE_STARS_H

#include "starkd.h"
#include "quadfile.h"
#include "fitstable.h"

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

int unpermute_stars(startree_t* starkdin, quadfile* quadin,
					startree_t** starkdout, quadfile* quadout,
					anbool sweep, anbool check,
					char** args, int argc);

int unpermute_stars_tagalong(startree_t* starkdin,
							 fitstable_t* tagalong_out);

#endif
