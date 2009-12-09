/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
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

#ifndef UNPERMUTE_QUADS_H
#define UNPERMUTE_QUADS_H

#include "quadfile.h"
#include "codekd.h"

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

int unpermute_quads(quadfile* quadin, codetree* ckdtin,
					quadfile* quadout, codetree** ckdtout,
					char** args, int argc);

#endif
