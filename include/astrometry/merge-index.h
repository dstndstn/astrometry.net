/*
  This file is part of the Astrometry.net suite.
  Copyright 2008, 2009 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
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
