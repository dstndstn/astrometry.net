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

#ifndef CODETREE_H
#define CODETREE_H

#include "codekd.h"
#include "codefile.h"
#include "fitstable.h"

/**
 */
codetree* codetree_build(codefile* codes,
						 int Nleaf, int datatype, int treetype,
						 int buildopts,
						 char** args, int argc);

int codetree_files(const char* codefn, const char* ckdtfn,
				   int Nleaf, int datatype, int treetype,
				   int buildopts,
				   char** args, int argc);

#endif
