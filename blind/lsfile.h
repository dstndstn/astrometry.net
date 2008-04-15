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

#ifndef LS_FILE_H
#define LS_FILE_H

#include <stdio.h>
#include "bl.h"

/**
   __ls files look like this:

   NumFields=20
   # comment
   3,3.7,5.6,4.3,1,3,7
   2,7,8,9,0

   ie, for each field, the number of objects is given,
   followed by the positions of the objects.

   Returns a pl* containing a dl* for
   each field.  Each dl is a list of
   doubles of the positions of the objects.

   The "dimension" argument says how many elements to read
   per field, ie, what the dimensionality of the position is.
*/
pl* read_ls_file(FILE* fid, int dimension);

/**
   Frees the list returned by "read_ls_file".
*/
void ls_file_free(pl* l);

/**
   Returns the number of fields in this file, or -1 if
   reading fails.
*/
int read_ls_file_header(FILE* fid);

/**
   Reads one field from the file.

   The returned dl* contains (dimension * numpoints)
   doubles.
*/
dl* read_ls_file_field(FILE* fid, int dimension);

#endif
