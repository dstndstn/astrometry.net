/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef LS_FILE_H
#define LS_FILE_H

#include <stdio.h>
#include "astrometry/bl.h"

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
