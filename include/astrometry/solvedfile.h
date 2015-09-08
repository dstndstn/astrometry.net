/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef SOLVEDFILE_H
#define SOLVEDFILE_H

#include "astrometry/bl.h"
#include "astrometry/an-bool.h"

/**
 All field numbers are 1-indexed.

 The solvedfiles themselves are 0-indexed, but this module handles
 that.
 */

int solvedfile_get(char* fn, int fieldnum);

int solvedfile_getsize(char* fn);

/**
   Get a list of unsolved fields between "firstfield" and
   "lastfield", up to a maximum of "maxfields" (no limit if
   "maxfields" is zero).
 */
il* solvedfile_getall(char* fn, int firstfield, int lastfield, int maxfields);

/**
   Same as "getall" except return solved fields.
 */
il* solvedfile_getall_solved(char* fn, int firstfield, int lastfield, int maxfields);

int solvedfile_set(char* fn, int fieldnum);

/*
 Set an array of fields.  Note that the "vals" array is 0-indexed;
 vals[0] corresponds to field 1.
 This *only sets* elements, it does *not* reset (clear) values in
 the file.
 */
int solvedfile_set_array(char* fn, anbool* vals, int N);

/**
 Sets the file to the given values and size (possibly truncating it!)
 */
int solvedfile_set_file(char* fn, anbool* vals, int N);

int solvedfile_setsize(char* fn, int fieldnum);

#endif
