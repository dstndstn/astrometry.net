/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/
#ifndef RESORT_XYLIST_H
#define RESORT_XYLIST_H

/*
 Sorts the xylist in *infn*, writing the output to *outfn*.

 The sorting order is strange.  The algorithm first performs two
 indirect sorts, by flux and by flux+background.  It then interleaves
 these two results, without duplication.  This can be useful if you
 don't really trust the background subtraction...
 */
int resort_xylist(const char* infn, const char* outfn,
                  const char* fluxcol, const char* backcol,
                  int ascending);

#endif

