/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef FILEUTILS_H
#define FILEUTILS_H

/*
 Removes '.' and '..' references from a path.
 Collapses '//' to '/'.
 Does NOT care whether the file actually exists.
 Does NOT resolve symlinks.
 Assumes '/' is the path separator.

 Returns a newly-allocated string which should be freed with free().
 */
char* an_canonicalize_file_name(const char* fn);

char* resolve_path(const char* filename, const char* basedir);

char* find_executable(const char* progname, const char* sibling);

#endif
