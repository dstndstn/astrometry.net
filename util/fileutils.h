/*
  This file is part of the Astrometry.net suite.
  Copyright 2014 Dustin Lang.

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
