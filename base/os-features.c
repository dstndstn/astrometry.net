/*
  This file is part of the Astrometry.net suite.
  Copyright 2008, 2012 Dustin Lang.

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
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include "os-features.h"

#if defined(_POSIX_SYNCHRONIZED_IO) && (_POSIX_SYNCHRONIZED_IO > 0)
#define NEED_FDATASYNC 0
#else
#define NEED_FDATASYNC 1
#endif

#if NEED_CANONICALIZE_FILE_NAME
char* canonicalize_file_name(const char* fn) {
    char* path = malloc(1024);
    char* canon;
    canon = realpath(fn, path);
    if (!canon) {
        free(path);
        return NULL;
    }
    path = realloc(path, strlen(path) + 1);
    return path;
}
#endif

#if NEED_FDATASYNC
int fdatasync(int fd) {
    return fsync(fd);
}
#endif

#if NEED_QSORT_R
#include "qsort_reentrant.c"
#endif
