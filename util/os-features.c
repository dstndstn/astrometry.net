/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include "os-features.h"

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

#if NEED_QSORT_R
#include "qsort_reentrant.c"
#endif
