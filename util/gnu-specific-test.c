/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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
#include <stdio.h>

#ifdef TEST_CANONICALIZE_FILE_NAME
int main() {
    char* path = canonicalize_file_name("/");
    free(path);
    printf("#define NEED_CANONICALIZE_FILE_NAME 0\n");
    return 0;
}
#endif

#ifdef TEST_QSORT_R
static int cmp(void* u, const void* a, const void* b) {
    return 0;
}
int main() {
    int array;
    qsort_r(&array, 1, sizeof(int), NULL, cmp);
    printf("#define NEED_QSORT_R 0\n");
    return 0;
}
#endif

#ifdef TEST_DECLARE_QSORT_R
// Test whether just declaring qsort_r as we do causes a compile failure.
// This is seen on Ubuntu 8.10, where apparently they declared it in a
// conflicting way in stdlib.h .  Way to go, guys.

//// NOTE: this declaration must match gnu-specific-test.c .
void qsort_r(void *base, size_t nmemb, size_t sz,
             void *userdata,
             int (*compar)(void *, const void *, const void *));

int main() {
    printf("#define NEED_DECLARE_QSORT_R 1\n");
    return 0;
}
#endif
