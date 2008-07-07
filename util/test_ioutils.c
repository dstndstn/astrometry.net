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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

#include "ioutils.h"

void test_streq_1(CuTest* tc) {
    CuAssertIntEquals(tc, 1, streq(NULL, NULL));
    CuAssertIntEquals(tc, 0, streq(NULL, ""));
    CuAssertIntEquals(tc, 0, streq("", NULL));
    CuAssertIntEquals(tc, 1, streq("", ""));
    CuAssertIntEquals(tc, 0, streq("", "a"));
    CuAssertIntEquals(tc, 1, streq("a", "a"));
    CuAssertIntEquals(tc, 1, streq("yes", "yes"));
}


static void assertCanon(CuTest* tc, char* in, char* out) {
    char* canon = an_canonicalize_file_name(in);
    CuAssertPtrNotNull(tc, canon);
    if (strcmp(canon, out))
        printf("Input \"%s\", expected \"%s\", got \"%s\"\n", in, out, canon);
    CuAssertIntEquals(tc, 0, strcmp(canon, out));
    free(canon);
}

void test_canon_1(CuTest* tc) {
    assertCanon(tc, "//path/to/a/.//./file/with/../junk", "/path/to/a/file/junk");
}

void test_canon_2(CuTest* tc) {
    assertCanon(tc, "/", "/");
}

void test_canon_2b(CuTest* tc) {
    assertCanon(tc, ".", ".");
}

void test_canon_2c(CuTest* tc) {
    assertCanon(tc, "..", "..");
}

void test_canon_3(CuTest* tc) {
    assertCanon(tc, "x/../y", "y");
}

void test_canon_3b(CuTest* tc) {
    assertCanon(tc, "x/../y/../z/a/b/c/d/../../../e", "z/a/e");
}

void test_canon_4(CuTest* tc) {
    // HACK... this probably ISN'T what it should do.
    //assertCanon(tc, "../y", "y");

    assertCanon(tc, "../y", "../y");
}

void test_canon_4b(CuTest* tc) {
    assertCanon(tc, "../../../y", "../../../y");
}

void test_canon_4c(CuTest* tc) {
    assertCanon(tc, "../../../../y", "../../../../y");
}

void test_canon_5(CuTest* tc) {
    assertCanon(tc, "/../..//x", "/x");
}

