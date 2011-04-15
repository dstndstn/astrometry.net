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
#include "log.h"
#include "tic.h"
#include "md5.h"

void test_run_command(CuTest* tc) {
	int rtn;
	sl* outlines = NULL;
	sl* errlines = NULL;
	char* cmd;
	FILE* fid;
	char* tmpfn;
	int N;
	int i;
	int trial;
	char* str1;
	char* str2;
	/*
	 md5_context md5c;
	 char md5_A[64];
	 char md5_B[64];
	 */
	log_init(3);

	str1 = "test test\ntest";
	asprintf_safe(&cmd, "echo '%s'", str1);
	rtn = run_command_get_outputs(cmd, &outlines, &errlines);
	CuAssertIntEquals(tc, 0, rtn);
	str2 = sl_join(outlines, "\n");
	printf("got string: \"%s\"\n", str2);
	CuAssertIntEquals(tc, TRUE, streq(str1, str2));
	sl_free2(outlines);
	sl_free2(errlines);

	tmpfn = create_temp_file("test_run_command", "/tmp");
	CuAssertPtrNotNull(tc, tmpfn);

	for (trial=0; trial<4; trial++) {
		double t0;
		printf("test_ioutils:test_run_command() trial %i\n", trial);
		fid = fopen(tmpfn, "wb");
		CuAssertPtrNotNull(tc, fid);
		N = 102400;
		//md5_starts(&md5c);
		for (i=0; i<N; i++) {
			int nw;
			char c;
			if (trial == 0) {
				c = random() % 256;
			} else if (trial == 1) {
				c = '\n';
			} else if ((trial == 2) || (trial == 3)) {
				c = 'A';
			}
			//md5_update(&md5c, &c, 1);
			nw = fwrite(&c, 1, 1, fid);
			CuAssertIntEquals(tc, 1, nw);
		}
		rtn = fclose(fid);
		CuAssertIntEquals(tc, 0, rtn);
		//md5_file_hex(tmpfn, md5_A);

		if (trial == 3) {
			asprintf(&cmd, "sleep 1; dd if=%s bs=1k; sleep 1", tmpfn);
		} else {
			asprintf(&cmd, "sleep 1; dd if=%s bs=1k 1>&2; sleep 1", tmpfn);
		}
		t0 = timenow();
		rtn = run_command_get_outputs(cmd, &outlines, &errlines);
		printf("That took %g sec\n", timenow() - t0);
		CuAssertIntEquals(tc, 0, rtn);

		/*
		 char* sout = sl_join(outlines, "\n");
		 char* serr = sl_join(errlines, "\n");
		 */

		free(cmd);
		sl_free2(outlines);
		sl_free2(errlines);
	}
}

void test_split_long_string(CuTest* tc) {
	sl* lst;
	lst = split_long_string("", 60, 80, NULL);
	CuAssertPtrNotNull(tc, lst);
	CuAssertIntEquals(tc, 0, sl_size(lst));
	sl_free2(lst);

	lst = split_long_string("really long line     that will get broken into"
							" several pieces", 6, 10, NULL);
	CuAssertPtrNotNull(tc, lst);
	//printf("%s\n", sl_join(lst, "<<\n"));
	CuAssertIntEquals(tc, 7, sl_size(lst));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 0), "really"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 1), "long line"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 2), "that will"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 3), "get broken"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 4), "into"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 5), "several"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 6), "pieces"));
	sl_free2(lst);
	
	// Arguable whether this is correct handling of multiple spaces...
	lst = split_long_string("extremely long line     with ridiculously long words necessitating hyphenationizing (?!)",
							6, 10, NULL);
	CuAssertPtrNotNull(tc, lst);
	//printf("%s\n", sl_join(lst, "<<\n"));
	CuAssertIntEquals(tc, 12, sl_size(lst));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 0), "extre-"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 1), "mely long"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 2), "line"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 3), "with"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 4), "ridiculou-"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 5), "sly long"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 6), "words"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 7), "necessita-"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 8), "ting"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 9), "hyphenati-"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 10), "onizing"));
	CuAssertIntEquals(tc, 1, streq(sl_get(lst, 11), "(?!)"));
	sl_free2(lst);
}

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

