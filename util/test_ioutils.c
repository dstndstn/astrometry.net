/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"
#include "ioutils.h"
#include "fileutils.h"
#include "log.h"
#include "tic.h"
#include "md5.h"

void test_run_command_1(CuTest* tc) {
    int rtn;
    sl* outlines = NULL;
    sl* errlines = NULL;
    char* cmd;
    char* str2;
    char txt[10240];
    log_init(3);

    /*
     str1 = "test test\ntest";
     asprintf_safe(&cmd, "echo '%s'", str1);
     rtn = run_command_get_outputs(cmd, &outlines, &errlines);
     CuAssertIntEquals(tc, 0, rtn);
     str2 = sl_join(outlines, "\n");
     printf("got string: \"%s\"\n", str2);
     CuAssertIntEquals(tc, TRUE, streq(str1, str2));
     sl_free2(outlines);
     sl_free2(errlines);
     free(str2);

     printf("\n\npart 2\n\n");
     cmd = "(for ((i=0; i<1024; i++)); do /bin/echo -n \"X\"; done) | cat -";
     rtn = run_command_get_outputs(cmd, &outlines, &errlines);
     CuAssertIntEquals(tc, 0, rtn);
     str2 = sl_join(outlines, "\n");
     printf("got string: \"%s\"\n", str2);
     sl_free2(outlines);
     sl_free2(errlines);
     free(str2);
     */
    // test 'buffer full with no newline' branch
    // (single read)
    memset(txt, 'X', 1024);
    txt[1024] = '\0';
    asprintf_safe(&cmd, "printf %%s '%s'", txt);
    rtn = run_command_get_outputs(cmd, &outlines, &errlines);
    CuAssertIntEquals(tc, 0, rtn);
    str2 = sl_join(outlines, "\n");
    //printf("got string: \"%s\"\n", str2);
    CuAssertIntEquals(tc, 1, sl_size(outlines));
    CuAssertIntEquals(tc, 1024, strlen(str2));
    sl_free2(outlines);
    sl_free2(errlines);
    free(str2);

    // test 'buffer full with no newline' branch
    // (two reads)
    memset(txt, 'X', 512);
    txt[512] = '\0';
    asprintf_safe(&cmd, "printf %%s '%s'; printf %%s '%s'", txt, txt);
    rtn = run_command_get_outputs(cmd, &outlines, &errlines);
    CuAssertIntEquals(tc, 0, rtn);
    str2 = sl_join(outlines, "\n");
    //printf("got string: \"%s\"\n", str2);
    CuAssertIntEquals(tc, 1, sl_size(outlines));
    CuAssertIntEquals(tc, 1024, strlen(str2));
    sl_free2(outlines);
    sl_free2(errlines);
    free(str2);

    // test 'buffer full with no newline' branch
    // and 'flushing the last line'
    // (three reads, not exactly aligned)
    memset(txt, 'X', 500);
    txt[500] = '\0';
    // oops, this is not portable /bin/sh!
    //asprintf_safe(&cmd, "for ((i=0; i<3; i++)); do printf %%s '%s'; done", txt);
    asprintf_safe(&cmd, "for x in 1 2 3; do printf %%s '%s'; done", txt);
    printf("Command: \"%s\"\n", cmd);
    rtn = run_command_get_outputs(cmd, &outlines, &errlines);
    //rtn = run_command_get_outputs(cmd, NULL, NULL);
    //printf("return value: %i\n", rtn);
    CuAssertIntEquals(tc, 0, rtn);
    str2 = sl_join(outlines, "");
    //printf("got string: \"%s\"\n", str2);
    CuAssertIntEquals(tc, 2, sl_size(outlines));
    CuAssertIntEquals(tc, 1500, strlen(str2));
    sl_free2(outlines);
    sl_free2(errlines);
    free(str2);

    memset(txt, 'X', 500);
    txt[499] = '\n';
    txt[500] = '\0';
    //asprintf_safe(&cmd, "for ((i=0; i<3; i++)); do printf %%s '%s'; done", txt);
    asprintf_safe(&cmd, "for x in 1 2 3; do printf %%s '%s'; done", txt);
    rtn = run_command_get_outputs(cmd, &outlines, &errlines);
    CuAssertIntEquals(tc, 0, rtn);
    str2 = sl_join(outlines, "\n");
    //printf("got string: \"%s\"\n", str2);
    CuAssertIntEquals(tc, 3, sl_size(outlines));
    CuAssertIntEquals(tc, 1499, strlen(str2));
    sl_free2(outlines);
    sl_free2(errlines);
    free(str2);

    // single line ending with newline.
    memset(txt, 'X', 1024);
    txt[1023] = '\n';
    txt[1024] = '\0';
    asprintf_safe(&cmd, "printf %%s '%s'", txt);
    rtn = run_command_get_outputs(cmd, &outlines, &errlines);
    CuAssertIntEquals(tc, 0, rtn);
    str2 = sl_join(outlines, "");
    //printf("got string: \"%s\"\n", str2);
    CuAssertIntEquals(tc, 1, sl_size(outlines));
    CuAssertIntEquals(tc, 1023, strlen(str2));
    sl_free2(outlines);
    sl_free2(errlines);
    free(str2);

    // multiple lines -- normal
    // written all at once -- hits 'moved N to start of block' branch.
    memset(txt, 'X', 2000);
    txt[200] = '\n';
    txt[400] = '\n';
    txt[600] = '\n';
    txt[1000] = '\n';
    txt[1500] = '\n';
    txt[1800] = '\n';
    txt[1999] = '\n';
    txt[2000] = '\0';
    asprintf_safe(&cmd, "printf %%s '%s'", txt);
    rtn = run_command_get_outputs(cmd, &outlines, &errlines);
    CuAssertIntEquals(tc, 0, rtn);
    str2 = sl_join(outlines, "");
    //printf("got string: \"%s\"\n", str2);
    CuAssertIntEquals(tc, 7, sl_size(outlines));
    CuAssertIntEquals(tc, 1993, strlen(str2));
    sl_free2(outlines);
    sl_free2(errlines);
    free(str2);

    // multiple lines -- normal
    // written all at once -- hits 'moved N to start of block' branch.
    memset(txt, 'X', 200);
    txt[200] = '\n';
    txt[201] = '\0';
    //asprintf_safe(&cmd, "for ((i=0; i<10; i++)); do printf %%s '%s'; sleep 0.1; done", txt);
    asprintf_safe(&cmd, "for x in 1 2 3 4 5 6 7 8 9 10; do printf %%s '%s'; sleep 0.1; done", txt);
    rtn = run_command_get_outputs(cmd, &outlines, &errlines);
    CuAssertIntEquals(tc, 0, rtn);
    str2 = sl_join(outlines, "");
    //printf("got string: \"%s\"\n", str2);
    CuAssertIntEquals(tc, 10, sl_size(outlines));
    CuAssertIntEquals(tc, 10*200, strlen(str2));
    sl_free2(outlines);
    sl_free2(errlines);
    free(str2);
}


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
    log_init(3);

    tmpfn = create_temp_file("test_run_command", NULL);
    CuAssertPtrNotNull(tc, tmpfn);

    for (trial=0; trial<4; trial++) {
        double t0;
        printf("test_ioutils:test_run_command() trial %i\n", trial);
        fid = fopen(tmpfn, "wb");
        CuAssertPtrNotNull(tc, fid);
        N = 102400;
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
            nw = fwrite(&c, 1, 1, fid);
            CuAssertIntEquals(tc, 1, nw);
        }
        rtn = fclose(fid);
        CuAssertIntEquals(tc, 0, rtn);

        if (trial == 3) {
            asprintf(&cmd, "sleep 0.2; dd if=%s bs=1k; sleep 0.2", tmpfn);
        } else {
            asprintf(&cmd, "sleep 0.2; dd if=%s bs=1k 1>&2; sleep 0.2", tmpfn);
        }
        t0 = timenow();
        rtn = run_command_get_outputs(cmd, &outlines, &errlines);
        printf("That took %g sec\n", timenow() - t0);
        CuAssertIntEquals(tc, 0, rtn);
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

