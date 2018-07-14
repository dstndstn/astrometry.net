/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

#include "qfits_header.h"
#include "qfits_rw.h"

#include "fitsioutils.h"
#include "qfits_header.h"

static void expect(CuTest* tc, const char* header, const char* key, const char* val) {
    char buf[4096];
    qfits_header* hdr;
    char* str;
    fits_use_error_system();
    memset(buf, ' ', sizeof(buf));
    sprintf(buf,
            "SIMPLE  = T / this is FITS                                                      "
            "%s"
            "END                                                                             ",
            header);
    // erase the null-termination.
    buf[strlen(buf)-1] = ' ';
    hdr = qfits_header_read_hdr_string((unsigned char*)buf, sizeof(buf));
    CuAssertPtrNotNull(tc, hdr);
    str = fits_get_long_string(hdr, key);
    CuAssertStrEquals(tc, val, str);
    free(str);
    qfits_header_destroy(hdr);
}

void test_from_paper_1(CuTest* tc) {
    expect(tc,
           "SVALUE  = 'This is a long string value &'                                       "
           "CONTINUE  'extending&   '                                                       "
           "CONTINUE  ' over 3 lines.'                                                      ",
           "SVALUE",
           "This is a long string value extending over 3 lines.");
}

void test_from_paper_2(CuTest* tc) {
    expect(tc,
           "SVALUE  = 'This is a long string value &'                                       "
           "MAXVOLT =   12.5                                                                "
           "CONTINUE  ' over 3 lines.'                                                      ",
           "SVALUE",
           "This is a long string value &");
}

void test_from_paper_page1_3(CuTest* tc) {
    // non-significant space characters may occur between the amp and closing quote.
    expect(tc,
           "SVALUE  = 'This is a long string value &   '                                    "
           "CONTINUE  'spread&    '                                                         "
           "CONTINUE  ' over 3 lines.'                                                      ",
           "SVALUE",
           "This is a long string value spread over 3 lines.");
}

void test_from_paper_page1_5(CuTest* tc) {
    // the substring may be anywhere in columns 11-80 and may be preceded by non-significant spaces.
    expect(tc,
           "SVALUE  = 'This is a long string value &   '                                    "
           "CONTINUE                                   'spread&    '                        "
           "CONTINUE                                                        ' over 3 lines.'",
           "SVALUE",
           "This is a long string value spread over 3 lines.");
}

void test_from_paper_page1_5b(CuTest* tc) {
    // A comment string may follow the substring; if present it must be separated from the substring
    // by at least one space.
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE                                   'spread&    ' /comment 2             "
           "CONTINUE                                                  ' over 3 lines.'    / ",
           "SVALUE",
           "This is a long string value spread over 3 lines.");
}

void test_from_paper_page2_1(CuTest* tc) {
    // if the last non-space character in the initial value is &, then it may be a continued line, if:
    // --the next keyword is CONTINUE
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "VALUE2  = 42                                                                    ",
           "SVALUE",
           "This is a long string value &");
    // -- bytes 9 and 10 contains spaces (no = in byte 9)
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE= 'maybe it does...'                                                    ",
           "SVALUE",
           "This is a long string value &");
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE 'maybe it does...'                                                     ",
           "SVALUE",
           "This is a long string value &");
    // -- bytes 11 through 80 contain a char string enclosed in single quotes,
    //    optionally preceded and followed by spaces, optionally followed by a
    //    comment.
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE  maybe it is not in quotes...                                          ",
           "SVALUE",
           "This is a long string value &");
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE  'maybe it has a start quote but no finishing one                      ",
           "SVALUE",
           "This is a long string value &");
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE  'maybe it has quote '' chars inside but no final quote                ",
           "SVALUE",
           "This is a long string value &");
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE       'with leading and trailing space     '                           ",
           "SVALUE",
           "This is a long string value with leading and trailing space");
    expect(tc,
           "SVALUE  = 'This is a long string value &   '               / comment 1          "
           "CONTINUE       'with leading and trailing space     '  /comment with quotes ''' ",
           "SVALUE",
           "This is a long string value with leading and trailing space");
}

void test_from_paper_page2_notes(CuTest* tc) {
    // CONTINUE not following a regular keyword is ok.
    expect(tc,
           "SVALUE  = 'This is a long string value    '               / comment 1           "
           "CONTINUE  'this should be ignored' // comment                                   ",
           "SVALUE",
           "This is a long string value");
}

void test_write_long_string(CuTest* tc) {
    qfits_header* hdr;
    char* str;

    hdr = qfits_header_default();
    fits_header_add_int(hdr, "VAL1", 42, "First value");
    fits_header_addf_longstring(hdr, "VAL2", "Second value",
                                "This is a very very very long string %s %i %s",
                                "with lots of special characters like the "
                                "following four single-quotes >>>>''''<<<< "
                                "and lots of ampersands &&&&&&&&&&&&&&&&&& "
                                "and the number", 42, "which of course has "
                                "special significance.  This sentence ends with"
                                " an ampersand.&");

    str = fits_get_long_string(hdr, "VAL2");
    CuAssertStrEquals(tc, "This is a very very very long string with lots of "
                      "special characters like the following four single-quotes "
                      ">>>>''''<<<< and lots of ampersands &&&&&&&&&&&&&&&&&& "
                      "and the number 42 which of course has special "
                      "significance.  This sentence ends with an ampersand.&",
                      str);
    free(str);

    qfits_header_destroy(hdr);
}


