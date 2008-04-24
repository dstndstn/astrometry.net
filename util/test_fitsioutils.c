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

#include "fitsioutils.h"
#include "qfits_header.h"

void test_from_paper_1(CuTest* tc) {
    //FILE* tmpf = tmpfile();
    char buf[4096];
    int i;
    qfits_header* hdr;
    char* str;

    fits_use_error_system();

    memset(buf, ' ', sizeof(buf));
    sprintf(buf,
            "SIMPLE  = T / this is FITS                                                      "
            "SVALUE  = 'This is a long string value &'                                       "
            "CONTINUE  'extending&   '                                                       "
            "CONTINUE  ' over 3 lines.'                                                      "
            "END                                                                             "
            );
    // erase the null-termination.
    buf[strlen(buf)-1] = ' ';

    hdr = qfits_header_read_hdr_string(buf, sizeof(buf));

    //qfits_header_debug_dump(hdr);

    CuAssertPtrNotNull(tc, hdr);
    /*
     printf("Got header:\n");
     qfits_header_dump(hdr, stdout);
     */

    str = fits_get_long_string(hdr, "SVALUE");
    CuAssertStrEquals(tc, "This is a long string value extending over 3 lines.",
                      str);
    free(str);

    qfits_header_destroy(hdr);
}

void test_write_long_string(CuTest* tc) {
    qfits_header* hdr;
    FILE* tmpf;
    char* str;

    tmpf = tmpfile();

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
    qfits_header_dump(hdr, tmpf);

    //qfits_header_debug_dump(hdr);

    /*
     rewind(tmpf);
     for (;;) {
     char line[81];
     if (!fgets(line, sizeof(line), tmpf))
     break;
     printf("Line: >>%s<<\n", line);
     }
     */

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


