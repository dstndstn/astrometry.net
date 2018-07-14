/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdlib.h>

#include "xylist.h"
#include "fitsioutils.h"
#include "qfits_header.h"
#include "log.h"

#include "cutest.h"

static char* get_tmpfile(int i) {
    static char fn[256];
    sprintf(fn, "/tmp/test-xylist-%i", i);
    return strdup(fn);
}

void test_tagalong(CuTest* ct) {
    xylist_t *in, *out;
    char* fn = get_tmpfile(2);
    starxy_t fld;
    starxy_t infld;

    fits_use_error_system();
    log_init(LOG_VERB);

    out = xylist_open_for_writing(fn);
    CuAssertPtrNotNull(ct, out);
    xylist_set_include_flux(out, TRUE);
    CuAssertIntEquals(ct, 0, xylist_write_primary_header(out));

    int N = 100;
    double x[N];
    double y[N];
    double flux[N];
    double tagalong[2*N];
    int i;

    srand(0);

    for (i=0; i<N; i++) {
        float f;
        x[i] = i + 1;
        y[i] = 3 * i;
        flux[i] = 10 * i;
        //f = rand() / (double)RAND_MAX;
        f = 10000 + i;
        tagalong[2*i]   = f;
        //f = rand() / (double)RAND_MAX;
        f = 11000 + i;
        tagalong[2*i+1] = f;
    }
    fld.N = N;
    fld.x = x;
    fld.y = y;
    fld.flux = flux;

    int tagcol = xylist_add_tagalong_column(out, fitscolumn_double_type(),
                                            2, fitscolumn_float_type(),
                                            "TAGALONG", "tags");
    //CuAssertIntEquals(ct, 3, tagcol);
    CuAssertIntEquals(ct, 0, tagcol);

    CuAssertIntEquals(ct, 0, xylist_write_header(out));
    CuAssertIntEquals(ct, 0, xylist_write_field(out, &fld));

    int rtn = xylist_write_tagalong_column(out, tagcol, 0, N, tagalong, 2*sizeof(double));
    CuAssertIntEquals(ct, 0, rtn);

    CuAssertIntEquals(ct, 0, xylist_fix_header(out));
    CuAssertIntEquals(ct, 0, xylist_close(out));
    out = NULL;



    in = xylist_open(fn);
    CuAssertPtrNotNull(ct, in);
    CuAssertIntEquals(ct, 1, xylist_n_fields(in));
    CuAssertIntEquals(ct, 0, strcmp(in->antype, AN_FILETYPE_XYLS));
    CuAssertPtrNotNull(ct, xylist_read_field(in, &infld));
    CuAssertIntEquals(ct, N, infld.N);
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fld.x[i], infld.x[i]);
        CuAssertIntEquals(ct, fld.y[i], infld.y[i]);
        CuAssertIntEquals(ct, fld.flux[i], infld.flux[i]);
    }
    CuAssertPtrEquals(ct, NULL, infld.background);

    double* intag = xylist_read_tagalong_column(in, "TAGALONG", fitscolumn_double_type());
    CuAssertIntEquals(ct, 0, memcmp(tagalong, intag, 2 * N * sizeof(double)));
    free(intag);

    starxy_free_data(&infld);
    CuAssertIntEquals(ct, 0, xylist_close(in));
    in = NULL;

    free(fn);
}


void test_simple_xy(CuTest* ct) {
    xylist_t *in, *out;
    char* fn = get_tmpfile(1);
    starxy_t fld;
    starxy_t infld;
    fits_use_error_system();

    out = xylist_open_for_writing(fn);
    CuAssertPtrNotNull(ct, out);
    xylist_set_include_flux(out, TRUE);
    CuAssertIntEquals(ct, 0, xylist_write_primary_header(out));
    CuAssertIntEquals(ct, 0, xylist_write_header(out));

    int N = 100;
    double x[N];
    double y[N];
    double flux[N];
    int i;

    for (i=0; i<N; i++) {
        x[i] = i + 1;
        y[i] = 3 * i;
        flux[i] = 10 * i;
    }
    fld.N = N;
    fld.x = x;
    fld.y = y;
    fld.flux = flux;

    CuAssertIntEquals(ct, 0, xylist_write_field(out, &fld));
    CuAssertIntEquals(ct, 0, xylist_fix_header(out));
    CuAssertIntEquals(ct, 0, xylist_close(out));
    out = NULL;


    in = xylist_open(fn);
    CuAssertPtrNotNull(ct, in);

    CuAssertIntEquals(ct, 1, xylist_n_fields(in));

    CuAssertIntEquals(ct, 0, strcmp(in->antype, AN_FILETYPE_XYLS));
    CuAssertPtrNotNull(ct, xylist_read_field(in, &infld));
    CuAssertIntEquals(ct, N, infld.N);
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fld.x[i], infld.x[i]);
        CuAssertIntEquals(ct, fld.y[i], infld.y[i]);
        CuAssertIntEquals(ct, fld.flux[i], infld.flux[i]);
    }
    CuAssertPtrEquals(ct, NULL, infld.background);

    starxy_free_data(&infld);
    CuAssertIntEquals(ct, 0, xylist_close(in));
    in = NULL;

    CuAssertIntEquals(ct, TRUE, xylist_is_file_xylist(fn, 1, NULL, NULL, NULL));
    CuAssertIntEquals(ct, FALSE, xylist_is_file_xylist("no-such-file", 1, NULL, NULL, NULL));
    CuAssertIntEquals(ct, FALSE, xylist_is_file_xylist(fn, 1, "XXX", "YYY", NULL));

    free(fn);
}

void test_read_write_xy(CuTest* ct) {
    xylist_t *in, *out;
    char* fn = get_tmpfile(0);
    starxy_t fld;
    starxy_t infld;

    log_init(LOG_VERB);
    logverb("using temp file %s\n", fn);
    fits_use_error_system();

    out = xylist_open_for_writing(fn);
    CuAssertPtrNotNull(ct, out);

    qfits_header* hdr = xylist_get_primary_header(out);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "KEYA", 42, "Comment");

    CuAssertIntEquals(ct, 0, xylist_write_primary_header(out));

    xylist_set_xname(out, "XXX");
    xylist_set_yname(out, "YYY");
    xylist_set_ytype(out, TFITS_BIN_TYPE_E);
    xylist_set_xunits(out, "pix");
    xylist_set_yunits(out, "piy");

    xylist_set_include_flux(out, TRUE);

    hdr = xylist_get_header(out);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "KEYB", 43, "CommentB");

    CuAssertIntEquals(ct, 0, xylist_write_header(out));

    int N = 10;
    double x[N];
    double y[N];
    double flux[N];
    int i;

    for (i=0; i<N; i++) {
        x[i] = 10 * i;
        y[i] = 5 * i;
        flux[i] = 100 * i;
    }
    fld.N = 10;
    fld.x = x;
    fld.y = y;
    fld.flux = flux;

    CuAssertIntEquals(ct, 0, xylist_write_field(out, &fld));
    CuAssertIntEquals(ct, 0, xylist_fix_header(out));

    xylist_set_xname(out, "X2");
    xylist_set_yname(out, "Y2");
    xylist_set_xunits(out, "ux");
    xylist_set_yunits(out, "uy");

    xylist_set_include_flux(out, FALSE);

    xylist_next_field(out);

    int N2 = 5;
    fld.N = N2;

    CuAssertIntEquals(ct, 0, xylist_write_header(out));
    CuAssertIntEquals(ct, 0, xylist_write_field(out, &fld));
    CuAssertIntEquals(ct, 0, xylist_fix_header(out));
    CuAssertIntEquals(ct, 0, xylist_close(out));

    out = NULL;

    
    in = xylist_open(fn);
    CuAssertPtrNotNull(ct, in);

    CuAssertIntEquals(ct, 2, xylist_n_fields(in));

    hdr = xylist_get_primary_header(in);
    CuAssertPtrNotNull(ct, hdr);
    char* typ = fits_get_dupstring(hdr, "AN_FILE");
    CuAssertPtrNotNull(ct, typ);
    CuAssertIntEquals(ct, 0, strcmp(typ, "XYLS"));
    free(typ);

    CuAssertIntEquals(ct, 42, qfits_header_getint(hdr, "KEYA", -1));

    xylist_set_xname(in, "XXX");
    xylist_set_yname(in, "YYY");

    hdr = xylist_get_header(in);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 43, qfits_header_getint(hdr, "KEYB", -1));
    CuAssertPtrNotNull(ct, xylist_read_field(in, &infld));

    CuAssertIntEquals(ct, N, infld.N);
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fld.x[i], infld.x[i]);
        CuAssertIntEquals(ct, fld.y[i], infld.y[i]);
        CuAssertIntEquals(ct, fld.flux[i], infld.flux[i]);
    }
    CuAssertPtrEquals(ct, NULL, infld.background);

    starxy_free_data(&infld);

    int r = xylist_next_field(in);
    CuAssertIntEquals(ct, 0, r);

    // the columns are named differently...
    CuAssertPtrEquals(ct, NULL, xylist_read_field(in, &infld));

    xylist_set_xname(in, "X2");
    xylist_set_yname(in, "Y2");

    memset(&infld, 0, sizeof(infld));
    CuAssertPtrNotNull(ct, xylist_read_field(in, &infld));

    CuAssertIntEquals(ct, N2, infld.N);
    for (i=0; i<N2; i++) {
        CuAssertIntEquals(ct, fld.x[i], infld.x[i]);
        CuAssertIntEquals(ct, fld.y[i], infld.y[i]);
    }
    CuAssertPtrEquals(ct, NULL, infld.flux);
    CuAssertPtrEquals(ct, NULL, infld.background);

    starxy_free_data(&infld);

    r = xylist_next_field(in);
    // no such field...
    CuAssertIntEquals(ct, -1, r);
    CuAssertPtrEquals(ct, NULL, xylist_read_field(in, &infld));

    CuAssertIntEquals(ct, 0, xylist_close(in));

    // I love valgrind-clean tests
    free(fn);
}

