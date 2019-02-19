/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include "rdlist.h"
#include "fitsioutils.h"
#include "qfits_header.h"

#include "cutest.h"

static char* get_tmpfile(int i) {
    static char fn[256];
    sprintf(fn, "/tmp/test-rdlist-%i", i);
    return strdup(fn);
}

void test_simple(CuTest* ct) {
    rdlist_t *in, *out;
    char* fn = get_tmpfile(1);
    rd_t fld, infld;
    rd_t* infld2;

    out = rdlist_open_for_writing(fn);
    CuAssertPtrNotNull(ct, out);
    CuAssertIntEquals(ct, 0, rdlist_write_primary_header(out));
    CuAssertIntEquals(ct, 0, rdlist_write_header(out));

    int N = 100;
    double ra[N];
    double dec[N];
    int i;
    for (i=0; i<N; i++) {
        ra[i] = i + 1;
        dec[i] = 3 * i;
    }
    fld.N = N;
    fld.ra = ra;
    fld.dec = dec;

    CuAssertIntEquals(ct, 0, rdlist_write_field(out, &fld));
    CuAssertIntEquals(ct, 0, rdlist_fix_header(out));
    CuAssertIntEquals(ct, 0, rdlist_close(out));
    out = NULL;


    in = rdlist_open(fn);
    CuAssertPtrNotNull(ct, in);
    CuAssertIntEquals(ct, 0, strcmp(in->antype, AN_FILETYPE_RDLS));
    CuAssertPtrNotNull(ct, rdlist_read_field(in, &infld));
    CuAssertIntEquals(ct, N, infld.N);
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fld.ra[i], infld.ra[i]);
        CuAssertIntEquals(ct, fld.dec[i], infld.dec[i]);
    }
    rd_free_data(&infld);

    infld2 = rdlist_read_field(in, NULL);
    CuAssertPtrNotNull(ct, infld2);
    CuAssertIntEquals(ct, N, infld2->N);
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fld.ra[i], infld2->ra[i]);
        CuAssertIntEquals(ct, fld.dec[i], infld2->dec[i]);
    }
    rd_free(infld2);

    CuAssertIntEquals(ct, 0, rdlist_close(in));
    in = NULL;

    free(fn);
}

void test_read_write(CuTest* ct) {
    rdlist_t *in, *out;
    char* fn = get_tmpfile(0);
    rd_t fld;
    rd_t infld;

    out = rdlist_open_for_writing(fn);
    CuAssertPtrNotNull(ct, out);

    qfits_header* hdr = rdlist_get_primary_header(out);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "KEYA", 42, "Comment");

    CuAssertIntEquals(ct, 0, rdlist_write_primary_header(out));

    rdlist_set_raname(out, "RRR");
    rdlist_set_decname(out, "DDD");
    rdlist_set_ratype(out, TFITS_BIN_TYPE_E);
    rdlist_set_raunits(out, "deg");
    rdlist_set_decunits(out, "deg");

    hdr = rdlist_get_header(out);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "KEYB", 43, "CommentB");

    CuAssertIntEquals(ct, 0, rdlist_write_header(out));

    int N = 10;
    double ra[N];
    double dec[N];
    int i;

    for (i=0; i<N; i++) {
        ra[i] = 10 * i;
        dec[i] = 5 * i;
    }
    fld.N = 10;
    fld.ra = ra;
    fld.dec = dec;

    CuAssertIntEquals(ct, 0, rdlist_write_field(out, &fld));
    CuAssertIntEquals(ct, 0, rdlist_fix_header(out));

    rdlist_set_raname(out, "RA2");
    rdlist_set_decname(out, "DEC2");
    rdlist_set_raunits(out, "ur");
    rdlist_set_decunits(out, "ud");

    rdlist_next_field(out);

    int N2 = 5;
    fld.N = N2;

    CuAssertIntEquals(ct, 0, rdlist_write_header(out));
    CuAssertIntEquals(ct, 0, rdlist_write_field(out, &fld));
    CuAssertIntEquals(ct, 0, rdlist_fix_header(out));

    CuAssertIntEquals(ct, 0, rdlist_close(out));

    out = NULL;


    
    in = rdlist_open(fn);
    CuAssertPtrNotNull(ct, in);

    hdr = rdlist_get_primary_header(in);
    CuAssertPtrNotNull(ct, hdr);
    char* typ = fits_get_dupstring(hdr, "AN_FILE");
    CuAssertPtrNotNull(ct, typ);
    CuAssertIntEquals(ct, 0, strcmp(typ, "RDLS"));
    free(typ);

    CuAssertIntEquals(ct, 42, qfits_header_getint(hdr, "KEYA", -1));

    rdlist_set_raname(in, "RRR");
    rdlist_set_decname(in, "DDD");

    hdr = rdlist_get_header(in);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 43, qfits_header_getint(hdr, "KEYB", -1));

    CuAssertPtrNotNull(ct, rdlist_read_field(in, &infld));

    CuAssertIntEquals(ct, N, infld.N);
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fld.ra[i], infld.ra[i]);
        CuAssertIntEquals(ct, fld.dec[i], infld.dec[i]);
    }

    rd_free_data(&infld);

    rdlist_next_field(in);

    // the columns are named differently...
    CuAssertPtrEquals(ct, NULL, rdlist_read_field(in, &infld));

    rdlist_set_raname(in, "RA2");
    rdlist_set_decname(in, "DEC2");

    memset(&infld, 0, sizeof(infld));
    CuAssertPtrNotNull(ct, rdlist_read_field(in, &infld));

    CuAssertIntEquals(ct, N2, infld.N);
    for (i=0; i<N2; i++) {
        CuAssertIntEquals(ct, fld.ra[i], infld.ra[i]);
        CuAssertIntEquals(ct, fld.dec[i], infld.dec[i]);
    }
    rd_free_data(&infld);

    rdlist_next_field(in);
    // no such field...
    CuAssertPtrEquals(ct, NULL, rdlist_read_field(in, &infld));

    CuAssertIntEquals(ct, 0, rdlist_close(in));
    free(fn);
}
