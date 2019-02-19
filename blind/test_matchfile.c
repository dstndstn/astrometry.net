/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cutest.h"
#include "matchfile.h"
#include "matchobj.h"
#include "an-bool.h"
#include "starutil.h"

#define SAME(X) CuAssertIntEquals(tc, entry1.X, ein1->X)
#define SAMEF(X) CuAssertDblEquals(tc, entry1.X, ein1->X, 1e-10)
#define SAMESAME(X) CuAssertIntEquals(tc, 0, memcmp(ein1->X, entry1.X, sizeof(entry1.X)))

void test_read_matchfile(CuTest* tc) {
    char* fn = "/tmp/test-matchfile-0";

    MatchObj entry1;
    matchfile* out;
    matchfile* in;
    MatchObj* ein1;

    memset(&entry1, 0, sizeof(MatchObj));

    entry1.quadno = 123456;
    entry1.star[0] = 4;
    entry1.star[1] = 42;
    entry1.star[2] = 420;
    entry1.star[3] = 4200;
    entry1.star[4] = 42000;
    entry1.field[0] = 5;
    entry1.field[1] = 53;
    entry1.field[2] = 530;
    entry1.field[3] = 5300;
    entry1.field[4] = 53000;
    entry1.ids[0] = 0x1234;
    entry1.ids[1] = 0x12341234;
    entry1.ids[2] = 0x123412341234ULL;
    entry1.ids[3] = 0x12341234123412ULL;
    entry1.ids[4] = 0x1234123412341234ULL;
    entry1.code_err = 1e-6;
    entry1.quadpix[0] = 1.0;
    entry1.quadpix[1] = 2.0;
    entry1.quadpix[2] = 3.0;
    entry1.quadpix[3] = 4.0;
    entry1.quadpix[4] = 5.0;
    entry1.quadpix[5] = 6.0;
    entry1.quadpix[6] = 7.0;
    entry1.quadpix[7] = 8.0;
    entry1.quadpix[8] = 9.0;
    entry1.quadpix[9] = 10.0;
    entry1.quadxyz[0] = 11.0;
    entry1.quadxyz[1] = 22.0;
    entry1.quadxyz[2] = 33.0;
    entry1.quadxyz[3] = 44.0;
    entry1.quadxyz[4] = 55.0;
    entry1.quadxyz[5] = 66.0;
    entry1.quadxyz[6] = 77.0;
    entry1.quadxyz[7] = 88.0;
    entry1.quadxyz[8] = 99.0;
    entry1.quadxyz[9] = 110.0;
    entry1.quadxyz[10] = 111.0;
    entry1.quadxyz[11] = 122.0;
    entry1.quadxyz[12] = 133.0;
    entry1.quadxyz[13] = 144.0;
    entry1.quadxyz[14] = 155.0;
    entry1.dimquads = 5;
    entry1.quad_npeers = 2;
    //entry1.noverlap = 10;
    entry1.nconflict = 1;
    entry1.nfield = 20;
    entry1.nindex = 100;
    entry1.logodds = 450.0;
    entry1.nagree = 1;
    //entry1.scale = 1e100;
    entry1.fieldnum = 7;
    entry1.fieldfile = 43;
    entry1.indexid = 604;
    entry1.healpix = 4;
    sprintf(entry1.fieldname, "%s", "Hello Kitty");
    entry1.parity = TRUE;
    entry1.quads_tried = 600;
    entry1.quads_matched = 500;
    entry1.quads_scaleok = 400;
    entry1.nverified = 400;
    entry1.timeused = 4.05;

    matchobj_compute_derived(&entry1);

    out = matchfile_open_for_writing(fn);
    CuAssertPtrNotNull(tc, out);
    CuAssertIntEquals(tc, 0, matchfile_count(out));
    CuAssertIntEquals(tc, 0, matchfile_write_headers(out));
    CuAssertIntEquals(tc, 0, matchfile_write_match(out, &entry1));
    CuAssertIntEquals(tc, 1, matchfile_count(out));
    CuAssertIntEquals(tc, 0, matchfile_fix_headers(out));
    CuAssertIntEquals(tc, 0, matchfile_close(out));
    out = NULL;

    in = matchfile_open(fn);
    CuAssertPtrNotNull(tc, in);
    CuAssertIntEquals(tc, 1, matchfile_count(in));

    ein1 = matchfile_read_match(in);
    CuAssertPtrNotNull(tc, ein1);

    SAME(quadno);
    SAMESAME(star);
    SAMESAME(field);
    SAMESAME(ids);
    SAMESAME(quadpix);
    SAMESAME(quadxyz);
    SAME(dimquads);
    SAME(quad_npeers);
    //SAME(noverlap);
    SAME(nconflict);
    SAME(nfield);
    SAME(nindex);
    SAME(nagree);
    SAME(wcs_valid);
    SAME(fieldnum);
    SAME(fieldfile);
    SAME(indexid);
    SAME(healpix);
    SAMESAME(fieldname);
    SAME(parity);
    SAME(quads_tried);
    SAME(quads_matched);
    SAME(quads_scaleok);
    SAME(objs_tried);
    SAME(nverified);

    SAMEF(timeused);
    SAMEF(radius);
    SAMEF(scale);
    SAMEF(logodds);
    SAMEF(code_err);
    SAMESAME(center);

    CuAssertIntEquals(tc, 0, memcmp(ein1, &entry1, sizeof(MatchObj)));

    CuAssertIntEquals(tc, 0, matchfile_close(in));
    in = NULL;
}
