/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cutest.h"
#include "usnob.h"
#include "usnob-fits.h"
#include "an-bool.h"
#include "an-endian.h"
#include "starutil.h"

static void assertZeroObs(CuTest* tc, const struct observation* o) {
    CuAssertIntEquals(tc,     0, o->mag);
    CuAssertIntEquals(tc,     0, o->field);
    CuAssertIntEquals(tc,     0, o->survey);
    CuAssertIntEquals(tc,     0, o->star_galaxy);
    CuAssertIntEquals(tc,     0, o->xi_resid);
    CuAssertIntEquals(tc,     0, o->eta_resid);
    CuAssertIntEquals(tc,     0, o->calibration);
    CuAssertIntEquals(tc,     0, o->pmmscan);
}

static void check_line1(CuTest* tc, const usnob_entry* e) {
    const struct observation* o;

    CuAssertIntEquals(tc, 391376, rint(deg2arcsec(e->ra) / 0.01));

    CuAssertIntEquals(tc,  29304, rint(deg2arcsec(e->dec + 90) / 0.01));

    CuAssertIntEquals(tc,     0, e->motion_catalog);
    CuAssertIntEquals(tc,     8, rint(e->pm_prob / 0.1));
    CuAssertIntEquals(tc,  4997, rint((e->pm_dec + 10.0) / 0.002));
    CuAssertIntEquals(tc,  4999, rint((e->pm_ra  + 10.0) / 0.002));

    CuAssertIntEquals(tc,     0, e->diffraction_spike);
    CuAssertIntEquals(tc,     3, e->ndetections);
    CuAssertIntEquals(tc,     0, rint(deg2arcsec(e->sigma_dec_fit) / 0.1));
    CuAssertIntEquals(tc,     0, rint(deg2arcsec(e->sigma_ra_fit) / 0.1));
    CuAssertIntEquals(tc,     0, rint(e->sigma_pm_dec / 0.001));
    CuAssertIntEquals(tc,     4, rint(e->sigma_pm_ra  / 0.001));

    CuAssertIntEquals(tc,     1, e->ys4);
    CuAssertIntEquals(tc,   369, rint((e->epoch - 1950) / 0.1));
    CuAssertIntEquals(tc,     0, rint(deg2arcsec(e->sigma_dec) / 0.001));
    CuAssertIntEquals(tc,    65, rint(deg2arcsec(e->sigma_ra)  / 0.001));

    o = e->obs + 0;
    assertZeroObs(tc, o);

    o = e->obs + 1;
    CuAssertIntEquals(tc,     9, o->star_galaxy);
    CuAssertIntEquals(tc,     5, o->survey);
    CuAssertIntEquals(tc,     1, o->field);
    CuAssertIntEquals(tc,  1400, rint(o->mag / 0.01));

    o = e->obs + 2;
    CuAssertIntEquals(tc,     3, o->star_galaxy);
    CuAssertIntEquals(tc,     4, o->survey);
    CuAssertIntEquals(tc,     1, o->field);
    CuAssertIntEquals(tc,  1443, rint(o->mag / 0.01));

    o = e->obs + 3;
    CuAssertIntEquals(tc,     0, o->star_galaxy);
    CuAssertIntEquals(tc,     6, o->survey);
    CuAssertIntEquals(tc,     1, o->field);
    CuAssertIntEquals(tc,  1368, rint(o->mag / 0.01));

    o = e->obs + 4;
    assertZeroObs(tc, o);

    o = e->obs + 1;
    CuAssertIntEquals(tc,     1, o->calibration);
    CuAssertIntEquals(tc,  4999, rint((deg2arcsec(o->eta_resid) + 50.0) / 0.01));
    CuAssertIntEquals(tc,  5004, rint((deg2arcsec(o->xi_resid ) + 50.0) / 0.01));

    o = e->obs + 2;
    CuAssertIntEquals(tc,     1, o->calibration);
    CuAssertIntEquals(tc,  5000, rint((deg2arcsec(o->eta_resid) + 50.0) / 0.01));
    CuAssertIntEquals(tc,  4996, rint((deg2arcsec(o->xi_resid ) + 50.0) / 0.01));

    o = e->obs + 3;
    CuAssertIntEquals(tc,     1, o->calibration);
    CuAssertIntEquals(tc,  5000, rint((deg2arcsec(o->eta_resid) + 50.0) / 0.01));
    CuAssertIntEquals(tc,  4998, rint((deg2arcsec(o->xi_resid ) + 50.0) / 0.01));

    o = e->obs + 1;
    CuAssertIntEquals(tc, 228789, o->pmmscan);

    o = e->obs + 2;
    CuAssertIntEquals(tc, 368267, o->pmmscan);

    o = e->obs + 3;
    CuAssertIntEquals(tc, 298646, o->pmmscan);
}

void test_read_usnob(CuTest* tc) {
    // od -N 80 --width=10 -t x1 ~/raid1/USNOB10/000/b0000.cat | gawk '{for(i=2;i<=NF;i++){printf("0x%s, ",$i);}printf("\n");}'
    uint8_t line1[] = {
        0xd0, 0xf8, 0x05, 0x00, 0x78, 0x72, 0x00, 0x00, 0xd7, 0x96,
        0xa9, 0x32, 0x04, 0xa3, 0xe1, 0x11, 0x81, 0x48, 0x99, 0x51,
        0x00, 0x00, 0x00, 0x00, 0x08, 0x06, 0xa0, 0x38, 0xb3, 0x29,
        0x44, 0x14, 0x68, 0xb3, 0x93, 0x03, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xfc, 0xbd, 0xf0, 0x08, 0x04, 0xe5,
        0xf0, 0x08, 0x06, 0xe5, 0xf0, 0x08, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xb5, 0x7d, 0x03, 0x00, 0x8b, 0x9e,
        0x05, 0x00, 0x96, 0x8e, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    char* fn = "/tmp/test-usnob-0";
    int i;

    for (i=0; i<sizeof(line1)/sizeof(uint32_t); i++) {
        uint32_t* ul = (uint32_t*)line1;
        printf("byte %2i: %010u\n", 4*i, u32_letoh(ul[i]));
    }

    usnob_entry entry1;
    usnob_fits* out;
    usnob_fits* in;
    usnob_entry* ein1;

    memset(&entry1, 0, sizeof(usnob_entry));

    CuAssertIntEquals(tc, 0, usnob_parse_entry(line1, &entry1));
    check_line1(tc, &entry1);

    out = usnob_fits_open_for_writing(fn);
    CuAssertPtrNotNull(tc, out);
    CuAssertIntEquals(tc, 0, usnob_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, usnob_fits_write_headers(out));
    CuAssertIntEquals(tc, 0, usnob_fits_write_entry(out, &entry1));
    CuAssertIntEquals(tc, 1, usnob_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, usnob_fits_fix_headers(out));
    CuAssertIntEquals(tc, 0, usnob_fits_close(out));
    out = NULL;

    memset(&entry1, 0, sizeof(usnob_entry));

    in = usnob_fits_open(fn);
    CuAssertPtrNotNull(tc, in);
    CuAssertIntEquals(tc, 1, usnob_fits_count_entries(in));

    ein1 = usnob_fits_read_entry(in);
    CuAssertPtrNotNull(tc, ein1);
    check_line1(tc, ein1);

    CuAssertIntEquals(tc, 0, usnob_fits_close(in));
    in = NULL;
}
