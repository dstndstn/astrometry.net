/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cutest.h"
#include "nomad.h"
#include "nomad-fits.h"
#include "an-bool.h"
#include "an-endian.h"
#include "starutil.h"

static void check_line1(CuTest* tc, const nomad_entry* e) {
    CuAssertIntEquals(tc, 4125041, rint(deg2arcsec(e->ra) / 0.001));
    CuAssertIntEquals(tc,  292987, rint((deg2arcsec(e->dec + 90.0)) / 0.001));
    CuAssertIntEquals(tc,      15, rint((deg2arcsec(e->sigma_racosdec)) / 0.001));
    CuAssertIntEquals(tc,      28, rint((deg2arcsec(e->sigma_dec)) / 0.001));
    CuAssertIntEquals(tc,     190, rint(e->pm_racosdec / 0.0001));
    CuAssertIntEquals(tc,    -187, (int32_t)rint(e->pm_dec / 0.0001));
    CuAssertIntEquals(tc, 4294967109U, (uint32_t)((int32_t)rint(e->pm_dec / 0.0001)));
    CuAssertIntEquals(tc,      61, rint(e->sigma_pm_racosdec / 0.0001));
    CuAssertIntEquals(tc,      62, rint(e->sigma_pm_dec / 0.0001));
    CuAssertIntEquals(tc, 1998246, rint(e->epoch_ra / 0.001));
    CuAssertIntEquals(tc, 1997659, rint(e->epoch_dec / 0.001));
    CuAssertIntEquals(tc,   14710, rint(e->mag_B / 0.001));
    CuAssertIntEquals(tc,   13690, rint(e->mag_V / 0.001));
    CuAssertIntEquals(tc,   13680, rint(e->mag_R / 0.001));
    CuAssertIntEquals(tc,   12467, rint(e->mag_J / 0.001));
    CuAssertIntEquals(tc,   12131, rint(e->mag_H / 0.001));
    CuAssertIntEquals(tc,   11963, rint(e->mag_K / 0.001));
    CuAssertIntEquals(tc,       1, e->usnob_id);
    CuAssertIntEquals(tc, 1101364107, e->twomass_id);
    CuAssertIntEquals(tc,       1, e->yb6_id);
    CuAssertIntEquals(tc,       2, e->ucac2_id);
    CuAssertIntEquals(tc,       0, e->tycho2_id);

    CuAssertIntEquals(tc, 536875740,
                      ((e->astrometry_src << 0) |
                       (e->blue_src       << 3) |
                       (e->visual_src     << 6) |
                       (e->red_src        << 9) |
                       (e->usnob_fail       ?    0x1000 : 0) |
                       (e->twomass_fail     ?    0x2000 : 0) |
                       (e->tycho_astrometry ?   0x10000 : 0) |
                       (e->alt_radec        ?   0x20000 : 0) |
                       (e->alt_ucac         ?   0x80000 : 0) |
                       (e->alt_tycho        ?  0x100000 : 0) |
                       (e->blue_o           ?  0x200000 : 0) |
                       (e->red_e            ?  0x400000 : 0) |
                       (e->twomass_only     ?  0x800000 : 0) |
                       (e->hipp_astrometry  ? 0x1000000 : 0) |
                       (e->diffraction      ? 0x2000000 : 0) |
                       (e->confusion        ? 0x4000000 : 0) |
                       (e->bright_confusion ? 0x8000000 : 0) |
                       (e->bright_artifact  ? 0x10000000 : 0) |
                       (e->standard         ? 0x20000000 : 0)));

}

void test_read_nomad(CuTest* tc) {
    // od -N 88 --width=10 -t x1 ~/raid1/NOMAD/000/m0000.cat | gawk '{for(i=2;i<=NF;i++){printf("0x%s, ",$i);}printf("\n");}'
    uint8_t line1[] = {
        0x71, 0xf1, 0x3e, 0x00, 0x7b, 0x78, 0x04, 0x00, 0x0f, 0x00, 
        0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0xbe, 0x00, 0x00, 0x00, 
        0x45, 0xff, 0xff, 0xff, 0x3d, 0x00, 0x00, 0x00, 0x3e, 0x00, 
        0x00, 0x00, 0xa6, 0x7d, 0x1e, 0x00, 0x5b, 0x7b, 0x1e, 0x00, 
        0x76, 0x39, 0x00, 0x00, 0x7a, 0x35, 0x00, 0x00, 0x70, 0x35, 
        0x00, 0x00, 0xb3, 0x30, 0x00, 0x00, 0x63, 0x2f, 0x00, 0x00, 
        0xbb, 0x2e, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x8b, 0x7b, 
        0xa5, 0x41, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 
        0x00, 0x00, 0x00, 0x00, 0xdc, 0x12, 0x00, 0x20, 
    };
    char* fn = "/tmp/test-nomad-0";
    int i;

    for (i=0; i<sizeof(line1)/sizeof(uint32_t); i++) {
        uint32_t* ul = (uint32_t*)line1;
        printf("byte %2i: %010u\n", 4*i, u32_letoh(ul[i]));
    }

    nomad_entry entry1;
    nomad_fits* out;
    nomad_fits* in;
    nomad_entry* ein1;

    memset(&entry1, 0, sizeof(nomad_entry));

    CuAssertIntEquals(tc, 0, nomad_parse_entry(&entry1, line1));
    check_line1(tc, &entry1);

    out = nomad_fits_open_for_writing(fn);
    CuAssertPtrNotNull(tc, out);
    CuAssertIntEquals(tc, 0, nomad_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, nomad_fits_write_headers(out));
    CuAssertIntEquals(tc, 0, nomad_fits_write_entry(out, &entry1));
    CuAssertIntEquals(tc, 1, nomad_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, nomad_fits_fix_headers(out));
    CuAssertIntEquals(tc, 0, nomad_fits_close(out));
    out = NULL;

    memset(&entry1, 0, sizeof(nomad_entry));

    in = nomad_fits_open(fn);
    CuAssertPtrNotNull(tc, in);
    CuAssertIntEquals(tc, 1, nomad_fits_count_entries(in));

    ein1 = nomad_fits_read_entry(in);
    CuAssertPtrNotNull(tc, ein1);
    check_line1(tc, ein1);

    CuAssertIntEquals(tc, 0, nomad_fits_close(in));
    in = NULL;
}
