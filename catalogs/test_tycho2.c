/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cutest.h"
#include "tycho2.h"
#include "tycho2-fits.h"
#include "starutil.h"

static double deg2mas(double x) {
    return 1000.0 * deg2arcsec(x);
}

static void check_line1(CuTest* tc, const tycho2_entry* e) {
    CuAssertIntEquals(tc, 1, e->tyc1);
    CuAssertIntEquals(tc, 8, e->tyc2);
    CuAssertIntEquals(tc, 1, e->tyc3);
    CuAssertIntEquals(tc, 0, e->photo_center);
    CuAssertIntEquals(tc, 0, e->no_motion);

    CuAssertDblEquals(tc, 2.31750494, e->mean_ra,  1e-8);
    CuAssertDblEquals(tc, 2.23184345, e->mean_dec, 1e-8);
    CuAssertDblEquals(tc, -16.3, e->pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc,  -9.0, e->pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc, 68.0, deg2mas(e->sigma_mean_ra),  1e-1);
    CuAssertDblEquals(tc, 73.0, deg2mas(e->sigma_mean_dec), 1e-1);
    CuAssertDblEquals(tc, 1.7, e->sigma_pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc, 1.8, e->sigma_pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc, 1958.89, e->epoch_mean_ra,  1e-2);
    CuAssertDblEquals(tc, 1951.94, e->epoch_mean_dec, 1e-2);

    CuAssertIntEquals(tc, 4, e->nobs);

    CuAssertDblEquals(tc, 1.0, e->goodness_mean_ra,  1e-1);
    CuAssertDblEquals(tc, 1.0, e->goodness_mean_dec, 1e-1);
    CuAssertDblEquals(tc, 0.9, e->goodness_pm_ra,  1e-1);
    CuAssertDblEquals(tc, 1.0, e->goodness_pm_dec, 1e-1);

    CuAssertDblEquals(tc, 12.146, e->mag_BT,   1e-3);
    CuAssertDblEquals(tc,  0.158, e->sigma_BT, 1e-3);
    CuAssertDblEquals(tc, 12.146, e->mag_VT,   1e-3);
    CuAssertDblEquals(tc,  0.223, e->sigma_VT, 1e-3);
    CuAssertDblEquals(tc,  0.0, e->mag_HP,   1e-3);
    CuAssertDblEquals(tc,  0.0, e->sigma_HP, 1e-3);

    CuAssertDblEquals(tc, -1.0, e->prox, 1e-1);
    CuAssertIntEquals(tc, FALSE, e->tycho1_star);
    CuAssertIntEquals(tc, 0, e->hipparcos_id);
    CuAssertIntEquals(tc, 0, strlen(e->hip_ccdm));

    CuAssertDblEquals(tc, 2.31754222, e->ra,  1e-8);
    CuAssertDblEquals(tc, 2.23186444, e->dec, 1e-8);
    CuAssertDblEquals(tc, 1.67, e->epoch_ra - 1990.0, 1e-2);
    CuAssertDblEquals(tc, 1.54, e->epoch_dec- 1990.0, 1e-2);

    CuAssertDblEquals(tc,  88.0, deg2mas(e->sigma_ra),  1e-1);
    CuAssertDblEquals(tc, 100.8, deg2mas(e->sigma_dec), 1e-1);

    CuAssertIntEquals(tc, FALSE, e->double_star);
    CuAssertIntEquals(tc, FALSE, e->photo_center_treatment);

    CuAssertDblEquals(tc, -0.2, e->correlation, 1e-1);
}

static void check_line2(CuTest* tc, const tycho2_entry* e) {
    CuAssertIntEquals(tc, 1, e->tyc1);
    CuAssertIntEquals(tc, 13, e->tyc2);
    CuAssertIntEquals(tc, 1, e->tyc3);
    CuAssertIntEquals(tc, 0, e->photo_center);
    CuAssertIntEquals(tc, 0, e->no_motion);

    CuAssertDblEquals(tc, 1.12558209, e->mean_ra,  1e-8);
    CuAssertDblEquals(tc, 2.26739400, e->mean_dec, 1e-8);
    CuAssertDblEquals(tc,  27.7, e->pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc,  -0.5, e->pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc,  9.0, deg2mas(e->sigma_mean_ra),  1e-1);
    CuAssertDblEquals(tc, 12.0, deg2mas(e->sigma_mean_dec), 1e-1);
    CuAssertDblEquals(tc, 1.2, e->sigma_pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc, 1.2, e->sigma_pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc, 1990.76, e->epoch_mean_ra,  1e-2);
    CuAssertDblEquals(tc, 1989.25, e->epoch_mean_dec, 1e-2);

    CuAssertIntEquals(tc, 8, e->nobs);

    CuAssertDblEquals(tc, 1.0, e->goodness_mean_ra,  1e-1);
    CuAssertDblEquals(tc, 0.8, e->goodness_mean_dec, 1e-1);
    CuAssertDblEquals(tc, 1.0, e->goodness_pm_ra,  1e-1);
    CuAssertDblEquals(tc, 0.7, e->goodness_pm_dec, 1e-1);

    CuAssertDblEquals(tc, 10.488, e->mag_BT,   1e-3);
    CuAssertDblEquals(tc,  0.038, e->sigma_BT, 1e-3);
    CuAssertDblEquals(tc,  8.670, e->mag_VT,   1e-3);
    CuAssertDblEquals(tc,  0.015, e->sigma_VT, 1e-3);
    CuAssertDblEquals(tc,  0.0, e->mag_HP,   1e-3);
    CuAssertDblEquals(tc,  0.0, e->sigma_HP, 1e-3);

    CuAssertDblEquals(tc, -1.0, e->prox, 1e-1);
    CuAssertIntEquals(tc, TRUE, e->tycho1_star);
    CuAssertIntEquals(tc, 0, e->hipparcos_id);
    CuAssertIntEquals(tc, 0, strlen(e->hip_ccdm));

    CuAssertDblEquals(tc, 1.12551889, e->ra,  1e-8);
    CuAssertDblEquals(tc, 2.26739556, e->dec, 1e-8);
    CuAssertDblEquals(tc, 1.81, e->epoch_ra - 1990.0, 1e-2);
    CuAssertDblEquals(tc, 1.52, e->epoch_dec- 1990.0, 1e-2);

    CuAssertDblEquals(tc,   9.3, deg2mas(e->sigma_ra),  1e-1);
    CuAssertDblEquals(tc,  12.7, deg2mas(e->sigma_dec), 1e-1);

    CuAssertIntEquals(tc, FALSE, e->double_star);
    CuAssertIntEquals(tc, FALSE, e->photo_center_treatment);

    CuAssertDblEquals(tc, -0.2, e->correlation, 1e-1);
}

static void check_line3(CuTest* tc, const tycho2_entry* e) {
    CuAssertIntEquals(tc, 5, e->tyc1);
    CuAssertIntEquals(tc, 1505, e->tyc2);
    CuAssertIntEquals(tc, 1, e->tyc3);
    CuAssertIntEquals(tc, 0, e->photo_center);
    CuAssertIntEquals(tc, 0, e->no_motion);

    CuAssertDblEquals(tc, 3.25200241, e->mean_ra,  1e-8);
    CuAssertDblEquals(tc, 2.94960582, e->mean_dec, 1e-8);
    CuAssertDblEquals(tc,  47.8, e->pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc, -13.6, e->pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc,  8.0, deg2mas(e->sigma_mean_ra),  1e-1);
    CuAssertDblEquals(tc, 11.0, deg2mas(e->sigma_mean_dec), 1e-1);
    CuAssertDblEquals(tc, 1.0, e->sigma_pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc, 1.0, e->sigma_pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc, 1990.57, e->epoch_mean_ra,  1e-2);
    CuAssertDblEquals(tc, 1989.33, e->epoch_mean_dec, 1e-2);

    CuAssertIntEquals(tc, 11, e->nobs);

    CuAssertDblEquals(tc, 0.6, e->goodness_mean_ra,  1e-1);
    CuAssertDblEquals(tc, 1.2, e->goodness_mean_dec, 1e-1);
    CuAssertDblEquals(tc, 0.7, e->goodness_pm_ra,  1e-1);
    CuAssertDblEquals(tc, 1.2, e->goodness_pm_dec, 1e-1);

    CuAssertDblEquals(tc,  9.275, e->mag_BT,   1e-3);
    CuAssertDblEquals(tc,  0.020, e->sigma_BT, 1e-3);
    CuAssertDblEquals(tc,  8.738, e->mag_VT,   1e-3);
    CuAssertDblEquals(tc,  0.015, e->sigma_VT, 1e-3);
    CuAssertDblEquals(tc,  0.0, e->mag_HP,   1e-3);
    CuAssertDblEquals(tc,  0.0, e->sigma_HP, 1e-3);

    CuAssertDblEquals(tc, -1.0, e->prox, 1e-1);
    CuAssertIntEquals(tc, TRUE, e->tycho1_star);
    CuAssertIntEquals(tc, 1040, e->hipparcos_id);
    CuAssertIntEquals(tc, 2, strlen(e->hip_ccdm));
    CuAssertIntEquals(tc, 0, strcmp(e->hip_ccdm, "AB"));

    CuAssertDblEquals(tc, 3.25189250, e->ra,  1e-8);
    CuAssertDblEquals(tc, 2.94963750, e->dec, 1e-8);
    CuAssertDblEquals(tc, 1.76, e->epoch_ra - 1990.0, 1e-2);
    CuAssertDblEquals(tc, 1.45, e->epoch_dec- 1990.0, 1e-2);

    CuAssertDblEquals(tc,   8.2, deg2mas(e->sigma_ra),  1e-1);
    CuAssertDblEquals(tc,  10.5, deg2mas(e->sigma_dec), 1e-1);

    CuAssertIntEquals(tc, FALSE, e->double_star);
    CuAssertIntEquals(tc,  TRUE, e->photo_center_treatment);

    CuAssertDblEquals(tc, -0.2, e->correlation, 1e-1);
}

static void check_supp1(CuTest* tc, const tycho2_entry* e) {
    CuAssertIntEquals(tc, 2, e->tyc1);
    CuAssertIntEquals(tc, 1127, e->tyc2);
    CuAssertIntEquals(tc, 2, e->tyc3);
    CuAssertIntEquals(tc, 0, e->photo_center);
    CuAssertIntEquals(tc, 0, e->no_motion);

    CuAssertIntEquals(tc, FALSE, e->tycho1_star);

    CuAssertDblEquals(tc, 0.0, e->mean_ra,  1e-8);
    CuAssertDblEquals(tc, 0.0, e->mean_dec, 1e-8);
    CuAssertDblEquals(tc, 4.36837051, e->ra,  1e-8);
    CuAssertDblEquals(tc, 0.31948829, e->dec, 1e-8);

    CuAssertDblEquals(tc, -32.1, e->pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc,  -9.1, e->pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc,  13.6, deg2mas(e->sigma_ra),  1e-1);
    CuAssertDblEquals(tc,   8.3, deg2mas(e->sigma_dec), 1e-1);
    CuAssertDblEquals(tc, 8.2, e->sigma_pm_ra  * 1000.0, 1e-1);
    CuAssertDblEquals(tc, 5.0, e->sigma_pm_dec * 1000.0, 1e-1);

    CuAssertDblEquals(tc, 10.279, e->mag_HP,   1e-3);
    CuAssertDblEquals(tc,  0.041, e->sigma_HP, 1e-3);

    CuAssertDblEquals(tc, 75.0, deg2arcsec(e->prox) * 10.0, 1e-1);

    CuAssertIntEquals(tc, 1397, e->hipparcos_id);
    CuAssertIntEquals(tc, 1, strlen(e->hip_ccdm));
    CuAssertIntEquals(tc, 0, strcmp(e->hip_ccdm, "B"));


    CuAssertDblEquals(tc,  0.0, deg2mas(e->sigma_mean_ra),  1e-1);
    CuAssertDblEquals(tc,  0.0, deg2mas(e->sigma_mean_dec), 1e-1);
    CuAssertDblEquals(tc, 0.0, e->epoch_mean_ra,  1e-2);
    CuAssertDblEquals(tc, 0.0, e->epoch_mean_dec, 1e-2);
    CuAssertIntEquals(tc, 0, e->nobs);
    CuAssertDblEquals(tc, 0.0, e->goodness_mean_ra,  1e-1);
    CuAssertDblEquals(tc, 0.0, e->goodness_mean_dec, 1e-1);
    CuAssertDblEquals(tc, 0.0, e->goodness_pm_ra,  1e-1);
    CuAssertDblEquals(tc, 0.0, e->goodness_pm_dec, 1e-1);
    CuAssertDblEquals(tc,  0.0, e->mag_BT,   1e-3);
    CuAssertDblEquals(tc,  0.0, e->sigma_BT, 1e-3);
    CuAssertDblEquals(tc,  0.0, e->mag_VT,   1e-3);
    CuAssertDblEquals(tc,  0.0, e->sigma_VT, 1e-3);
    CuAssertDblEquals(tc,  0.0, e->epoch_ra, 1e-2);
    CuAssertDblEquals(tc,  0.0, e->epoch_dec, 1e-2);
    CuAssertIntEquals(tc, FALSE, e->double_star);
    CuAssertIntEquals(tc, FALSE, e->photo_center_treatment);
    CuAssertDblEquals(tc,  0.0, e->correlation, 1e-1);
}

void test_read_raw(CuTest* tc) {
    char* line1 = "0001 00008 1| |"
        "  2.31750494|  2.23184345|  -16.3|   -9.0|"
        " 68| 73| 1.7| 1.8|"
        "1958.89|1951.94| 4|"
        "1.0|1.0|0.9|1.0|"
        "12.146|0.158|12.146|0.223|"
        "999| |         |"
        "  2.31754222|  2.23186444|1.67|1.54|"
        " 88.0|100.8|"
        " |-0.2\r\n";

    char* line2 = "0001 00013 1| |"
        "  1.12558209|  2.26739400|   27.7|   -0.5|"
        "  9| 12| 1.2| 1.2|"
        "1990.76|1989.25| 8|"
        "1.0|0.8|1.0|0.7|"
        "10.488|0.038| 8.670|0.015|"
        "999|T|         |"
        "  1.12551889|  2.26739556|1.81|1.52|"
        "  9.3| 12.7|"
        " |-0.2\r\n";
    // not the third line in the file...
    char* line3 = "0005 01505 1| |"
        "  3.25200241|  2.94960582|   47.8|  -13.6|"
        "  8| 11| 1.0| 1.0|"
        "1990.57|1989.33|11|"
        "0.6|1.2|0.7|1.2|"
        " 9.275|0.020| 8.738|0.015|"
        "999|T|  1040AB |"
        "  3.25189250|  2.94963750|1.76|1.45|"
        "  8.2| 10.5|"
        "P|-0.2\r\n";

    // line 3 from suppl_1.dat
    char* supp1 = "0002 01127 2|H|"
        "004.36837051|+00.31948829|  -32.1|   -9.1|"
        " 13.6|  8.3|  8.2|  5.0|"
        "H|      |     |"
        "10.279|0.041|"
        " 75| |  1397B\r\n";

    char* fn = "/tmp/test-tycho2-0";

    tycho2_entry entry1;
    tycho2_entry entry2;
    tycho2_entry entry3;
    tycho2_entry entry4;
    tycho2_fits* out;
    tycho2_fits* in;
    tycho2_entry* ein1;
    tycho2_entry* ein2;
    tycho2_entry* ein3;
    tycho2_entry* ein4;

    memset(&entry1, 0, sizeof(tycho2_entry));
    memset(&entry2, 0, sizeof(tycho2_entry));
    memset(&entry3, 0, sizeof(tycho2_entry));
    memset(&entry4, 0, sizeof(tycho2_entry));

    CuAssertIntEquals(tc, 0, tycho2_guess_is_supplement(line1));
    CuAssertIntEquals(tc, 0, tycho2_parse_entry(line1, &entry1));
    check_line1(tc, &entry1);

    CuAssertIntEquals(tc, 0, tycho2_guess_is_supplement(line2));
    CuAssertIntEquals(tc, 0, tycho2_parse_entry(line2, &entry2));
    check_line2(tc, &entry2);

    CuAssertIntEquals(tc, 0, tycho2_guess_is_supplement(line3));
    CuAssertIntEquals(tc, 0, tycho2_parse_entry(line3, &entry3));
    check_line3(tc, &entry3);

    CuAssertIntEquals(tc, 1, tycho2_guess_is_supplement(supp1));
    CuAssertIntEquals(tc, 0, tycho2_supplement_parse_entry(supp1, &entry4));
    check_supp1(tc, &entry4);

    out = tycho2_fits_open_for_writing(fn);
    CuAssertPtrNotNull(tc, out);
    CuAssertIntEquals(tc, 0, tycho2_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, tycho2_fits_write_headers(out));
    CuAssertIntEquals(tc, 0, tycho2_fits_write_entry(out, &entry1));
    CuAssertIntEquals(tc, 0, tycho2_fits_write_entry(out, &entry2));
    CuAssertIntEquals(tc, 0, tycho2_fits_write_entry(out, &entry3));
    CuAssertIntEquals(tc, 0, tycho2_fits_write_entry(out, &entry4));
    CuAssertIntEquals(tc, 4, tycho2_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, tycho2_fits_fix_headers(out));
    CuAssertIntEquals(tc, 0, tycho2_fits_close(out));
    out = NULL;

    memset(&entry1, 0, sizeof(tycho2_entry));
    memset(&entry2, 0, sizeof(tycho2_entry));
    memset(&entry3, 0, sizeof(tycho2_entry));
    memset(&entry4, 0, sizeof(tycho2_entry));

    in = tycho2_fits_open(fn);
    CuAssertPtrNotNull(tc, in);
    CuAssertIntEquals(tc, 4, tycho2_fits_count_entries(in));

    ein1 = tycho2_fits_read_entry(in);
    CuAssertPtrNotNull(tc, ein1);
    check_line1(tc, ein1);

    ein2 = tycho2_fits_read_entry(in);
    CuAssertPtrNotNull(tc, ein2);
    check_line2(tc, ein2);

    ein3 = tycho2_fits_read_entry(in);
    CuAssertPtrNotNull(tc, ein3);
    check_line3(tc, ein3);

    ein4 = tycho2_fits_read_entry(in);
    CuAssertPtrNotNull(tc, ein4);
    check_supp1(tc, ein4);

    CuAssertIntEquals(tc, 0, tycho2_fits_close(in));
    in = NULL;
}
