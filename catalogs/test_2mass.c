/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "os-features.h"
#include "cutest.h"
#include "2mass.h"
#include "2mass-fits.h"
#include "an-bool.h"
#include "starutil.h"

void check_entry1(CuTest* tc, twomass_entry* entry) {
    CuAssertDblEquals(tc, 1.119851, entry->ra, 1e-6);
    CuAssertDblEquals(tc, -89.91861, entry->dec, 1e-6);
    CuAssertDblEquals(tc, 0.11, deg2arcsec(entry->err_major), 1e-2);
    CuAssertDblEquals(tc, 0.06, deg2arcsec(entry->err_minor), 1e-2);
    CuAssertDblEquals(tc, 90.0, entry->err_angle, 1e-1);
    CuAssertIntEquals(tc, 0, strcmp("00042876-8955069 ", entry->designation));

    CuAssertDblEquals(tc, 12.467, entry->j_m, 1e-3);
    CuAssertDblEquals(tc, 0.018, entry->j_cmsig, 1e-3);
    CuAssertDblEquals(tc, 0.021, entry->j_msigcom, 1e-3);
    CuAssertDblEquals(tc, 359.4, entry->j_snr, 1e-1);

    CuAssertDblEquals(tc, 12.131, entry->h_m, 1e-3);
    CuAssertDblEquals(tc, 0.025, entry->h_cmsig, 1e-3);
    CuAssertDblEquals(tc, 0.026, entry->h_msigcom, 1e-3);
    CuAssertDblEquals(tc, 224.7, entry->h_snr, 1e-1);

    CuAssertDblEquals(tc, 11.963, entry->k_m, 1e-3);
    CuAssertDblEquals(tc, 0.023, entry->k_cmsig, 1e-3);
    CuAssertDblEquals(tc, 0.025, entry->k_msigcom, 1e-3);
    CuAssertDblEquals(tc, 133.7, entry->k_snr, 1e-1);

    CuAssertIntEquals(tc, TWOMASS_QUALITY_A, entry->j_quality);
    CuAssertIntEquals(tc, TWOMASS_QUALITY_A, entry->h_quality);
    CuAssertIntEquals(tc, TWOMASS_QUALITY_A, entry->k_quality);

    CuAssert(tc, "jqual", twomass_quality_flag(entry->j_quality, TWOMASS_QUALITY_A));
    CuAssert(tc, "hqual", twomass_quality_flag(entry->h_quality, TWOMASS_QUALITY_A));
    CuAssert(tc, "kqual", twomass_quality_flag(entry->k_quality, TWOMASS_QUALITY_A));

    CuAssertIntEquals(tc, 2, entry->j_read_flag);
    CuAssertIntEquals(tc, 2, entry->h_read_flag);
    CuAssertIntEquals(tc, 2, entry->k_read_flag);

    CuAssertIntEquals(tc, 1, entry->j_blend_flag);
    CuAssertIntEquals(tc, 1, entry->h_blend_flag);
    CuAssertIntEquals(tc, 1, entry->k_blend_flag);

    CuAssertIntEquals(tc, TWOMASS_CC_NONE, entry->j_cc);
    CuAssertIntEquals(tc, TWOMASS_CC_NONE, entry->h_cc);
    CuAssertIntEquals(tc, TWOMASS_CC_NONE, entry->k_cc);

    CuAssertIntEquals(tc, 6, entry->j_ndet_M);
    CuAssertIntEquals(tc, 6, entry->j_ndet_N);
    CuAssertIntEquals(tc, 6, entry->h_ndet_M);
    CuAssertIntEquals(tc, 6, entry->h_ndet_N);
    CuAssertIntEquals(tc, 6, entry->k_ndet_M);
    CuAssertIntEquals(tc, 6, entry->k_ndet_N);

    CuAssertDblEquals(tc, 37.2, deg2arcsec(entry->proximity), 1e-1);
    CuAssertDblEquals(tc, 245, entry->prox_angle, 1e-1);
    CuAssertIntEquals(tc, 1329023254, entry->prox_key);

    CuAssertIntEquals(tc, 0, entry->galaxy_contam);
    CuAssertIntEquals(tc, FALSE, entry->minor_planet);
    CuAssertIntEquals(tc, 1101364107, entry->key);
    CuAssertIntEquals(tc, FALSE, entry->northern_hemisphere);
    CuAssertIntEquals(tc, 2000, entry->date_year);
    CuAssertIntEquals(tc, 9, entry->date_month);
    CuAssertIntEquals(tc, 22, entry->date_day);
    CuAssertIntEquals(tc, 64, entry->scan);

    CuAssertDblEquals(tc, 302.951, entry->glon, 1e-3);
    CuAssertDblEquals(tc, -27.208, entry->glat, 1e-3);
    CuAssertDblEquals(tc, 1.6, deg2arcsec(entry->x_scan), 1e-1);
    CuAssertDblEquals(tc, 2451809.7124, entry->jdate, 1e-4);

    CuAssertDblEquals(tc, 1.07, entry->j_psfchi, 1e-2);
    CuAssertDblEquals(tc, 1.18, entry->h_psfchi, 1e-2);
    CuAssertDblEquals(tc, 0.81, entry->k_psfchi, 1e-2);

    CuAssertDblEquals(tc, 12.481, entry->j_m_stdap, 1e-3);
    CuAssertDblEquals(tc,  0.014, entry->j_msig_stdap, 1e-3);
    CuAssertDblEquals(tc, 12.112, entry->h_m_stdap, 1e-3);
    CuAssertDblEquals(tc,  0.028, entry->h_msig_stdap, 1e-3);
    CuAssertDblEquals(tc, 11.980, entry->k_m_stdap, 1e-3);
    CuAssertDblEquals(tc,  0.012, entry->k_msig_stdap, 1e-3);

    CuAssertDblEquals(tc,  332, deg2arcsec(entry->dist_edge_ns), 1e-1);
    CuAssertDblEquals(tc,  251, deg2arcsec(entry->dist_edge_ew), 1e-1);
    CuAssertIntEquals(tc, FALSE, entry->dist_flag_ns);
    CuAssertIntEquals(tc, FALSE, entry->dist_flag_ew);
    CuAssertIntEquals(tc, 1, entry->dup_src);
    CuAssertIntEquals(tc, TRUE, entry->use_src);
    CuAssertIntEquals(tc, TWOMASS_ASSOCIATION_NONE, entry->association);

    CuAssert(tc, "dist null", !isfinite(entry->dist_opt));
    CuAssert(tc, "dist null2", twomass_is_null_float(entry->dist_opt));
    CuAssert(tc, "phiopt null", twomass_is_null_float(entry->phi_opt));
    CuAssert(tc, "bmopt null", twomass_is_null_float(entry->b_m_opt));
    CuAssert(tc, "vrmopt null", twomass_is_null_float(entry->vr_m_opt));
    CuAssertIntEquals(tc, 0, entry->nopt_mchs);
    CuAssertIntEquals(tc, TWOMASS_KEY_NULL, entry->xsc_key);

    CuAssertIntEquals(tc, 59038, entry->scan_key);
    CuAssertIntEquals(tc, 1357874, entry->coadd_key);
    CuAssertIntEquals(tc, 267, entry->coadd);
}

void test_read_2mass(CuTest* tc) {
    // Read some sample lines from the raw 2MASS catalog.
    // psc_aaa.gz line 1.
    char* line1 = "1.119851|-89.91861|0.11|0.06|90|00042876-8955069 |"
        "12.467|0.018|0.021|359.4|" // jmag
        "12.131|0.025|0.026|224.7|" // hmag
        "11.963|0.023|0.025|133.7|" // kmag
        "AAA|222|111|000|" // quality through cc
        "666666|" // ndet
        "37.2|245|1329023254|" // prox
        "0|0|1101364107|s|2000-09-22|64|" // galaxy_contam through scan
        "302.951|-27.208|1.6|2451809.7124|" // glat through jdate
        "1.07|1.18|0.81|" // psfchi
        "12.481|0.014|12.112|0.028|11.98|0.012|" // stdap
        "332|251|sw|1|1|0|" // dist through association
        "\\N|\\N|\\N|\\N|0|\\N|" // match
        "59038|1357874|267";
    twomass_entry entry;
    twomass_fits* out;
    twomass_fits* in;
    twomass_entry* ein;
    char* fn = "/tmp/test-2mass-0";

    memset(&entry, 0, sizeof(twomass_entry));
    CuAssertIntEquals(tc, 0, twomass_parse_entry(&entry, line1));

    check_entry1(tc, &entry);

    out = twomass_fits_open_for_writing(fn);
    CuAssertPtrNotNull(tc, out);
    CuAssertIntEquals(tc, 0, twomass_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, twomass_fits_write_headers(out));
    CuAssertIntEquals(tc, 0, twomass_fits_write_entry(out, &entry));
    CuAssertIntEquals(tc, 1, twomass_fits_count_entries(out));
    CuAssertIntEquals(tc, 0, twomass_fits_fix_headers(out));
    CuAssertIntEquals(tc, 0, twomass_fits_close(out));
    out = NULL;

    memset(&entry, 0, sizeof(twomass_entry));

    in = twomass_fits_open(fn);
    CuAssertPtrNotNull(tc, in);
    CuAssertIntEquals(tc, 1, twomass_fits_count_entries(in));
    ein = twomass_fits_read_entry(in);
    CuAssertPtrNotNull(tc, ein);
    check_entry1(tc, ein);
    CuAssertIntEquals(tc, 0, twomass_fits_close(in));
    in = NULL;

}


void test_fits_empty(CuTest* tc) {
    char* fn = "/tmp/test-2mass-1";
    twomass_fits* out;
    out = twomass_fits_open_for_writing(fn);
    CuAssertPtrNotNull(tc, out);
    CuAssertIntEquals(tc, 0, twomass_fits_write_headers(out));
    CuAssertIntEquals(tc, 0, twomass_fits_fix_headers(out));
    CuAssertIntEquals(tc, 0, twomass_fits_close(out));
}
