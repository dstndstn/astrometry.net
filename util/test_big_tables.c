/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stddef.h>

#include "fitstable.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "errors.h"
#include "log.h"

#include "cutest.h"

void test_big_table(CuTest* ct) {
    const char* filename = "/home/boss/products/NULL/wise/trunk/fits/wise-allsky-cat-part01.fits";
    fitstable_t* tab;
    int offset;
    int nelems;
    char* buffer;
    const char* racol = "RA";
    const char* deccol = "DEC";
    tfits_type any, dubl;

    fits_use_error_system();
    log_init(LOG_VERB);
    if (!file_exists(filename)) {
        printf("File %s doesn't exist; skipping test.\n", filename);
        return;
    }
    tab = fitstable_open(filename);
    if (!tab) {
        ERROR("Failed to open FITS table \"%s\"", filename);
        CuFail(ct, "Failed to open FITS table");
        return;
    }

    offset = 2000000;
    nelems = 1000;
    buffer = malloc(nelems * sizeof(double) * 2);
    CuAssertPtrNotNull(ct, buffer);
    //fitstable_read_nrows_data(table, offset, nelems, buffer);

    any = fitscolumn_any_type();
    dubl = fitscolumn_double_type();

    fitstable_add_read_column_struct(tab, dubl, 1, 0, any, racol, TRUE);
    fitstable_add_read_column_struct(tab, dubl, 1, sizeof(double), any, deccol, TRUE);

    if (fitstable_read_extension(tab, 1)) {
        ERROR("Failed to find RA and DEC columns (called \"%s\" and \"%s\" in the FITS file)", racol, deccol);
        CuFail(ct, "Failde to find RA,Dec\n");
    }

    if (fitstable_read_structs(tab, buffer, 2 * sizeof(double), offset, nelems)) {
        CuFail(ct, "Failed to fitstable_read_structs");
        return;
    }

}

