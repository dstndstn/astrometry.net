/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stddef.h>
#include <assert.h>

#include "scamp-catalog.h"
#include "fitstable.h"
#include "fitsioutils.h"
#include "qfits_header.h"
#include "errors.h"
#include "log.h"

scamp_cat_t* scamp_catalog_open_for_writing(const char* filename, anbool ref) {
    scamp_cat_t* scamp;
    scamp = calloc(1, sizeof(scamp_cat_t));
    scamp->table = fitstable_open_for_writing(filename);
    if (!scamp->table) {
        ERROR("Failed to open scamp catalog for writing");
        free(scamp);
        return NULL;
    }
    scamp->ref = ref;
    return scamp;
}

int scamp_catalog_write_field_header(scamp_cat_t* scamp, const qfits_header* hdr) {
    int i, N;
    qfits_header* h;
    char* hdrstring;
    qfits_header* freehdr = NULL;
    tfits_type dubl = fitscolumn_double_type();
    tfits_type i16  = fitscolumn_i16_type();

    if (fitstable_write_primary_header(scamp->table)) {
        ERROR("Failed to write scamp catalog primary header.\n");
        return -1;
    }

    if (!hdr) {
        freehdr = qfits_header_default();
        fits_header_add_int(freehdr, "BITPIX", 0, NULL);
        fits_header_add_int(freehdr, "NAXIS", 2, NULL);
        fits_header_add_int(freehdr, "NAXIS1", 0, NULL);
        fits_header_add_int(freehdr, "NAXIS2", 0, NULL);
        hdr = freehdr;
    }

    // Table is one row x one column (array of (N*80) characters)
    // TDIM1 is "(80, N)"
    // EXTNAME = "LDAC_IMHEAD"
    // 80 is FITS_LINESZ
    // How many cards are in the header?
    N = qfits_header_n(hdr);
    fitstable_add_write_column_array(scamp->table, fitscolumn_char_type(),
                                     N * FITS_LINESZ, "Field Header Card", NULL);

    h = fitstable_get_header(scamp->table);
    fits_header_addf(h, "TDIM1", "shape of header: FITS cards", "(%i, %i)", FITS_LINESZ, N);
    qfits_header_add(h, "EXTNAME", "LDAC_IMHEAD", "", NULL);

    if (fitstable_write_header(scamp->table)) {
        ERROR("Failed to write scamp catalog header.\n");
        return -1;
    }

    // +1 because qfits_header_write_line adds a trailing '\0'.
    hdrstring = malloc(N * FITS_LINESZ + 1);
    for (i=0; i<N; i++)
        if (qfits_header_write_line(hdr, i, hdrstring + i * FITS_LINESZ)) {
            ERROR("Failed to get scamp catalog field header line %i", i);
            return -1;
        }
    if (freehdr)
        qfits_header_destroy(freehdr);
    if (fitstable_write_row(scamp->table, hdrstring)) {
        ERROR("Failed to write scamp catalog field header");
        return -1;
    }
    free(hdrstring);

    if (fitstable_pad_with(scamp->table, ' ') ||
        fitstable_fix_header(scamp->table)) {
        ERROR("Failed to fix scamp catalog header.\n");
        return -1;
    }
    fitstable_next_extension(scamp->table);
    fitstable_clear_table(scamp->table);

    if (scamp->ref) {
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_ref_t, ra),
                                          dubl, "RA", "deg");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_ref_t, dec),
                                          dubl, "DEC", "deg");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_ref_t, err_a),
                                          dubl, "ERR_A", "pixels");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_ref_t, err_b),
                                          dubl, "ERR_B", "pixels");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_ref_t, mag),
                                          dubl, "MAG", "mag");
        // Not used by Scamp!
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_ref_t, err_mag),
                                          dubl, "MAG_ERR", "mag");
    } else {
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_obj_t, x),
                                          dubl, "X_IMAGE", "pixels");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_obj_t, y),
                                          dubl, "Y_IMAGE", "pixels");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_obj_t, err_a),
                                          dubl, "ERR_A", "pixels");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_obj_t, err_b),
                                          dubl, "ERR_B", "pixels");
        // Scamp ignores this.
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_obj_t, err_theta),
                                          dubl, "ERR_THETA", "deg");
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_obj_t, flux),
                                          dubl, "FLUX", NULL);
        fitstable_add_write_column_struct(scamp->table, dubl, 1, offsetof(scamp_obj_t, err_flux),
                                          dubl, "FLUX_ERR", NULL);
        // this column is optional.
        fitstable_add_write_column_struct(scamp->table, i16, 1, offsetof(scamp_obj_t, flags),
                                          i16, "FLAGS", NULL);
    }

    h = fitstable_get_header(scamp->table);
    qfits_header_add(h, "EXTNAME", "LDAC_OBJECTS", "", NULL);

    if (fitstable_write_header(scamp->table)) {
        ERROR("Failed to write scamp catalog object header.\n");
        return -1;
    }
    return 0;
}

int scamp_catalog_write_object(scamp_cat_t* scamp, const scamp_obj_t* obj) {
    assert(!scamp->ref);
    return fitstable_write_struct(scamp->table, obj);
}

int scamp_catalog_write_reference(scamp_cat_t* scamp, const scamp_ref_t* ref) {
    assert(scamp->ref);
    return fitstable_write_struct(scamp->table, ref);
}

int scamp_catalog_close(scamp_cat_t* scamp) {
    if (fitstable_fix_header(scamp->table) ||
        fitstable_close(scamp->table)) {
        ERROR("Failed to close scamp catalog");
        return -1;
    }
    free(scamp);
    return 0;
}

