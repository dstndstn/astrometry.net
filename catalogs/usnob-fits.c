/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "usnob-fits.h"
#include "fitsioutils.h"

// This is a naughty preprocessor function because it uses variables
// declared in the scope from which it is called.
#define ADDARR(ctype, ftype, col, units, member, arraysize)             \
    if (write) {                                                        \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(usnob_entry, member),      \
             ftype, col, units, TRUE);                                  \
    } else {                                                            \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(usnob_entry, member),      \
             any, col, units, TRUE);                                    \
    }

#define ADDCOL(ctype, ftype, col, units, member)        \
    ADDARR(ctype, ftype, col, units, member, 1)

static void add_columns(fitstable_t* tab, anbool write) {
    tfits_type any = fitscolumn_any_type();
    tfits_type d = fitscolumn_double_type();
    tfits_type f = fitscolumn_float_type();
    tfits_type u8 = fitscolumn_u8_type();
    tfits_type i16 = fitscolumn_i16_type();
    tfits_type i32 = fitscolumn_i32_type();
    tfits_type J = TFITS_BIN_TYPE_J;
    tfits_type u = fitscolumn_int_type();
    //tfits_type anbool = fitscolumn_bool_type();
    //tfits_type logical = fitscolumn_boolean_type();
    tfits_type bitfield = fitscolumn_bitfield_type();
    char* nil = " ";
    int ob;

    ADDCOL(d,  d,       "RA",                "deg",  ra);
    ADDCOL(d,  d,       "DEC",               "deg",  dec);

    ADDCOL(f,  f,       "SIGMA_RA",          "deg",  sigma_ra);
    ADDCOL(f,  f,       "SIGMA_DEC",         "deg",  sigma_dec);

    ADDCOL(f,  f,       "SIGMA_RA_FIT",      "deg",  sigma_ra_fit);
    ADDCOL(f,  f,       "SIGMA_DEC_FIT",     "deg",  sigma_dec_fit);

    ADDCOL(f,  f,       "PM_RA",             "arcsec/yr", pm_ra);
    ADDCOL(f,  f,       "PM_DEC",            "arcsec/yr", pm_dec);

    ADDCOL(f,  f,       "SIGMA_PM_RA",       "arcsec/yr", sigma_pm_ra);
    ADDCOL(f,  f,       "SIGMA_PM_DEC",      "arcsec/yr", sigma_pm_dec);

    ADDCOL(f,  f,       "PM_PROBABILITY",    nil, pm_prob);
    ADDCOL(f,  f,       "EPOCH",             "yr", epoch);

    ADDCOL(u8, u8,      "NUM_DETECTIONS",     nil, ndetections);

    ADDCOL(u,  J,       "USNOB_ID",           nil, usnob_id);

    for (ob=0; ob<5; ob++) {
        char field[256];
        sprintf(field, "MAGNITUDE_%i", ob);
        ADDCOL(f, f, field, "mag", obs[ob].mag);
        sprintf(field, "FIELD_%i", ob);
        ADDCOL(i16, i16, field, nil, obs[ob].field);
        sprintf(field, "SURVEY_%i", ob);
        ADDCOL(u8, u8, field, nil, obs[ob].survey);
        sprintf(field, "STAR_GALAXY_%i", ob);
        ADDCOL(u8, u8, field, nil, obs[ob].star_galaxy);
        sprintf(field, "XI_RESIDUAL_%i", ob);
        ADDCOL(f, f, field, "deg", obs[ob].xi_resid);
        sprintf(field, "ETA_RESIDUAL_%i", ob);
        ADDCOL(f, f, field, "deg", obs[ob].eta_resid);
        sprintf(field, "CALIBRATION_%i", ob);
        ADDCOL(u8, u8, field, nil, obs[ob].calibration);
        sprintf(field, "PMM_%i", ob);
        ADDCOL(i32, i32, field, nil, obs[ob].pmmscan);
    }

    ADDCOL(bitfield, bitfield, "FLAGS", nil, flags);

    // AN_DIFFRACTION_SPIKE is optional.
    fitstable_add_column_struct(tab, bitfield, 1,
                                offsetof(usnob_entry, an_diffraction_spike),
                                (write? bitfield : any),
                                "AN_DIFFRACTION_SPIKE", nil, FALSE);
}
#undef ADDCOL
#undef ADDARR

static int postprocess_read_structs(fitstable_t* table, void* struc,
                                    int stride, int offset, int N) {
    int i;
    usnob_entry* entries = struc;

    for (i=0; i<N; i++) {
        uint8_t flag = entries[i].flags;
        entries[i].diffraction_spike  = (flag >> 7) & 0x1;
        entries[i].motion_catalog     = (flag >> 6) & 0x1;
        entries[i].ys4                = (flag >> 5) & 0x1;
    }
    return 0;
}

int usnob_fits_remove_an_diffraction_spike_column(usnob_fits* usnob) {
    return fitstable_remove_column(usnob, "AN_DIFFRACTION_SPIKE");
}

usnob_entry* usnob_fits_read_entry(usnob_fits* cat) {
    return (usnob_entry*)fitstable_next_struct(cat);
}

int usnob_fits_read_entries(usnob_fits* cat, int offset,
                            int count, usnob_entry* entries) {
    return fitstable_read_structs(cat, entries, sizeof(usnob_entry), offset, count);
}

int usnob_fits_write_entry(usnob_fits* cat, usnob_entry* entry) {
    entry->flags =
        (entry->diffraction_spike  ? (1 << 7) : 0) |
        (entry->motion_catalog     ? (1 << 6) : 0) |
        (entry->ys4                ? (1 << 5) : 0);
    return fitstable_write_struct(cat, entry);
}

int usnob_fits_count_entries(usnob_fits* cat) {
    return fitstable_nrows(cat);
}

int usnob_fits_close(usnob_fits* usnob) {
    return fitstable_close(usnob);
}

usnob_fits* usnob_fits_open(char* fn) {
    usnob_fits* cat = NULL;
    cat = fitstable_open(fn);
    if (!cat)
        return NULL;
    add_columns(cat, FALSE);
    fitstable_use_buffered_reading(cat, sizeof(usnob_entry), 1000);
    cat->postprocess_read_structs = postprocess_read_structs;
    if (fitstable_read_extension(cat, 1)) {
        fprintf(stderr, "usnob-fits: table in extension 1 didn't contain the required columns.\n");
        fprintf(stderr, "  missing: ");
        fitstable_print_missing(cat, stderr);
        fprintf(stderr, "\n");
        usnob_fits_close(cat);
        return NULL;
    }
    return cat;
}

usnob_fits* usnob_fits_open_for_writing(char* fn) {
    usnob_fits* cat;
    qfits_header* hdr;
    cat = fitstable_open_for_writing(fn);
    if (!cat)
        return NULL;
    add_columns(cat, TRUE);
    hdr = fitstable_get_primary_header(cat);
    qfits_header_add(hdr, "USNOB", "T", "This is a USNO-B 1.0 catalog.", NULL);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_USNOB, "Astrometry.net file type", NULL);
    return cat;
}

int usnob_fits_write_headers(usnob_fits* cat) {
    if (fitstable_write_primary_header(cat))
        return -1;
    return fitstable_write_header(cat);
}

int usnob_fits_fix_headers(usnob_fits* cat) {
    if (fitstable_fix_primary_header(cat))
        return -1;
    return fitstable_fix_header(cat);
}

qfits_header* usnob_fits_get_header(usnob_fits* usnob) {
    return fitstable_get_primary_header(usnob);
}
