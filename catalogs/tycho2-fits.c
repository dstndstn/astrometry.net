/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "qfits_header.h"
#include "tycho2-fits.h"
#include "fitsioutils.h"

// This is a naughty preprocessor function because it uses variables
// declared in the scope from which it is called.
#define ADDARR(ctype, ftype, col, units, member, arraysize)             \
    if (write) {                                                        \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(tycho2_entry, member),     \
             ftype, col, units, TRUE);                                  \
    } else {                                                            \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(tycho2_entry, member),     \
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
    tfits_type I = TFITS_BIN_TYPE_I;
    tfits_type c = fitscolumn_char_type();
    tfits_type bitfield = fitscolumn_bitfield_type();
    char* nil = " ";

    ADDCOL(i16, I,      "TYC1",              nil,  tyc1);
    ADDCOL(i16, I,      "TYC2",              nil,  tyc2);
    ADDCOL(u8, u8,      "TYC3",              nil,  tyc3);

    ADDCOL(d,  d,       "RA",                "deg",  ra);
    ADDCOL(d,  d,       "DEC",               "deg",  dec);
    ADDCOL(d,  d,       "MEAN_RA",           "deg",  mean_ra);
    ADDCOL(d,  d,       "MEAN_DEC",          "deg",  mean_dec);

    ADDCOL(f,  f,       "SIGMA_RA",          "deg",  sigma_ra);
    ADDCOL(f,  f,       "SIGMA_DEC",         "deg",  sigma_dec);
    ADDCOL(f,  f,       "SIGMA_MEAN_RA",     "deg",  sigma_mean_ra);
    ADDCOL(f,  f,       "SIGMA_MEAN_DEC",    "deg",  sigma_mean_dec);

    ADDCOL(f,  f,       "PM_RA",             "arcsec/yr", pm_ra);
    ADDCOL(f,  f,       "PM_DEC",            "arcsyc/yr", pm_dec);

    ADDCOL(f,  f,       "SIGMA_PM_RA",       "arcsec/yr", sigma_pm_ra);
    ADDCOL(f,  f,       "SIGMA_PM_DEC",      "arcsyc/yr", sigma_pm_dec);

    ADDCOL(f,  f,       "EPOCH_RA",          "yr", epoch_ra);
    ADDCOL(f,  f,       "EPOCH_DEC",         "yr", epoch_dec);
    ADDCOL(f,  f,       "EPOCH_MEAN_RA",     "yr", epoch_mean_ra);
    ADDCOL(f,  f,       "EPOCH_MEAN_DEC",    "yr", epoch_mean_dec);

    ADDCOL(u8, u8,      "NOBSERVATIONS",     nil, nobs);

    ADDCOL(f,  f,       "GOODNESS_MEAN_RA",  nil, goodness_mean_ra);
    ADDCOL(f,  f,       "GOODNESS_MEAN_DEC", nil, goodness_mean_dec);
    ADDCOL(f,  f,       "GOODNESS_PM_RA",    nil, goodness_pm_ra);
    ADDCOL(f,  f,       "GOODNESS_PM_DEC",   nil, goodness_pm_dec);

    ADDCOL(f,  f,       "MAG_BT",            "mag", mag_BT);
    ADDCOL(f,  f,       "SIGMA_MAG_BT",      "mag", sigma_BT);
    ADDCOL(f,  f,       "MAG_VT",            "mag", mag_VT);
    ADDCOL(f,  f,       "SIGMA_MAG_VT",      "mag", sigma_VT);
    ADDCOL(f,  f,       "MAG_HP",            "mag", mag_HP);
    ADDCOL(f,  f,       "SIGMA_MAG_HP",      "mag", sigma_HP);

    ADDCOL(f,  f,       "PROX",              "deg", prox);
    ADDCOL(f,  f,       "CORRELATION",       nil, correlation);
    ADDCOL(i32,J,       "HIPPARCOS_ID",      nil, hipparcos_id);

    ADDARR(c,  c,       "CCDM",              nil, hip_ccdm, 3);

    if (write)
        fitstable_add_write_column(tab, bitfield, "FLAGS", nil);
}
#undef ADDCOL
#undef ADDARR

static int postprocess_read_structs(fitstable_t* table, void* struc,
                                    int stride, int offset, int N) {
    uint8_t* flags;
    int i;
    tycho2_fits* cat = table;
    tycho2_entry* entries = struc;

    // HACK?  Use staging area instead?
    flags = fitstable_read_column_offset(cat, "FLAGS", fitscolumn_u8_type(), offset, N);
                                         
    if (!flags)
        return -1;

    for (i=0; i<N; i++) {
        uint8_t flag = flags[i];
        entries[i].photo_center           = (flag >> 7) & 0x1;
        entries[i].no_motion              = (flag >> 6) & 0x1;
        entries[i].tycho1_star            = (flag >> 5) & 0x1;
        entries[i].double_star            = (flag >> 4) & 0x1;
        entries[i].photo_center_treatment = (flag >> 3) & 0x1;
        entries[i].hipparcos_star         = (flag >> 2) & 0x1;
    }
    free(flags);

    // Replace trailing spaces by \0.
    for (i=0; i<N; i++) {
        if (!entries[i].hip_ccdm[0])
            continue;
        if (entries[i].hip_ccdm[2] != ' ')
            continue;
        entries[i].hip_ccdm[2] = '\0';
        if (entries[i].hip_ccdm[1] != ' ')
            continue;
        entries[i].hip_ccdm[1] = '\0';
        if (entries[i].hip_ccdm[0] != ' ')
            continue;
        entries[i].hip_ccdm[0] = '\0';
    }
    return 0;
}

qfits_header* tycho2_fits_get_header(tycho2_fits* tycho2) {
    return fitstable_get_primary_header(tycho2);
}

tycho2_fits* tycho2_fits_open(char* fn) {
    tycho2_fits* cat = NULL;
    cat = fitstable_open(fn);
    if (!cat)
        return NULL;
    add_columns(cat, FALSE);
    fitstable_use_buffered_reading(cat, sizeof(tycho2_entry), 10000);
    cat->postprocess_read_structs = postprocess_read_structs;
    if (fitstable_read_extension(cat, 1)) {
        fprintf(stderr, "tycho2-fits: table in extension 1 didn't contain the required columns.\n");
        fprintf(stderr, "  missing: ");
        fitstable_print_missing(cat, stderr);
        fprintf(stderr, "\n");
        tycho2_fits_close(cat);
        return NULL;
    }
    return cat;
}

tycho2_fits* tycho2_fits_open_for_writing(char* fn) {
    tycho2_fits* cat;
    qfits_header* hdr;
    cat = fitstable_open_for_writing(fn);
    if (!cat)
        return NULL;
    add_columns(cat, TRUE);
    hdr = fitstable_get_primary_header(cat);
    qfits_header_add(hdr, "TYCHO_2", "T", "This is a Tycho-2 catalog.", NULL);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_TYCHO2, "Astrometry.net file type", NULL);
    return cat;
}

tycho2_entry* tycho2_fits_read_entry(tycho2_fits* cat) {
    return (tycho2_entry*)fitstable_next_struct(cat);
}

int tycho2_fits_read_entries(tycho2_fits* cat, int offset,
                             int count, tycho2_entry* entries) {
    return fitstable_read_structs(cat, entries, sizeof(tycho2_entry), offset, count);
}

int tycho2_fits_write_entry(tycho2_fits* cat, tycho2_entry* entry) {
    uint8_t flags;

    int rtn = fitstable_write_struct(cat, entry);
    if (rtn)
        return rtn;

    flags =
        (entry->photo_center           ? (1 << 7) : 0) |
        (entry->no_motion              ? (1 << 6) : 0) |
        (entry->tycho1_star            ? (1 << 5) : 0) |
        (entry->double_star            ? (1 << 4) : 0) |
        (entry->photo_center_treatment ? (1 << 3) : 0) |
        (entry->hipparcos_star         ? (1 << 2) : 0);
    // Can't just write_data() because write_struct() skips over
    //return fits_write_data_X(cat->fid, flags);
    return fitstable_write_one_column(cat, bl_size(cat->cols)-1,
                                      fitstable_nrows(cat)-1, 1, &flags, 0);
}

int tycho2_fits_count_entries(tycho2_fits* cat) {
    return fitstable_nrows(cat);
}

int tycho2_fits_close(tycho2_fits* cat) {
    return fitstable_close(cat);
}

int tycho2_fits_write_headers(tycho2_fits* cat) {
    if (fitstable_write_primary_header(cat))
        return -1;
    return fitstable_write_header(cat);
}

int tycho2_fits_fix_headers(tycho2_fits* cat) {
    if (fitstable_fix_primary_header(cat))
        return -1;
    return fitstable_fix_header(cat);
}

