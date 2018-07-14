/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "ucac3-fits.h"
#include "fitsioutils.h"

// This is a naughty preprocessor function because it uses variables
// declared in the scope from which it is called.
#define ADDARR(ctype, ftype, col, units, member, arraysize)             \
    if (write) {                                                        \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(ucac3_entry, member),      \
             ftype, col, units, TRUE);                                  \
    } else {                                                            \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(ucac3_entry, member),      \
             any, col, units, TRUE);                                    \
    }

#define ADDCOL(ctype, ftype, col, units, member)        \
    ADDARR(ctype, ftype, col, units, member, 1)

static void add_columns(fitstable_t* tab, anbool write) {
    tfits_type any = fitscolumn_any_type();
    tfits_type d = fitscolumn_double_type();
    tfits_type f = fitscolumn_float_type();
    tfits_type u8 = fitscolumn_u8_type();
    tfits_type i32 = fitscolumn_i32_type();
    tfits_type J = TFITS_BIN_TYPE_J;
    char* nil = " ";

    ADDCOL(d,  d,   "RA",                "deg",  ra);
    ADDCOL(d,  d,   "DEC",               "deg",  dec);
    ADDCOL(f,  f,   "SIGMA_RA",          "deg",  sigma_ra);
    ADDCOL(f,  f,   "SIGMA_DEC",         "deg",  sigma_dec);
    ADDCOL(f,  f,   "PM_RA",             "arcsec/yr", pm_ra);
    ADDCOL(f,  f,   "PM_DEC",            "arcsyc/yr", pm_dec);
    ADDCOL(f,  f,   "SIGMA_PM_RA",       "arcsec/yr", sigma_pm_ra);
    ADDCOL(f,  f,   "SIGMA_PM_DEC",      "arcsyc/yr", sigma_pm_dec);
    ADDCOL(f,  f,   "EPOCH_RA",          "yr", epoch_ra);
    ADDCOL(f,  f,   "EPOCH_DEC",         "yr", epoch_dec);
    ADDCOL(f,  f,   "MAG",               "mag", mag);
    ADDCOL(f,  f,   "MAG_ERR",           "mag", mag_err);
    ADDCOL(f,  f,   "APMAG",             "mag", apmag);
    ADDCOL(f,  f,   "JMAG",              "mag", jmag);
    ADDCOL(f,  f,   "HMAG",              "mag", hmag);
    ADDCOL(f,  f,   "KMAG",              "mag", kmag);
    ADDCOL(f,  f,   "JMAG_ERR",          "mag", jmag_err);
    ADDCOL(f,  f,   "HMAG_ERR",          "mag", hmag_err);
    ADDCOL(f,  f,   "KMAG_ERR",          "mag", kmag_err);
    ADDCOL(f,  f,   "BMAG",              "mag", bmag);
    ADDCOL(f,  f,   "R2MAG",             "mag", r2mag);
    ADDCOL(f,  f,   "IMAG",              "mag", imag);

    ADDCOL(u8, u8,  "JFLAGS",             nil, twomass_jflags);
    ADDCOL(u8, u8,  "HFLAGS",             nil, twomass_hflags);
    ADDCOL(u8, u8,  "KFLAGS",             nil, twomass_kflags);
    ADDCOL(u8, u8,  "CLBL",               nil, clbl);
    ADDCOL(u8, u8,  "BQUALITY",           nil, bquality);
    ADDCOL(u8, u8,  "R2QUALITY",          nil, r2quality);
    ADDCOL(u8, u8,  "IQUALITY",           nil, iquality);

    ADDCOL(u8, u8,  "OBJTYPE",            nil, objtype);
    ADDCOL(u8, u8,  "DOUBLESTAR",         nil, doublestar);
    ADDCOL(u8, u8,  "NAVAIL",             nil, navail);
    ADDCOL(u8, u8,  "NUSED",              nil, nused);
    ADDCOL(u8, u8,  "NPM",                nil, npm);
    ADDCOL(u8, u8,  "NMATCH",             nil, nmatch);

    ADDARR(u8, u8,  "MATCHFLAGS",         nil, matchflags, 10);
	
    ADDCOL(u8, u8,  "YALE_CFLAG",        nil, yale_cflag);
    ADDCOL(u8, u8,  "YALE_GFLAG",        nil, yale_gflag);
    ADDCOL(u8, u8,  "LEDA_GFLAG",        nil, leda_flag);
    ADDCOL(u8, u8,  "TMXS_GFLAG",        nil, twomass_extsource_flag);

    ADDCOL(i32,J,   "TWOMASS_ID",         nil, twomass_id);
    ADDCOL(i32,J,   "MPOS",               nil, mpos);
}
#undef ADDCOL
#undef ADDARR

ucac3_entry* ucac3_fits_read_entry(ucac3_fits* cat) {
    return (ucac3_entry*)fitstable_next_struct(cat);
}

int ucac3_fits_read_entries(ucac3_fits* cat, int offset,
                            int count, ucac3_entry* entries) {
    return fitstable_read_structs(cat, entries, sizeof(ucac3_entry), offset, count);
}

int ucac3_fits_write_entry(ucac3_fits* cat, ucac3_entry* entry) {
    /*
     entry->flags[0] =
     (entry->usnob_fail        ? (1 << 7) : 0) |
     (entry->twomass_fail      ? (1 << 6) : 0) |
     (entry->tycho_astrometry  ? (1 << 5) : 0) |
     (entry->alt_radec         ? (1 << 4) : 0) |
     (entry->alt_ucac          ? (1 << 3) : 0) |
     (entry->alt_tycho         ? (1 << 2) : 0) |
     (entry->blue_o            ? (1 << 1) : 0) |
     (entry->red_e             ? (1 << 0) : 0);
     entry->flags[1] = 
     (entry->twomass_only      ? (1 << 7) : 0) |
     (entry->hipp_astrometry   ? (1 << 6) : 0) |
     (entry->diffraction       ? (1 << 5) : 0) |
     (entry->confusion         ? (1 << 4) : 0) |
     (entry->bright_confusion  ? (1 << 3) : 0) |
     (entry->bright_artifact   ? (1 << 2) : 0) |
     (entry->standard          ? (1 << 1) : 0);
     */
    return fitstable_write_struct(cat, entry);
}

int ucac3_fits_count_entries(ucac3_fits* cat) {
    return fitstable_nrows(cat);
}

int ucac3_fits_close(ucac3_fits* ucac3) {
    return fitstable_close(ucac3);
}

ucac3_fits* ucac3_fits_open(char* fn) {
    ucac3_fits* cat = NULL;
    cat = fitstable_open(fn);
    if (!cat)
        return NULL;
    add_columns(cat, FALSE);
    fitstable_use_buffered_reading(cat, sizeof(ucac3_entry), 1000);
    if (fitstable_read_extension(cat, 1)) {
        fprintf(stderr, "ucac3-fits: table in extension 1 didn't contain the required columns.\n");
        fprintf(stderr, "  missing: ");
        fitstable_print_missing(cat, stderr);
        fprintf(stderr, "\n");
        ucac3_fits_close(cat);
        return NULL;
    }
    return cat;
}

ucac3_fits* ucac3_fits_open_for_writing(char* fn) {
    ucac3_fits* cat;
    qfits_header* hdr;
    cat = fitstable_open_for_writing(fn);
    if (!cat)
        return NULL;
    add_columns(cat, TRUE);
    hdr = fitstable_get_primary_header(cat);
    qfits_header_add(hdr, "UCAC3", "T", "This is a UCAC3 catalog.", NULL);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_UCAC3, "Astrometry.net file type", NULL);
    return cat;
}

int ucac3_fits_write_headers(ucac3_fits* cat) {
    if (fitstable_write_primary_header(cat))
        return -1;
    return fitstable_write_header(cat);
}

int ucac3_fits_fix_headers(ucac3_fits* cat) {
    if (fitstable_fix_primary_header(cat))
        return -1;
    return fitstable_fix_header(cat);
}

