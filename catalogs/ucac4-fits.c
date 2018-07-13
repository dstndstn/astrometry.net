/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "ucac4-fits.h"
#include "fitsioutils.h"

// This is a naughty preprocessor function because it uses variables
// declared in the scope from which it is called.
#define ADDARR(ctype, ftype, col, units, member, arraysize)             \
    if (write) {                                                        \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(ucac4_entry, member),      \
             ftype, col, units, TRUE);                                  \
    } else {                                                            \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(ucac4_entry, member),      \
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
    ADDCOL(f,  f,   "SIG_RA",            "deg",  sigma_ra);
    ADDCOL(f,  f,   "SIG_DEC",           "deg",  sigma_dec);
    ADDCOL(f,  f,   "PM_RA",             "arcsec/yr", pm_rac);
    ADDCOL(f,  f,   "PM_DEC",            "arcsyc/yr", pm_dec);
    ADDCOL(f,  f,   "SIG_PM_R",          "arcsec/yr", sigma_pm_ra);
    ADDCOL(f,  f,   "SIG_PM_D",          "arcsyc/yr", sigma_pm_dec);
    ADDCOL(f,  f,   "EPOCH_RA",          "yr", epoch_ra);
    ADDCOL(f,  f,   "EPOCH_DE",          "yr", epoch_dec);
    ADDCOL(f,  f,   "MAG",               "mag", mag);
    ADDCOL(f,  f,   "SIGMAG",            "mag", mag_err);
    ADDCOL(f,  f,   "APMAG",             "mag", apmag);
    ADDCOL(f,  f,   "JMAG",              "mag", jmag);
    ADDCOL(f,  f,   "HMAG",              "mag", hmag);
    ADDCOL(f,  f,   "KMAG",              "mag", kmag);
    ADDCOL(f,  f,   "JMAG_ERR",          "mag", jmag_err);
    ADDCOL(f,  f,   "HMAG_ERR",          "mag", hmag_err);
    ADDCOL(f,  f,   "KMAG_ERR",          "mag", kmag_err);
    ADDCOL(f,  f,   "BMAG",              "mag", Bmag);
    ADDCOL(f,  f,   "VMAG",              "mag", Vmag);
    ADDCOL(f,  f,   "GMAG",              "mag", gmag);
    ADDCOL(f,  f,   "RMAG",              "mag", rmag);
    ADDCOL(f,  f,   "IMAG",              "mag", imag);
    ADDCOL(f,  f,   "BMAG_ERR",          "mag", Bmag_err);
    ADDCOL(f,  f,   "VMAG_ERR",          "mag", Vmag_err);
    ADDCOL(f,  f,   "GMAG_ERR",          "mag", gmag_err);
    ADDCOL(f,  f,   "RMAG_ERR",          "mag", rmag_err);
    ADDCOL(f,  f,   "IMAG_ERR",          "mag", imag_err);

    ADDCOL(u8, u8,  "ICQ_FL_J",           nil, twomass_jflags);
    ADDCOL(u8, u8,  "ICQ_FL_H",           nil, twomass_hflags);
    ADDCOL(u8, u8,  "ICQ_FL_K",           nil, twomass_kflags);

    ADDCOL(u8, u8,  "OBJTYPE",            nil, objtype);
    ADDCOL(u8, u8,  "CDF",                nil, doublestar);
    ADDCOL(u8, u8,  "NA_ONE",             nil, navail);
    ADDCOL(u8, u8,  "NU_ONE",             nil, nused);
    ADDCOL(u8, u8,  "CU_ONE",             nil, nmatch);
	
    ADDCOL(u8, u8,  "GC_FLG",             nil, yale_gc_flag);
    ADDCOL(u8, u8,  "LEDA_FLG",           nil, leda_flag);
    ADDCOL(u8, u8,  "TMXS_FLG",           nil, twomass_extsource_flag);

    ADDCOL(i32,J,   "ICF",                nil, catalog_flags);
    ADDCOL(i32,J,   "X_TWO_M",            nil, twomass_id);
    ADDCOL(i32,J,   "RNM",                nil, mpos);
    ADDCOL(i32,J,   "ZN_TWO",             nil, ucac2_zone);
    ADDCOL(i32,J,   "RN_TWO",             nil, ucac2_number);
}
#undef ADDCOL
#undef ADDARR

ucac4_entry* ucac4_fits_read_entry(ucac4_fits* cat) {
    return (ucac4_entry*)fitstable_next_struct(cat);
}

int ucac4_fits_read_entries(ucac4_fits* cat, int offset,
                            int count, ucac4_entry* entries) {
    return fitstable_read_structs(cat, entries, sizeof(ucac4_entry), offset, count);
}

int ucac4_fits_write_entry(ucac4_fits* cat, ucac4_entry* entry) {
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

int ucac4_fits_count_entries(ucac4_fits* cat) {
    return fitstable_nrows(cat);
}

int ucac4_fits_close(ucac4_fits* ucac4) {
    return fitstable_close(ucac4);
}

ucac4_fits* ucac4_fits_open(char* fn) {
    ucac4_fits* cat = NULL;
    cat = fitstable_open(fn);
    if (!cat)
        return NULL;
    add_columns(cat, FALSE);
    fitstable_use_buffered_reading(cat, sizeof(ucac4_entry), 1000);
    if (fitstable_read_extension(cat, 1)) {
        fprintf(stderr, "ucac4-fits: table in extension 1 didn't contain the required columns.\n");
        fprintf(stderr, "  missing: ");
        fitstable_print_missing(cat, stderr);
        fprintf(stderr, "\n");
        ucac4_fits_close(cat);
        return NULL;
    }
    return cat;
}

ucac4_fits* ucac4_fits_open_for_writing(char* fn) {
    ucac4_fits* cat;
    qfits_header* hdr;
    cat = fitstable_open_for_writing(fn);
    if (!cat)
        return NULL;
    add_columns(cat, TRUE);
    hdr = fitstable_get_primary_header(cat);
    qfits_header_add(hdr, "UCAC4", "T", "This is a UCAC4 catalog.", NULL);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_UCAC4, "Astrometry.net file type", NULL);
    return cat;
}

int ucac4_fits_write_headers(ucac4_fits* cat) {
    if (fitstable_write_primary_header(cat))
        return -1;
    return fitstable_write_header(cat);
}

int ucac4_fits_fix_headers(ucac4_fits* cat) {
    if (fitstable_fix_primary_header(cat))
        return -1;
    return fitstable_fix_header(cat);
}

