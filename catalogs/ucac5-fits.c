/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

// Author: Vladimir Kouprianov, Skynet RTN, University of North Carolina at Chapel Hill

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "ucac5-fits.h"
#include "fitsioutils.h"

// This is a naughty preprocessor function because it uses variables
// declared in the scope from which it is called.
#define ADDARR(ctype, ftype, col, units, member, arraysize)             \
    if (write) {                                                        \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(ucac5_entry, member),      \
             ftype, col, units, TRUE);                                  \
    } else {                                                            \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(ucac5_entry, member),      \
             any, col, units, TRUE);                                    \
    }

#define ADDCOL(ctype, ftype, col, units, member)        \
    ADDARR(ctype, ftype, col, units, member, 1)

static void add_columns(fitstable_t* tab, anbool write, anbool full) {
    tfits_type any = fitscolumn_any_type();
    tfits_type d = fitscolumn_double_type();
    tfits_type f = fitscolumn_float_type();
    tfits_type u8 = fitscolumn_u8_type();
    tfits_type i64 = fitscolumn_i64_type();
    char* nil = " ";

    if (full) {
        ADDCOL(i64, i64, "ID", nil, srcid);
    }

    ADDCOL(d, d, "RA",  "deg", ra);
    ADDCOL(d, d, "DEC", "deg", dec);

    if (full) {
        ADDCOL(d,  d,  "RA_GAIA",      "deg",    rag);
        ADDCOL(d,  d,  "DEC_GAIA",     "deg",    dcg);
        ADDCOL(f,  f,  "RA_GAIA_ERR",  "deg",    erg);
        ADDCOL(f,  f,  "DEC_GAIA_ERR", "deg",    erg);
        ADDCOL(u8, u8, "FLAGS",        nil,      flg);
        ADDCOL(u8, u8, "NUM_POS",      nil,      nu);
        ADDCOL(f,  f,  "UCAC_EPOCH",   "yr",     epu);
        ADDCOL(d,  d,  "RA_UCAC",      "deg",    ira);
        ADDCOL(d,  d,  "DEC_UCAC",     "deg",    idc);
        ADDCOL(f,  f,  "PM_RA",        "deg/yr", pmur);
        ADDCOL(f,  f,  "PM_DEC",       "deg/yr", pmud);
        ADDCOL(f,  f,  "PM_RA_ERR",    "deg/yr", pmer);
        ADDCOL(f,  f,  "PM_DEC_ERR",   "deg/yr", pmed);
        ADDCOL(f,  f,  "GMAG",         "mag",    gmag);
    }

    ADDCOL(f, f, "MAG", "mag", umag);

    if (full) {
        ADDCOL(f, f, "RMAG", "mag", rmag);
        ADDCOL(f, f, "JMAG", "mag", jmag);
        ADDCOL(f, f, "HMAG", "mag", hmag);
        ADDCOL(f, f, "KMAG", "mag", kmag);
    }
}
#undef ADDCOL
#undef ADDARR

ucac5_entry* ucac5_fits_read_entry(ucac5_fits* cat) {
    return (ucac5_entry*)fitstable_next_struct(cat);
}

int ucac5_fits_read_entries(ucac5_fits* cat, int offset,
                            int count, ucac5_entry* entries) {
    return fitstable_read_structs(cat, entries, sizeof(ucac5_entry), offset, count);
}

int ucac5_fits_write_entry(ucac5_fits* cat, ucac5_entry* entry) {
    return fitstable_write_struct(cat, entry);
}

int ucac5_fits_count_entries(ucac5_fits* cat) {
    return fitstable_nrows(cat);
}

int ucac5_fits_close(ucac5_fits* ucac5) {
    return fitstable_close(ucac5);
}

ucac5_fits* ucac5_fits_open(char* fn, anbool full) {
    ucac5_fits* cat = NULL;
    cat = fitstable_open(fn);
    if (!cat)
        return NULL;
    add_columns(cat, FALSE, full);
    fitstable_use_buffered_reading(cat, sizeof(ucac5_entry), 1000);
    if (fitstable_read_extension(cat, 1)) {
        fprintf(stderr, "ucac5-fits: table in extension 1 didn't contain the required columns.\n");
        fprintf(stderr, "  missing: ");
        fitstable_print_missing(cat, stderr);
        fprintf(stderr, "\n");
        ucac5_fits_close(cat);
        return NULL;
    }
    return cat;
}

ucac5_fits* ucac5_fits_open_for_writing(char* fn, anbool full) {
    ucac5_fits* cat;
    qfits_header* hdr;
    cat = fitstable_open_for_writing(fn);
    if (!cat)
        return NULL;
    add_columns(cat, TRUE, full);
    hdr = fitstable_get_primary_header(cat);
    qfits_header_add(hdr, "UCAC5", "T", "This is a UCAC5 catalog.", NULL);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_UCAC5, "Astrometry.net file type", NULL);
    return cat;
}

int ucac5_fits_write_headers(ucac5_fits* cat) {
    if (fitstable_write_primary_header(cat))
        return -1;
    return fitstable_write_header(cat);
}

int ucac5_fits_fix_headers(ucac5_fits* cat) {
    if (fitstable_fix_primary_header(cat))
        return -1;
    return fitstable_fix_header(cat);
}

