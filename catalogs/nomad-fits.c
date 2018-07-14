/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "nomad-fits.h"
#include "fitsioutils.h"

// This is a naughty preprocessor function because it uses variables
// declared in the scope from which it is called.
#define ADDARR(ctype, ftype, col, units, member, arraysize)             \
    if (write) {                                                        \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(nomad_entry, member),      \
             ftype, col, units, TRUE);                                  \
    } else {                                                            \
        fitstable_add_column_struct                                     \
            (tab, ctype, arraysize, offsetof(nomad_entry, member),      \
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
    tfits_type bitfield = fitscolumn_bitfield_type();
    char* nil = " ";

    ADDCOL(d,  d,   "RA",                "deg",  ra);
    ADDCOL(d,  d,   "DEC",               "deg",  dec);
    ADDCOL(f,  f,   "SIGMA_RACOSDEC",    "deg",  sigma_racosdec);
    ADDCOL(f,  f,   "SIGMA_DEC",         "deg",  sigma_dec);
    ADDCOL(f,  f,   "PM_RACOSDEC",       "arcsec/yr", pm_racosdec);
    ADDCOL(f,  f,   "PM_DEC",            "arcsyc/yr", pm_dec);
    ADDCOL(f,  f,   "SIGMA_PM_RACOSDEC", "arcsec/yr", sigma_pm_racosdec);
    ADDCOL(f,  f,   "SIGMA_PM_DEC",      "arcsyc/yr", sigma_pm_dec);
    ADDCOL(f,  f,   "EPOCH_RA",          "yr", epoch_ra);
    ADDCOL(f,  f,   "EPOCH_DEC",         "yr", epoch_dec);
    ADDCOL(f,  f,   "MAG_B",             "mag", mag_B);
    ADDCOL(f,  f,   "MAG_V",             "mag", mag_V);
    ADDCOL(f,  f,   "MAG_R",             "mag", mag_R);
    ADDCOL(f,  f,   "MAG_J",             "mag", mag_J);
    ADDCOL(f,  f,   "MAG_H",             "mag", mag_H);
    ADDCOL(f,  f,   "MAG_K",             "mag", mag_K);
    ADDCOL(i32,J,   "USNOB_ID",           nil, usnob_id);
    ADDCOL(i32,J,   "TWOMASS_ID",         nil, twomass_id);
    ADDCOL(i32,J,   "YB6_ID",             nil, yb6_id);
    ADDCOL(i32,J,   "UCAC2_ID",           nil, ucac2_id);
    ADDCOL(i32,J,   "TYCHO2_ID",          nil, tycho2_id);
    ADDCOL(u8, u8,  "ASTROMETRY_SRC",     nil, astrometry_src);
    ADDCOL(u8, u8,  "BLUE_SRC",           nil, blue_src);
    ADDCOL(u8, u8,  "VISUAL_SRC",         nil, visual_src);
    ADDCOL(u8, u8,  "RED_SRC",            nil, red_src);
    ADDCOL(i32, J,  "NOMAD_ID",           nil, nomad_id);
    ADDARR(bitfield, bitfield, "FLAGS",   nil, flags, 2);
}
#undef ADDCOL
#undef ADDARR

static int postprocess_read_structs(fitstable_t* table, void* struc,
                                    int stride, int offset, int N) {
    int i;
    nomad_entry* entries = struc;
    for (i=0; i<N; i++) {
        uint8_t flag;
        flag = entries[i].flags[0];
        entries[i].usnob_fail         = (flag >> 7) & 0x1;
        entries[i].twomass_fail       = (flag >> 6) & 0x1;
        entries[i].tycho_astrometry   = (flag >> 5) & 0x1;
        entries[i].alt_radec          = (flag >> 4) & 0x1;
        entries[i].alt_ucac           = (flag >> 3) & 0x1;
        entries[i].alt_tycho          = (flag >> 2) & 0x1;
        entries[i].blue_o             = (flag >> 1) & 0x1;
        entries[i].red_e              = (flag >> 0) & 0x1;
        flag = entries[i].flags[1];
        entries[i].twomass_only       = (flag >> 7) & 0x1;
        entries[i].hipp_astrometry    = (flag >> 6) & 0x1;
        entries[i].diffraction        = (flag >> 5) & 0x1;
        entries[i].confusion          = (flag >> 4) & 0x1;
        entries[i].bright_confusion   = (flag >> 3) & 0x1;
        entries[i].bright_artifact    = (flag >> 2) & 0x1;
        entries[i].standard           = (flag >> 1) & 0x1;
    }
    return 0;
}

nomad_entry* nomad_fits_read_entry(nomad_fits* cat) {
    return (nomad_entry*)fitstable_next_struct(cat);
}

int nomad_fits_read_entries(nomad_fits* cat, int offset,
                            int count, nomad_entry* entries) {
    return fitstable_read_structs(cat, entries, sizeof(nomad_entry), offset, count);
}

int nomad_fits_write_entry(nomad_fits* cat, nomad_entry* entry) {
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
    return fitstable_write_struct(cat, entry);
}

int nomad_fits_count_entries(nomad_fits* cat) {
    return fitstable_nrows(cat);
}

int nomad_fits_close(nomad_fits* nomad) {
    return fitstable_close(nomad);
}

nomad_fits* nomad_fits_open(char* fn) {
    nomad_fits* cat = NULL;
    cat = fitstable_open(fn);
    if (!cat)
        return NULL;
    add_columns(cat, FALSE);
    fitstable_use_buffered_reading(cat, sizeof(nomad_entry), 1000);
    cat->postprocess_read_structs = postprocess_read_structs;
    if (fitstable_read_extension(cat, 1)) {
        fprintf(stderr, "nomad-fits: table in extension 1 didn't contain the required columns.\n");
        fprintf(stderr, "  missing: ");
        fitstable_print_missing(cat, stderr);
        fprintf(stderr, "\n");
        nomad_fits_close(cat);
        return NULL;
    }
    return cat;
}

nomad_fits* nomad_fits_open_for_writing(char* fn) {
    nomad_fits* cat;
    qfits_header* hdr;
    cat = fitstable_open_for_writing(fn);
    if (!cat)
        return NULL;
    add_columns(cat, TRUE);
    hdr = fitstable_get_primary_header(cat);
    qfits_header_add(hdr, "NOMAD", "T", "This is a NOMAD 1.0 catalog.", NULL);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_NOMAD, "Astrometry.net file type", NULL);
    qfits_header_add(hdr, "COMMENT", "The FLAGS variable is composed of 15 boolean values packed into 2 bytes.", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "  Byte 0:", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x80: UBBIT / usnob_fail", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x40: TMBIT / twomass_fail", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x20: TYBIT / tycho_astrometry", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x10: XRBIT / alt_radec", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x08: IUCBIT / alt_ucac", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x04: ITYBIT / alt_tycho", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x02: OMAGBIT / blue_o", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x01: EMAGBIT / red_e", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "  Byte 1:", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x80: TMONLY / twomass_only", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x40: HIPAST / hipp_astrometry", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x20: SPIKE / diffraction", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x10: TYCONF / confusion", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x08: BSCONF / bright_confusion", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x04: BSART / bright_artifact", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x02: USEME / standard", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "    0x01: unused", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "  Note that the ITMBIT and EXCAT bits were not set for any entry in the ", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "  released NOMAD catalog, so were not included here.", NULL, NULL);
    return cat;
}

int nomad_fits_write_headers(nomad_fits* cat) {
    if (fitstable_write_primary_header(cat))
        return -1;
    return fitstable_write_header(cat);
}

int nomad_fits_fix_headers(nomad_fits* cat) {
    if (fitstable_fix_primary_header(cat))
        return -1;
    return fitstable_fix_header(cat);
}

