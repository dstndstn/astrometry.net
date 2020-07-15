/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "matchfile.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "sip.h"
#include "mathutil.h"

// This is a naughty preprocessor function because it uses variables
// declared in the calling scope.
#define ADDARR(ctype, ftype, col, units, member, arraysize)     \
    if (write) {                                                \
        fitstable_add_column_struct                             \
            (tab, ctype, arraysize, offsetof(MatchObj, member), \
             ftype, col, units, TRUE);                          \
    } else {                                                    \
        fitstable_add_column_struct                             \
            (tab, ctype, arraysize, offsetof(MatchObj, member), \
             any, col, units, FALSE);                           \
    }

//TRUE);

#define ADDCOL(ctype, ftype, col, units, member)        \
    ADDARR(ctype, ftype, col, units, member, 1)

static void add_columns(fitstable_t* tab, anbool write) {
    tfits_type any = fitscolumn_any_type();
    tfits_type d = fitscolumn_double_type();
    tfits_type f = fitscolumn_float_type();
    tfits_type u8 = fitscolumn_u8_type();
    tfits_type i16 = fitscolumn_i16_type();
    tfits_type i32 = fitscolumn_i32_type();
    tfits_type i64 = fitscolumn_i64_type();
    tfits_type i = fitscolumn_int_type();
    tfits_type logical = fitscolumn_boolean_type();
    tfits_type b = fitscolumn_bool_type();
    tfits_type c = fitscolumn_char_type();
    char* nil = " ";
    MatchObj mo;

    ADDCOL(i,  i32, "QUAD",            nil, quadno);
    ADDCOL(u8, u8,  "DIMQUADS",        nil, dimquads);
    ADDARR(i,  i32, "STARS",           nil, star, DQMAX);
    ADDARR(i,  i32, "FIELDOBJS",       nil, field, DQMAX);
    ADDARR(i64,i64, "IDS",             nil, ids, DQMAX);
    ADDCOL(f,  f,   "CODEERR",         nil, code_err);
    ADDARR(d,  d,   "QUADPDI",         nil, quadpix, 2*DQMAX);
    ADDARR(d,  d,   "QUADPIX",          nil, quadpix_orig, 2*DQMAX);
    ADDARR(d,  d,   "QUADXYZ",         nil, quadxyz, 3*DQMAX);
    ADDARR(d,  d,   "CENTERXYZ",       nil, center, 3);
    ADDCOL(d,  d,   "RADIUS",          "DEG", radius_deg);
    ADDCOL(i,  i32,   "NMATCH",          nil, nmatch);
    ADDCOL(i  ,i32,   "NDISTRACT",       nil, ndistractor);
    ADDCOL(i  ,i32,   "NCONFLICT",       nil, nconflict);
    ADDCOL(i  ,i32,   "NFIELD",          nil, nfield);
    ADDCOL(i  ,i32,   "NINDEX",          nil, nindex);
    ADDCOL(i  ,i32,   "NAGREE",          nil, nagree);
    //ADDCOL(i16,i16, "BESTI",           nil, besti);
    ADDARR(d,  d,   "CRVAL",           nil, wcstan.crval, 2);
    ADDARR(d,  d,   "CRPIX",           nil, wcstan.crpix, 2);
    ADDARR(d,  d,   "CD",              nil, wcstan.cd, 4);
    ADDCOL(b, logical, "WCS_VALID",    nil, wcs_valid);
    ADDCOL(i,i32,   "FIELDNUM",        nil, fieldnum);
    ADDCOL(i,i32,   "FIELDID",         nil, fieldfile);
    ADDCOL(i16,i16, "INDEXID",         nil, indexid);
    ADDCOL(i16,i16, "HEALPIX",         nil, healpix);
    ADDCOL(i16,i16, "HPNSIDE",         nil, hpnside);
    ADDARR(c,  c,   "FIELDNAME",       nil, fieldname, sizeof(mo.fieldname)-1);
    ADDCOL(b, logical, "PARITY",       nil, parity);
    ADDCOL(i,i32,   "QTRIED",          nil, quads_tried);
    ADDCOL(i,i32,   "QMATCHED",        nil, quads_matched);
    ADDCOL(i,i32,   "QSCALEOK",        nil, quads_scaleok);
    ADDCOL(i16,i16, "QPEERS",          nil, quad_npeers);
    ADDCOL(i,i32,   "NVERIFIED",       nil, nverified);
    ADDCOL(f,  f,   "TIMEUSED",        "s", timeused);
    ADDCOL(f,  f,   "LOGODDS",         nil, logodds);
    ADDCOL(f,  f,   "WORSTLOGODDS",    nil, worstlogodds);
}
#undef ADDCOL
#undef ADDARR

static int postprocess_read_structs(fitstable_t* table, void* struc,
                                    int stride, int offset, int N) {
    MatchObj* mo = struc;
    int i;
    for (i=0; i<N; i++)
        matchobj_compute_derived(mo + i);
    return 0;
}

MatchObj* matchfile_read_match(matchfile* mf) {
    return (MatchObj*)fitstable_next_struct(mf);
}

int matchfile_pushback_match(matchfile* m) {
    return fitstable_pushback(m);
}

int matchfile_read_matches(matchfile* mf, MatchObj* entries,
                           int offset, int count) {
    return fitstable_read_structs(mf, entries, sizeof(MatchObj), offset, count);
}

int matchfile_write_match(matchfile* mf, MatchObj* entry) {
    return fitstable_write_struct(mf, entry);
}

int matchfile_count(matchfile* mf) {
    return fitstable_nrows(mf);
}

int matchfile_close(matchfile* nomad) {
    return fitstable_close(nomad);
}

matchfile* matchfile_open(const char* fn) {
    matchfile* mf = NULL;
    mf = fitstable_open(fn);
    if (!mf)
        return NULL;
    add_columns(mf, FALSE);
    fitstable_use_buffered_reading(mf, sizeof(MatchObj), 1000);
    mf->postprocess_read_structs = postprocess_read_structs;
    if (fitstable_read_extension(mf, 1)) {
        fprintf(stderr, "matchfile: table in extension 1 didn't contain the required columns.\n");
        fprintf(stderr, "  missing: ");
        fitstable_print_missing(mf, stderr);
        fprintf(stderr, "\n");
        matchfile_close(mf);
        return NULL;
    }
    return mf;
}

matchfile* matchfile_open_for_writing(char* fn) {
    matchfile* mf;
    qfits_header* hdr;
    mf = fitstable_open_for_writing(fn);
    if (!mf)
        return NULL;
    add_columns(mf, TRUE);
    hdr = fitstable_get_primary_header(mf);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_MATCH, "Astrometry.net file type", NULL);
    return mf;
}

int matchfile_write_headers(matchfile* mf) {
    if (fitstable_write_primary_header(mf))
        return -1;
    return fitstable_write_header(mf);
}

int matchfile_fix_headers(matchfile* mf) {
    if (fitstable_fix_primary_header(mf))
        return -1;
    return fitstable_fix_header(mf);
}

pl* matchfile_get_matches_for_field(matchfile* mf, int field) {
    pl* list = pl_new(256);
    for (;;) {
        MatchObj* mo = matchfile_read_match(mf);
        MatchObj* copy;
        if (!mo) break;
        if (mo->fieldnum != field) {
            // push back the newly-read entry...
            matchfile_pushback_match(mf);
            break;
        }
        copy = malloc(sizeof(MatchObj));
        memcpy(copy, mo, sizeof(MatchObj));
        pl_append(list, copy);
    }
    return list;
}

