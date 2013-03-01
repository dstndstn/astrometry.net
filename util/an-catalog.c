/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <stdio.h>

#include <assert.h>
#include <stddef.h>
#include <errno.h>
#include <string.h>

#include "an-catalog.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "os-features.h"
#include "errors.h"

// This is a naughty preprocessor function because it uses variables
// declared in the scope from which it is called.
#define ADDCOL(ctype, ftype, col, units, member)                        \
    if (write) {                                                        \
        fitstable_add_column_struct                                     \
            (tab, ctype, 1, offsetof(an_entry, member),                 \
             ftype, col, units, TRUE);                                  \
    } else {                                                            \
        fitstable_add_column_struct                                     \
            (tab, ctype, 1, offsetof(an_entry, member),                 \
             any, col, units, TRUE);                                    \
    }

static void add_columns(fitstable_t* tab, anbool write) {
    tfits_type any = fitscolumn_any_type();
    tfits_type d = fitscolumn_double_type();
    tfits_type f = fitscolumn_float_type();
    tfits_type u8 = fitscolumn_u8_type();
    tfits_type i = fitscolumn_int_type();
    tfits_type i64 = fitscolumn_i64_type();
    tfits_type c = fitscolumn_char_type();
    int ob;
    char* nil = " ";

    ADDCOL(d, d, "RA",  "deg", ra);
    ADDCOL(d, d, "DEC", "deg", dec);
    ADDCOL(f, f, "SIGMA_RA",  "deg", sigma_ra);
    ADDCOL(f, f, "SIGMA_DEC", "deg", sigma_dec);
    ADDCOL(f, f, "MOTION_RA",  "arcsec/yr", motion_ra);
    ADDCOL(f, f, "MOTION_DEC", "arcsec/yr", motion_dec);
    ADDCOL(f, f, "SIGMA_MOTION_RA",  "arcsec/yr", sigma_motion_ra);
    ADDCOL(f, f, "SIGMA_MOTION_DEC", "arcsec/yr", sigma_motion_dec);
    ADDCOL(u8, u8, "NOBSERVATIONS", nil, nobs);
    ADDCOL(i64, i64, "ID", nil, id);

	for (ob=0; ob<AN_N_OBSERVATIONS; ob++) {
		char fld[32];
		sprintf(fld, "CATALOG_%i", ob);
        ADDCOL(u8, u8, fld, nil, obs[ob].catalog);
		sprintf(fld, "BAND_%i", ob);
        ADDCOL(c, c, fld, nil, obs[ob].band);
		sprintf(fld, "ID_%i", ob);
        ADDCOL(i, i, fld, nil, obs[ob].id);
		sprintf(fld, "MAG_%i", ob);
        ADDCOL(f, f, fld, "mag", obs[ob].mag);
		sprintf(fld, "SIGMA_MAG_%i", ob);
        ADDCOL(f, f, fld, "mag", obs[ob].sigma_mag);
    }
}
#undef ADDCOL

int an_catalog_read_entries(an_catalog* cat, int offset,
							int count, an_entry* entries) {
    return fitstable_read_structs(cat, entries, sizeof(an_entry),
                                  offset, count);
}

an_entry* an_catalog_read_entry(an_catalog* cat) {
    return (an_entry*)fitstable_next_struct(cat);
}

int an_catalog_write_entry(an_catalog* cat, an_entry* entry) {
    return fitstable_write_struct(cat, entry);
}

int an_catalog_count_entries(an_catalog* cat) {
	return fitstable_nrows(cat);
}

int an_catalog_close(an_catalog* cat) {
    if (fitstable_close(cat)) {
        fprintf(stderr, "Error closing AN catalog file: %s\n", strerror(errno));
        return -1;
    }
	return 0;
}

qfits_header* an_catalog_get_primary_header(const an_catalog* cat) {
    return fitstable_get_primary_header(cat);
}

int an_catalog_sync(an_catalog* cat) {
    FILE* fid = cat->fid;
    off_t offset = ftello(fid);
    if (offset == -1) {
        SYSERROR("Failed to get file offset while syncing file %s", cat->fn);
        return -1;
    }
    if (fits_pad_file(fid) ||
        fdatasync(fileno(fid))) {
        SYSERROR("Error padding and syncing AN catalog file for file %s", cat->fn);
        return -1;
    }
    if (an_catalog_fix_headers(cat)) {
        ERROR("Error fixing headers while syncing AN catalog file %s", cat->fn);
        return -1;
    }
    
    if (fseeko(fid, offset, SEEK_SET)) {
        SYSERROR("Failed to seek back to the end of the file while syncing file %s", cat->fn);
        return -1;
    }
    return 0;
}

an_catalog* an_catalog_open(char* fn) {
	an_catalog* cat = NULL;
    cat = fitstable_open(fn);
    if (!cat) {
        fprintf(stderr, "an-catalog: failed to open table.\n");
        an_catalog_close(cat);
        return NULL;
    }
    add_columns(cat, FALSE);
    an_catalog_set_blocksize(cat, 1000);
    if (fitstable_read_extension(cat, 1)) {
        fprintf(stderr, "an-catalog: table in extension 1 didn't contain the required columns.\n");
        an_catalog_close(cat);
        return NULL;
    }
	return cat;
}

an_catalog* an_catalog_open_for_writing(char* fn) {
	an_catalog* cat;
    qfits_header* hdr;
    cat = fitstable_open_for_writing(fn);
    if (!cat) {
        fprintf(stderr, "an-catalog: failed to open table.\n");
        an_catalog_close(cat);
        return NULL;
    }
    add_columns(cat, TRUE);
    hdr = fitstable_get_primary_header(cat);
	qfits_header_add(hdr, "AN_CAT", "T", "This is an Astrometry.net catalog.", NULL);
    qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_ANCAT, "Astrometry.net file type", NULL);
    return cat;
}

int an_catalog_write_headers(an_catalog* cat) {
    if (fitstable_write_primary_header(cat))
        return -1;
    return fitstable_write_header(cat);
}

int an_catalog_fix_headers(an_catalog* cat) {
    if (fitstable_fix_primary_header(cat))
        return -1;
    return fitstable_fix_header(cat);
}

void an_catalog_set_blocksize(an_catalog* cat, int block) {
    fitstable_use_buffered_reading(cat, sizeof(an_entry), block);
}

int64_t an_catalog_get_id(int catversion, int64_t starid) {
	int64_t mask = 0x000000ffffffffffLL;
	int64_t catv = catversion;
	int64_t id;
	catv = (catv << 48);
	assert((catv & (~mask)) == 0);
	id = (starid & mask) | catv;
	return id;
}
