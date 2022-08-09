/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <errors.h>

#include "qfits_header.h"
#include "xylist.h"
#include "fitstable.h"
#include "fitsioutils.h"
#include "an-bool.h"
#include "keywords.h"

static anbool is_writing(xylist_t* ls) {
    return (ls->table && ls->table->fid) ? TRUE : FALSE;
}
static anbool is_reading(xylist_t* ls) {
    return !is_writing(ls);
}

int xylist_get_imagew(xylist_t* ls) {
    int W;
    qfits_header* hdr = xylist_get_header(ls);
    W = qfits_header_getint(hdr, "IMAGEW", 0);
    if (!W) {
        // some axy's have IMAGEW/H only in the primary hdr...
        hdr = xylist_get_primary_header(ls);
        W = qfits_header_getint(hdr, "IMAGEW", 0);
    }
    return W;
}

int xylist_get_imageh(xylist_t* ls) {
    qfits_header* hdr = xylist_get_header(ls);
    int H;
    H = qfits_header_getint(hdr, "IMAGEH", 0);
    if (!H) {
        // some axy's have IMAGEW/H only in the primary hdr...
        hdr = xylist_get_primary_header(ls);
        H = qfits_header_getint(hdr, "IMAGEH", 0);
    }
    return H;
}

anbool xylist_is_file_xylist(const char* fn, int ext,
                             const char* xcolumn, const char* ycolumn,
                             char** reason) {
    int rtn;
    xylist_t* xyls;
    err_t* err;

    errors_push_state();
    err = errors_get_state();
    err->print_f = NULL;
    err->save = TRUE;

    xyls = xylist_open(fn);
    if (!xyls) {
        goto bail;
    }

    if (fitstable_n_extensions(xyls->table) < 2) {
        ERROR("FITS file does not have any extensions");
        goto bail;
    }

    if (ext) {
        if (xylist_open_extension(xyls, ext)) {
            ERROR("Failed to open xylist extension %i", ext);
            goto bail;
        }
    } else {
        ext = 1;
    }

    if (xcolumn)
        xylist_set_xname(xyls, xcolumn);
    if (ycolumn)
        xylist_set_yname(xyls, ycolumn);

    fitstable_add_read_column_struct(xyls->table, fitscolumn_double_type(),
                                     1, 0, fitscolumn_any_type(), xyls->xname, TRUE);
    fitstable_add_read_column_struct(xyls->table, fitscolumn_double_type(),
                                     1, 0, fitscolumn_any_type(), xyls->yname, TRUE);

    rtn = fitstable_read_extension(xyls->table, ext);
    if (rtn)
        fitstable_error_report_missing(xyls->table);
    xylist_close(xyls);
    if (rtn)
        goto bail;

    errors_pop_state();
    return TRUE;

 bail:
    if (reason) *reason = error_get_errs(err, ": ");
    errors_pop_state();
    return FALSE;
}

static xylist_t* xylist_new() {
    xylist_t* xy = calloc(1, sizeof(xylist_t));
    xy->xname = "X";
    xy->yname = "Y";
    xy->xtype = TFITS_BIN_TYPE_D;
    xy->ytype = TFITS_BIN_TYPE_D;
    return xy;
}

xylist_t* xylist_open(const char* fn) {
    qfits_header* hdr;
    xylist_t* ls = NULL;
    ls = xylist_new();
    if (!ls) {
        ERROR("Failed to allocate xylist");
        return NULL;
    }
    ls->table = fitstable_open_mixed(fn);
    if (!ls->table) {
        ERROR("Failed to open FITS table %s", fn);
        free(ls);
        return NULL;
    }
    ls->table->extension = 1;

    hdr = fitstable_get_primary_header(ls->table);
    ls->antype = fits_get_dupstring(hdr, "AN_FILE");
    // not including primary extension...
    ls->nfields = fitstable_n_extensions(ls->table) - 1;
    ls->include_flux = TRUE;
    ls->include_background = TRUE;
    assert(is_reading(ls));
    return ls;
}

xylist_t* xylist_open_for_writing(const char* fn) {
    xylist_t* ls;
    qfits_header* hdr;
    ls = xylist_new();
    if (!ls) {
        ERROR("Failed to allocate xylist");
        return NULL;
    }
    ls->table = fitstable_open_for_writing(fn);
    if (!ls->table) {
        ERROR("Failed to open FITS table for writing");
        free(ls);
        return NULL;
    }
    // since we have to call xylist_next_field() before writing the first one...
    ls->table->extension = 0;

    xylist_set_antype(ls, AN_FILETYPE_XYLS);
    hdr = fitstable_get_primary_header(ls->table);
    qfits_header_add(hdr, "AN_FILE", ls->antype, "Astrometry.net file type", NULL);
    assert(is_writing(ls));
    return ls;
}

int xylist_add_tagalong_column(xylist_t* ls, tfits_type c_type,
                               int arraysize, tfits_type fits_type,
                               const char* name, const char* units) {
    assert(is_writing(ls));
    fitstable_add_write_column_struct(ls->table, c_type, arraysize,
                                      0, fits_type, name, units);
    return fitstable_ncols(ls->table) - 1;
}

int xylist_write_tagalong_column(xylist_t* ls, int colnum,
                                 int offset, int N,
                                 void* data, int datastride) {
    assert(is_writing(ls));
    return fitstable_write_one_column(ls->table, colnum, offset, N,
                                      data, datastride);
}

void* xylist_read_tagalong_column(xylist_t* ls, const char* colname,
                                  tfits_type c_type) {
    assert(is_reading(ls));
    return fitstable_read_column_array(ls->table, colname, c_type);
}

sl* xylist_get_tagalong_column_names(xylist_t* ls, sl* lst) {
    char* x;
    char* y;
    assert(is_reading(ls));
    lst = fitstable_get_fits_column_names(ls->table, lst);
    x = sl_remove_string_bycaseval(lst, ls->xname);
    y = sl_remove_string_bycaseval(lst, ls->yname);
    free(x);
    free(y);
    return lst;
}

void xylist_set_antype(xylist_t* ls, const char* type) {
    free(ls->antype);
    ls->antype = strdup(type);
}

int xylist_close(xylist_t* ls) {
    int rtn = 0;
    if (ls->table) {
        if (fitstable_close(ls->table)) {
            ERROR("Failed to close xylist table");
            rtn = -1;
        }
    }
    free(ls->antype);
    free(ls);
    return rtn;
}

void xylist_set_xname(xylist_t* ls, const char* name) {
    ls->xname = name;
}
void xylist_set_yname(xylist_t* ls, const char* name) {
    ls->yname = name;
}
void xylist_set_xtype(xylist_t* ls, tfits_type type) {
    ls->xtype = type;
}
void xylist_set_ytype(xylist_t* ls, tfits_type type) {
    ls->ytype = type;
}
void xylist_set_xunits(xylist_t* ls, const char* units) {
    ls->xunits = units;
}
void xylist_set_yunits(xylist_t* ls, const char* units) {
    ls->yunits = units;
}

void xylist_set_include_flux(xylist_t* ls, anbool inc) {
    ls->include_flux = inc;
}

void xylist_set_include_background(xylist_t* ls, anbool inc) {
    ls->include_background = inc;
}

int xylist_n_fields(xylist_t* ls) {
    return ls->nfields;
}

int xylist_write_one_row(xylist_t* ls, starxy_t* fld, int row) {
    // FIXME -- does this work if you're using background but not flux?
    assert(is_writing(ls));
    return fitstable_write_row(ls->table, fld->x + row, fld->y + row,
                               ls->include_flux ? fld->flux + row : NULL,
                               ls->include_background ? fld->background + row : NULL);
}

int xylist_write_one_row_data(xylist_t* ls, double x, double y,
                              double flux, double bg) {
    assert(is_writing(ls));
    return fitstable_write_row(ls->table, &x, &y,
                               ls->include_flux ? &flux : NULL,
                               ls->include_background ? &bg : NULL);
}

int xylist_write_field(xylist_t* ls, starxy_t* fld) {
    int i;
    assert(is_writing(ls));
    assert(fld);
    for (i=0; i<fld->N; i++) {
        if (fitstable_write_row(ls->table, fld->x + i, fld->y + i,
                                ls->include_flux ? fld->flux + i : NULL,
                                ls->include_background ? fld->background + i : NULL))
            return -1;
    }
    return 0;
}

starxy_t* xylist_read_field(xylist_t* ls, starxy_t* fld) {
    anbool freeit = FALSE;
    tfits_type dubl = fitscolumn_double_type();
    assert(is_reading(ls));

    if (!ls->table->table)
        xylist_open_field(ls, ls->table->extension);

    if (!ls->table->table) {
        // FITS table not found.
        return NULL;
    }

    if (!fld) {
        fld = calloc(1, sizeof(starxy_t));
        freeit = TRUE;
    }

    fld->N = fitstable_nrows(ls->table);
    fld->x = fitstable_read_column(ls->table, ls->xname, dubl);
    fld->y = fitstable_read_column(ls->table, ls->yname, dubl);
    if (ls->include_flux)
        fld->flux = fitstable_read_column(ls->table, "FLUX", dubl);
    else
        fld->flux = NULL;
    if (ls->include_background)
        fld->background = fitstable_read_column(ls->table, "BACKGROUND", dubl);
    else
        fld->background = NULL;

    if (!(fld->x && fld->y)) {
        free(fld->x);
        free(fld->y);
        free(fld->flux);
        free(fld->background);
        if (freeit)
            free(fld);
        return NULL;
    }
    return fld;
}

starxy_t* xylist_read_field_num(xylist_t* ls, int ext, starxy_t* fld) {
    starxy_t* rtn;
    assert(is_reading(ls));
    if (xylist_open_field(ls, ext)) {
        ERROR("Failed to open field %i from xylist", ext);
        return NULL;
    }
    rtn = xylist_read_field(ls, fld);
    if (!rtn)
        ERROR("Failed to read field %i from xylist", ext);
    return rtn;
}

int xylist_open_field(xylist_t* ls, int i) {
    return fitstable_open_extension(ls->table, i);
}
int xylist_open_extension(xylist_t* ls, int i) {
    return fitstable_open_extension(ls->table, i);
}

/*
 Used for both reading and writing.

 --when writing: start a new field.  Set up the table and header
 structures so that they can be added to before writing the field
 header.

 --when reading: move on to the next extension.
 */
int xylist_next_field(xylist_t* ls) {
    if (is_writing(ls)) {
        fitstable_next_extension(ls->table);
        fitstable_clear_table(ls->table);
        ls->nfields++;
    } else {
        int rtn = fitstable_open_next_extension(ls->table);
        //int rtn = fitstable_open_extension(ls->table, ls->table->extension);
        if (rtn)
            return rtn;
    }
    return 0;
}

qfits_header* xylist_get_primary_header(xylist_t* ls) {
    qfits_header* hdr;
    hdr = fitstable_get_primary_header(ls->table);
    // ??
    qfits_header_mod(hdr, "AN_FILE", ls->antype, "Astrometry.net file type");
    return hdr;
}

qfits_header* xylist_get_header(xylist_t* ls) {
    if (is_writing(ls) && !ls->table->header) {
        fitstable_add_write_column_convert(ls->table, ls->xtype,
                                           fitscolumn_double_type(),
                                           ls->xname, ls->xunits);
        fitstable_add_write_column_convert(ls->table, ls->ytype,
                                           fitscolumn_double_type(),
                                           ls->yname, ls->yunits);

        if (ls->include_flux)
            fitstable_add_write_column_convert(ls->table,
                                               fitscolumn_double_type(),
                                               fitscolumn_double_type(),
                                               "FLUX", "fluxunits");
        if (ls->include_background)
            fitstable_add_write_column_convert(ls->table,
                                               fitscolumn_double_type(),
                                               fitscolumn_double_type(),
                                               "BACKGROUND", "fluxunits");

        fitstable_new_table(ls->table);
    }
    if (is_reading(ls) && !ls->table->header)
        xylist_open_field(ls, ls->table->extension);
    return fitstable_get_header(ls->table);
}

int xylist_write_primary_header(xylist_t* ls) {
    assert(is_writing(ls));
    // ensure we've added the AN_FILE header...
    xylist_get_primary_header(ls);
    return fitstable_write_primary_header(ls->table);
}

int xylist_fix_primary_header(xylist_t* ls) {
    assert(is_writing(ls));
    return fitstable_fix_primary_header(ls->table);
}

int xylist_write_header(xylist_t* ls) {
    assert(is_writing(ls));
    // ensure we've added our columns to the table...
    xylist_get_header(ls);
    return fitstable_write_header(ls->table);
}

int xylist_fix_header(xylist_t* ls) {
    assert(is_writing(ls));
    return fitstable_fix_header(ls->table);
}

