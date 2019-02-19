/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>

#include "rdlist.h"

void rd_getradec(const rd_t* f, int i, double* ra, double* dec) {
    assert(i < f->N);
    *ra  = f->ra [i];
    *dec = f->dec[i];
}

double rd_getra (rd_t* f, int i) {
    assert(i < f->N);
    return f->ra[i];
}

double rd_getdec(rd_t* f, int i) {
    assert(i < f->N);
    return f->dec[i];
}

void rd_setra (rd_t* f, int i, double ra) {
    assert(i < f->N);
    f->ra[i] = ra;
}

void rd_setdec(rd_t* f, int i, double dec) {
    assert(i < f->N);
    f->dec[i] = dec;
}

int rd_n(rd_t* r) {
    return r->N;
}

void rd_free_data(rd_t* f) {
    if (!f) return;
    free(f->ra);
    free(f->dec);
}

void rd_free(rd_t* f) {
    rd_free_data(f);
    free(f);
}

void rd_alloc_data(rd_t* f, int N) {
    f->ra = malloc(N * sizeof(double));
    f->dec = malloc(N * sizeof(double));
    f->N = N;
}

rd_t* rd_alloc(int N) {
    rd_t* rd = calloc(1, sizeof(rd_t));
    rd_alloc_data(rd, N);
    return rd;
}

void rd_copy(rd_t* dest, int dest_offset, const rd_t* src, int src_offset, int N) {
    int i;
    for (i=0; i<N; i++) {
        dest->ra [i + dest_offset] = src->ra [i + src_offset];
        dest->dec[i + dest_offset] = src->dec[i + src_offset];
    }
}

rd_t* rd_get_subset(const rd_t* src, int offset, int N) {
    rd_t* dest = rd_alloc(N);
    rd_copy(dest, 0, src, offset, N);
    return dest;
}

void rd_from_array(rd_t* r, double* radec, int N) {
    int i;
    rd_alloc_data(r, N);
    for (i=0; i<r->N; i++) {
        r->ra [i] = radec[i*2];
        r->dec[i] = radec[i*2+1];
    }
}

void rd_from_dl(rd_t* r, dl* l) {
    int i;
    rd_alloc_data(r, dl_size(l)/2);
    for (i=0; i<r->N; i++) {
        r->ra [i] = dl_get(l, i*2);
        r->dec[i] = dl_get(l, i*2+1);
    }
}

void rdlist_set_raname(rdlist_t* ls, const char* name) {
    xylist_set_xname(ls, name);
}
void rdlist_set_decname(rdlist_t* ls, const char* name) {
    xylist_set_yname(ls, name);
}
void rdlist_set_ratype(rdlist_t* ls, tfits_type type) {
    xylist_set_xtype(ls, type);
}
void rdlist_set_dectype(rdlist_t* ls, tfits_type type) {
    xylist_set_ytype(ls, type);
}
void rdlist_set_raunits(rdlist_t* ls, const char* units) {
    xylist_set_xunits(ls, units);
}
void rdlist_set_decunits(rdlist_t* ls, const char* units) {
    xylist_set_yunits(ls, units);
}


/*
 qfits_header* rdlist_get_header(rdlist* ls) {
 return xylist_get_header(ls);
 }
 
 qfits_header* rdlist_get_field_header(rdlist* ls) {
 return xylist_get_field_header(ls, ls->field);
 }
 */

rdlist_t* rdlist_open(const char* fn) {
    rdlist_t* rtn = xylist_open(fn);
    if (!rtn) return NULL;
    rdlist_set_raname(rtn, "RA");
    rdlist_set_decname(rtn, "DEC");
    xylist_set_include_flux(rtn, FALSE);
    xylist_set_include_background(rtn, FALSE);
    return rtn;
}

rd_t* rdlist_read_field(rdlist_t* ls, rd_t* fld) {
    starxy_t xy;
    if (!xylist_read_field(ls, &xy)) {
        return NULL;
    }
    if (!fld) {
        fld = calloc(1, sizeof(rd_t));
    }
    fld->ra  = xy.x;
    fld->dec = xy.y;
    fld->N   = xy.N;
    return fld;
}

rd_t* rdlist_read_field_num(rdlist_t* ls, int ext, rd_t* fld) {
    if (rdlist_open_field(ls, ext)) {
        return NULL;
    }
    return rdlist_read_field(ls, fld);
}

int rdlist_write_field(rdlist_t* ls, rd_t* fld) {
    starxy_t xy;
    memset(&xy, 0, sizeof(starxy_t));
    xy.x = fld->ra;
    xy.y = fld->dec;
    xy.N = fld->N;
    return xylist_write_field(ls, &xy);
}

int rdlist_write_one_row(rdlist_t* ls, rd_t* fld, int row) {
    starxy_t xy;
    memset(&xy, 0, sizeof(starxy_t));
    xy.x = fld->ra;
    xy.y = fld->dec;
    xy.N = fld->N;
    return xylist_write_one_row(ls, &xy, row);
}

int rdlist_write_one_radec(rdlist_t* ls, double ra, double dec) {
    starxy_t xy;
    xy.x = &ra;
    xy.y = &dec;
    xy.N = 1;
    return xylist_write_one_row(ls, &xy, 0);
}

rdlist_t* rdlist_open_for_writing(const char* fn) {
    rdlist_t* rtn = xylist_open_for_writing(fn);
    xylist_set_antype(rtn, AN_FILETYPE_RDLS);
    rdlist_set_raname (rtn, "RA");
    rdlist_set_decname(rtn, "DEC");
    rdlist_set_raunits (rtn, "deg");
    rdlist_set_decunits(rtn, "deg");
    rdlist_set_ratype (rtn, TFITS_BIN_TYPE_D);
    rdlist_set_dectype(rtn, TFITS_BIN_TYPE_D);
    xylist_set_include_flux(rtn, FALSE);
    xylist_set_include_background(rtn, FALSE);
    return rtn;
}

