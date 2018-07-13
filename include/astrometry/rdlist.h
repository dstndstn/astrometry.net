/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef RDLIST_H
#define RDLIST_H

#include <stdio.h>
#include <sys/types.h>

#include "astrometry/xylist.h"
#include "astrometry/starutil.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_RDLS "RDLS"

typedef xylist_t rdlist_t;

struct rd_t {
    double* ra;
    double* dec;
    int N;
};
typedef struct rd_t rd_t;

void rd_getradec(const rd_t* f, int i, double* ra, double* dec);
double rd_getra (rd_t* f, int i);
double rd_getdec(rd_t* f, int i);
void rd_setra (rd_t* f, int i, double ra);
void rd_setdec(rd_t* f, int i, double dec);
int rd_n(rd_t* f);

void rd_from_dl(rd_t* r, dl* l);
void rd_from_array(rd_t* r, double* radec, int N);

// Just free the data, not the field itself.
void rd_free_data(rd_t* f);

void rd_free(rd_t* f);

void rd_alloc_data(rd_t* f, int N);

rd_t* rd_alloc(int N);

void rd_copy(rd_t* dest, int dest_offset, const rd_t* src, int src_offset, int N);

rd_t* rd_get_subset(const rd_t* src, int offset, int N);

rdlist_t* rdlist_open(const char* fn);

rdlist_t* rdlist_open_for_writing(const char* fn);

//void rdlist_set_antype(rdlist_t* ls, const char* type);
#define rdlist_set_antype xylist_set_antype

void rdlist_set_raname(rdlist_t* ls, const char* name);
void rdlist_set_decname(rdlist_t* ls, const char* name);
void rdlist_set_ratype(rdlist_t* ls, tfits_type type);
void rdlist_set_dectype(rdlist_t* ls, tfits_type type);
void rdlist_set_raunits(rdlist_t* ls, const char* units);
void rdlist_set_decunits(rdlist_t* ls, const char* units);

//int rdlist_write_primary_header(rdlist_t* ls);
#define rdlist_write_primary_header xylist_write_primary_header

#define rdlist_fix_primary_header xylist_fix_primary_header

//void rdlist_next_field(rdlist_t* ls);
#define rdlist_next_field xylist_next_field

#define rdlist_open_field xylist_open_field

#define rdlist_n_fields xylist_n_fields

//int rdlist_write_header(rdlist_t* ls);
#define rdlist_write_header xylist_write_header

int rdlist_write_field(rdlist_t* ls, rd_t* fld);

int rdlist_write_one_row(rdlist_t* ls, rd_t* fld, int row);

int rdlist_write_one_radec(rdlist_t* ls, double ra, double dec);

#define rdlist_add_tagalong_column xylist_add_tagalong_column

#define rdlist_write_tagalong_column xylist_write_tagalong_column

#define rdlist_read_tagalong_column xylist_read_tagalong_column

// (input rd_t* is optional; if not given, a new one is allocated and returned.)
rd_t* rdlist_read_field(rdlist_t* ls, rd_t* fld);

rd_t* rdlist_read_field_num(rdlist_t* ls, int ext, rd_t* fld);

//int rdlist_fix_header(rdlist_t* ls);
#define rdlist_fix_header xylist_fix_header

//int rdlist_close(rdlist_t* ls);
#define rdlist_close xylist_close

//qfits_header* rdlist_get_primary_header(rdlist_t* ls);
#define rdlist_get_primary_header xylist_get_primary_header

//qfits_header* rdlist_get_header(rdlist_t* ls);
#define rdlist_get_header xylist_get_header

#endif

