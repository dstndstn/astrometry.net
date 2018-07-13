/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef XYLIST_H
#define XYLIST_H

#include "astrometry/starxy.h"
#include "astrometry/fitstable.h"

#define AN_FILETYPE_XYLS "XYLS"

/**
 Writing:

 xylist_open_for_writing()
 xylist_write_primary_header()

 for (each extension) {
 // optionally:
 xylist_add_tagalong_column()

 xylist_write_header()
 
 // either:
 xylist_write_one_row() // repeatedly
 // or:
 xylist_write_field()

 // optionally:
 xylist_write_tagalong_column()

 xylist_fix_header()

 xylist_next_field()
 }

 xylist_fix_primary_header()
 xylist_close()



 Reading:


 xylist_t* xyls = xylist_open("my.xyls");
 int nf = xylist_n_fields(xyls);
 starxy_t* xy = xylist_read_field(xyls, NULL);


 */

/*
 One table per field.
 One row per star.
 */
struct xylist_t {
    int parity;

    fitstable_t* table;

    char* antype; // Astrometry.net filetype string.

    const char* xname;
    const char* yname;
    const char* xunits;
    const char* yunits;
    tfits_type xtype;
    tfits_type ytype;

    anbool include_flux;
    anbool include_background;

    // When reading: total number of fields in this file.
    int nfields;
};
typedef struct xylist_t xylist_t;

xylist_t* xylist_open(const char* fn);

xylist_t* xylist_open_for_writing(const char* fn);

void xylist_set_antype(xylist_t* ls, const char* type);

void xylist_set_xname(xylist_t* ls, const char* name);
void xylist_set_yname(xylist_t* ls, const char* name);
void xylist_set_xtype(xylist_t* ls, tfits_type type);
void xylist_set_ytype(xylist_t* ls, tfits_type type);
void xylist_set_xunits(xylist_t* ls, const char* units);
void xylist_set_yunits(xylist_t* ls, const char* units);

void xylist_set_include_flux(xylist_t* ls, anbool inc);
void xylist_set_include_background(xylist_t* ls, anbool inc);


// when writing.
// Returns the column number of the added column; use this number
// in the call to xylist_write_tagalong_column.
int xylist_add_tagalong_column(xylist_t* ls, tfits_type c_type,
                               int arraysize, tfits_type fits_type,
                               const char* name, const char* units);
int xylist_write_tagalong_column(xylist_t* ls, int colnum,
                                 int offset, int N,
                                 void* data, int datastride);

// when reading.
// Returns the tagged-along column names.
// If 'lst' is non-NULL, the names will be added to it; otherwise a new
// sl* will be allocated and returned.
sl* xylist_get_tagalong_column_names(xylist_t* ls, sl* lst);

void* xylist_read_tagalong_column(xylist_t* ls, const char* colname,
                                  tfits_type c_type);

int xylist_write_primary_header(xylist_t* ls);

int xylist_fix_primary_header(xylist_t* ls);

int xylist_next_field(xylist_t* ls);

//int xylist_start_field(xylist_t* ls);

int xylist_open_field(xylist_t* ls, int i);
int xylist_open_extension(xylist_t* ls, int i);

int xylist_write_header(xylist_t* ls);

int xylist_write_field(xylist_t* ls, starxy_t* fld);

int xylist_write_one_row(xylist_t* ls, starxy_t* fld, int row);

int xylist_write_one_row_data(xylist_t* ls, double x, double y, double flux, double bg);

// (input starxy_t* is optional; if not given, a new one is allocated and returned.)
starxy_t* xylist_read_field(xylist_t* ls, starxy_t* fld);

starxy_t* xylist_read_field_num(xylist_t* ls, int ext, starxy_t* fld);

int xylist_fix_header(xylist_t* ls);

int xylist_close(xylist_t* ls);

qfits_header* xylist_get_primary_header(xylist_t* ls);

qfits_header* xylist_get_header(xylist_t* ls);

int xylist_get_imagew(xylist_t* ls);
int xylist_get_imageh(xylist_t* ls);

int xylist_n_fields(xylist_t* ls);

// Is the given filename an xylist?
anbool xylist_is_file_xylist(const char* fn, int ext,
                             const char* xcolumn, const char* ycolumn,
                             char** reason);

#endif
