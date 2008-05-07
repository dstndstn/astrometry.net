/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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

#include <string.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>
#include <sys/param.h>
#include <errors.h>

#include "fitstable.h"
#include "fitsioutils.h"
#include "fitsfile.h"
#include "ioutils.h"

struct fitscol_t {
    char* colname;

    tfits_type fitstype;
    tfits_type ctype;
    char* units;
    int arraysize;

    bool required;

    // size of one data item
    // computed: fits_sizeof({fits,c}type)
    int fitssize;
    int csize;

    // When being used to write to a C struct, the offset in the struct.
    bool in_struct;
    int coffset;

    /*
     // Called to retrieve data to be written to the output file.
     //void (*get_data_callback)(void* data, int offset, int N, fitscol_t* col, void* user);
     //void* get_data_user;
     // Called when data has been read from the input file.
     //void (*put_data_callback)(void* data, int offset, int N, fitscol_t* col, void* user);
     //void* put_data_user;
     // Where to read/write data from/to.
     void* cdata;
     int cdata_stride;
     */

    // column number of the FITS table.
    int col;
};
typedef struct fitscol_t fitscol_t;

static void fitstable_add_columns(fitstable_t* tab, fitscol_t* cols, int Ncols);
static void fitstable_add_column(fitstable_t* tab, fitscol_t* col);
static void fitstable_create_table(fitstable_t* tab);

/*
 int fitstable_read_array(const fitstable_t* tab,
 //const fitscol_t* cols, int Ncols,
 int offset, int N,
 void* data, int stride);

 int fitstable_write_array(const fitstable_t* tab,
 int offset, int N,
 const void* data, int stride);
 */

static int ncols(const fitstable_t* t) {
    return bl_size(t->cols);
}
static fitscol_t* getcol(const fitstable_t* t, int i) {
    return bl_access(t->cols, i);
}

bool is_writing(const fitstable_t* t) {
    return t->fid ? TRUE : FALSE;
}

tfits_type fitscolumn_int_type() {
    switch (sizeof(int)) {
    case 2:
        return TFITS_BIN_TYPE_I;
    case 4:
        return TFITS_BIN_TYPE_J;
    case 8:
        return TFITS_BIN_TYPE_K;
    }
    return -1;
}

tfits_type fitscolumn_double_type() {
    return TFITS_BIN_TYPE_D;
}

tfits_type fitscolumn_float_type() {
    return TFITS_BIN_TYPE_E;
}

tfits_type fitscolumn_char_type() {
    return TFITS_BIN_TYPE_A;
}

tfits_type fitscolumn_u8_type() {
    return TFITS_BIN_TYPE_B;
}
tfits_type fitscolumn_i16_type() {
    return TFITS_BIN_TYPE_I;
}
tfits_type fitscolumn_i32_type() {
    return TFITS_BIN_TYPE_J;
}
tfits_type fitscolumn_i64_type() {
    return TFITS_BIN_TYPE_K;
}
tfits_type fitscolumn_boolean_type() {
    return TFITS_BIN_TYPE_L;
}

tfits_type fitscolumn_bool_type() {
    return TFITS_BIN_TYPE_B;
}

tfits_type fitscolumn_bitfield_type() {
    return TFITS_BIN_TYPE_X;
}

// When reading: allow this column to match to any FITS type.
tfits_type fitscolumn_any_type() {
    return (tfits_type)-1;
}

int fitscolumn_get_size(fitscol_t* col) {
    return col->fitssize * col->arraysize;
}

int fitstable_ncols(fitstable_t* t) {
    return ncols(t);
}

int fitstable_row_size(fitstable_t* t) {
    // FIXME - should this return the size of the *existing* FITS table
    // (when reading), or just the columns we care about (those in "cols")?
    return t->table->tab_w;
    /*
     int i, N, sz;
     N = ncols(t);
     sz = 0;
     for (i=0; i<N; i++)
     sz += fitscolumn_get_size(getcol(t, i));
     return sz;
     */
}

void fitstable_add_write_column(fitstable_t* tab, tfits_type t,
                                const char* name, const char* units) {
    fitstable_add_write_column_array_convert(tab, t, t, 1, name, units);
}

void fitstable_add_write_column_convert(fitstable_t* tab,
                                        tfits_type fits_type,
                                        tfits_type c_type,
                                        const char* name,
                                        const char* units) {
    fitstable_add_write_column_array_convert(tab, fits_type, c_type, 1, name, units);
}

void fitstable_add_write_column_array(fitstable_t* tab, tfits_type t,
                                      int arraysize,
                                      const char* name,
                                      const char* units) {
    fitstable_add_write_column_array_convert(tab, t, t, arraysize, name, units);
}

void fitstable_add_write_column_array_convert(fitstable_t* tab,
                                              tfits_type fits_type,
                                              tfits_type c_type,
                                              int arraysize,
                                              const char* name,
                                              const char* units) {
    fitscol_t col;
    memset(&col, 0, sizeof(fitscol_t));
    col.colname = strdup_safe(name);
    col.units = strdup_safe(units);
    col.fitstype = fits_type;
    col.ctype = c_type;
    col.arraysize = arraysize;
    col.in_struct = FALSE;
    fitstable_add_column(tab, &col);
}

void fitstable_add_write_column_struct(fitstable_t* tab,
                                       tfits_type c_type,
                                       int arraysize,
                                       int structoffset,
                                       tfits_type fits_type,
                                       const char* name,
                                       const char* units) {
    fitstable_add_column_struct(tab, c_type, arraysize, structoffset,
                                fits_type, name, units, FALSE);
}

void fitstable_add_read_column_struct(fitstable_t* tab,
                                      tfits_type c_type,
                                      int arraysize,
                                      int structoffset,
                                      tfits_type fits_type,
                                      const char* name,
                                      bool required) {
    fitstable_add_column_struct(tab, c_type, arraysize, structoffset,
                                fits_type, name, NULL, required);
}

void fitstable_add_column_struct(fitstable_t* tab,
                                 tfits_type c_type,
                                 int arraysize,
                                 int structoffset,
                                 tfits_type fits_type,
                                 const char* name,
                                 const char* units,
                                 bool required) {
    fitscol_t col;
    memset(&col, 0, sizeof(fitscol_t));
    col.colname = strdup_safe(name);
    col.units = strdup_safe(units);
    col.fitstype = fits_type;
    col.ctype = c_type;
    col.arraysize = arraysize;
    col.in_struct = TRUE;
    col.coffset = structoffset;
    col.required = required;
    fitstable_add_column(tab, &col);
}

int fitstable_read_structs(fitstable_t* tab, void* struc,
                           int strucstride, int offset, int N) {
    int i;
    void* tempdata = NULL;
    int highwater = 0;

    for (i=0; i<ncols(tab); i++) {
        void* dest;
        int stride;
        void* finaldest;
        int finalstride;
        fitscol_t* col = getcol(tab, i);
        if (col->col == -1)
            continue;
        if (!col->in_struct)
            continue;
        finaldest = ((char*)struc) + col->coffset;
        finalstride = strucstride;

        if (col->fitstype != col->ctype) {
            int NB = col->fitssize * col->arraysize * N;
            if (NB > highwater) {
                free(tempdata);
                tempdata = malloc(NB);
                highwater = NB;
            }
            dest = tempdata;
            stride = col->fitssize * col->arraysize;
        } else {
            dest = finaldest;
            stride = finalstride;
        }

        // Read from FITS file...
        qfits_query_column_seq_to_array(tab->table, col->col, offset, N, dest, stride);

        if (col->fitstype != col->ctype) {
            fits_convert_data(finaldest, finalstride, col->ctype,
                              dest, stride, col->fitstype,
                              col->arraysize, N);
        }
    }
    free(tempdata);

    if (tab->postprocess_read_structs)
        return tab->postprocess_read_structs(tab, struc, strucstride, offset, N);

    return 0;
}

int fitstable_read_struct(fitstable_t* tab, int offset, void* struc) {
    return fitstable_read_structs(tab, struc, 0, offset, 1);
}

int fitstable_write_struct(fitstable_t* table, const void* struc) {
    int i;
    char* buf = NULL;
    int Nbuf = 0;
	int ret = 0;

    for (i=0; i<ncols(table); i++) {
        fitscol_t* col;
        void* columndata;
        col = getcol(table, i);
        if (!col->in_struct)
            // Set "columndata" to NULL, which causes fits_write_data_array
            // to skip the required number of bytes.
            columndata = NULL;
        else
            columndata = ((char*)struc) + col->coffset;
        if (columndata && col->fitstype != col->ctype) {
            int sz = MAX(256, MAX(col->csize, col->fitssize) * col->arraysize);
            if (sz > Nbuf) {
                free(buf);
                buf = malloc(sz);
            }
            fits_convert_data(buf, col->fitssize, col->fitstype,
                              columndata, col->csize, col->ctype,
                              col->arraysize, 1);
            columndata = buf;
        }
        ret = fits_write_data_array(table->fid, columndata,
                                    col->fitstype, col->arraysize);
        if (ret)
            break;
    }
    free(buf);
    table->table->nr++;
    return ret;
}

int fitstable_write_one_column(fitstable_t* table, int colnum,
                               int rowoffset, int nrows,
                               const void* src, int src_stride) {
    off_t foffset;
    off_t start;
    int i;
    char* buf = NULL;
    fitscol_t* col;

    foffset = ftello(table->fid);

    // jump to row start...
    start = table->end_table_offset + table->table->tab_w * rowoffset;
    // + column start
    for (i=0; i<colnum; i++) {
        col = getcol(table, i);
        start += col->fitssize * col->arraysize;
    }

    if (fseeko(table->fid, start, SEEK_SET)) {
        SYSERROR("Failed to fseeko() to the start of the file.");
        return -1;
    }

    col = getcol(table, colnum);
    if (col->fitstype != col->ctype) {
        int sz;
        sz = col->fitssize * col->arraysize * nrows;
        buf = malloc(sz);
        fits_convert_data(buf, col->fitssize * col->arraysize, col->fitstype,
                          src, src_stride, col->ctype,
                          col->arraysize, nrows);
        src = buf;
        src_stride = col->fitssize * col->arraysize;
    }

    for (i=0; i<nrows; i++) {
        if (fseeko(table->fid, start + i * table->table->tab_w, SEEK_SET) ||
            fits_write_data_array(table->fid, src, col->fitstype, col->arraysize)) {
            SYSERROR("Failed to write row %i of column %i", rowoffset+i, colnum);
            return -1;
        }
        src = ((const char*)src) + src_stride;
    }
    free(buf);

    if (fseeko(table->fid, foffset, SEEK_SET)) {
        SYSERROR("Failed to restore file offset.");
        return -1;
    }
    return 0;
}

int fitstable_write_row(fitstable_t* table, ...) {
	va_list ap;
	int ncols = fitstable_ncols(table);
	int i;
    char* buf = NULL;
    int Nbuf = 0;
	int ret = 0;

	va_start(ap, table);
	for (i=0; i<ncols; i++) {
		fitscol_t* col;
        col = bl_access(table->cols, i);
		void *columndata;
        if (col->in_struct) {
            // Set "columndata" to NULL, which causes fits_write_data_array
            // to skip the required number of bytes.
            columndata = NULL;
        } else {
            columndata = va_arg(ap, void *);
        }

        if (columndata && col->fitstype != col->ctype) {
            int sz = MAX(256, MAX(col->csize, col->fitssize) * col->arraysize);
            if (sz > Nbuf) {
                free(buf);
                buf = malloc(sz);
            }
            fits_convert_data(buf, col->fitssize, col->fitstype,
                              columndata, col->csize, col->ctype,
                              col->arraysize, 1);
            columndata = buf;
        }
        ret = fits_write_data_array(table->fid, columndata,
                                    col->fitstype, col->arraysize);
        if (ret)
            break;
    }
	va_end(ap);
    free(buf);
    table->table->nr++;
    return ret;
}

void fitstable_clear_table(fitstable_t* tab) {
    bl_remove_all(tab->cols);
}

static void* read_array(const fitstable_t* tab,
                        const char* colname, tfits_type ctype,
                        bool array_ok, int offset, int Nread) {
    int colnum;
    qfits_col* col;
    int fitssize;
    int csize;
    int fitstype;
    int arraysize;
    void* data;
    int N;

    colnum = fits_find_column(tab->table, colname);
    if (colnum == -1) {
        ERROR("Column \"%s\" not found in FITS table %s.\n", colname, tab->fn);
        return NULL;
    }
    col = tab->table->col + colnum;
    if (!array_ok && (col->atom_nb != 1)) {
        ERROR("Column \"%s\" in FITS table %s is an array of size %i, not a scalar.\n",
              colname, tab->fn, col->atom_nb);
        return NULL;
    }

    arraysize = col->atom_nb;
    fitstype = col->atom_type;
    fitssize = fits_get_atom_size(fitstype);
    csize = fits_get_atom_size(ctype);
    N = tab->table->nr;
    if (Nread == -1)
        Nread = N;
    if (offset == -1)
        offset = 0;
    data = calloc(MAX(csize, fitssize), Nread * arraysize);

    qfits_query_column_seq_to_array(tab->table, colnum, offset, Nread, data, 
                                    fitssize * arraysize);

    if (fitstype != ctype) {
        if (csize <= fitssize) {
            fits_convert_data(data, csize * arraysize, ctype,
                              data, fitssize * arraysize, fitstype,
                              arraysize, Nread);
            if (csize < fitssize)
                data = realloc(data, csize * Nread * arraysize);
        } else if (csize > fitssize) {
            // HACK - stride backwards from the end of the array
            fits_convert_data(((char*)data) + ((Nread*arraysize)-1) * csize,
                              -csize, ctype,
                              ((char*)data) + ((Nread*arraysize)-1) * fitssize,
                              -fitssize, fitstype,
                              1, Nread * arraysize);
        }
    }
	return data;
}

void* fitstable_read_column(const fitstable_t* tab,
                            const char* colname, tfits_type ctype) {
    return read_array(tab, colname, ctype, FALSE, -1, -1);
}

void* fitstable_read_column_array(const fitstable_t* tab,
                                  const char* colname, tfits_type t) {
    return read_array(tab, colname, t, TRUE, -1, -1);
}

void* fitstable_read_column_offset(const fitstable_t* tab,
                                   const char* colname, tfits_type ctype,
                                   int offset, int N) {
    return read_array(tab, colname, ctype, FALSE, offset, N);
}

qfits_header* fitstable_get_primary_header(const fitstable_t* t) {
    return t->primheader;
}

qfits_header* fitstable_get_header(fitstable_t* t) {
    if (!t->header) {
        fitstable_new_table(t);
    }
    return t->header;
}

void fitstable_next_extension(fitstable_t* tab) {
    if (tab->fid)
        fits_pad_file(tab->fid);
    qfits_table_close(tab->table);
    qfits_header_destroy(tab->header);
    tab->extension++;
    tab->table = NULL;
    tab->header = NULL;
}

static fitstable_t* fitstable_new() {
    fitstable_t* tab;
    tab = calloc(1, sizeof(fitstable_t));
    if (!tab)
        return tab;
    tab->cols = bl_new(8, sizeof(fitscol_t));
    return tab;
}

fitstable_t* fitstable_open(const char* fn) {
    fitstable_t* tab;
    tab = fitstable_new();
    if (!tab) {
		ERROR("Failed to allocate new FITS table structure.");
        goto bailout;
	}
    tab->extension = 1;
    tab->fn = strdup_safe(fn);
    tab->primheader = qfits_header_read(fn);
    if (!tab->primheader) {
        ERROR("Failed to read primary FITS header from %s.", fn);
        goto bailout;
    }
    if (fitstable_open_extension(tab, tab->extension)) {
        ERROR("Failed to open extension %i in file %s.", tab->extension, fn);
        goto bailout;
    }
	return tab;
 bailout:
    if (tab) {
        fitstable_close(tab);
    }
    return NULL;
}

fitstable_t* fitstable_open_for_writing(const char* fn) {
    fitstable_t* tab;
    tab = fitstable_new();
    if (!tab)
        goto bailout;
    tab->fn = strdup_safe(fn);
    tab->fid = fopen(fn, "wb");
	if (!tab->fid) {
		SYSERROR("Couldn't open output file %s for writing", fn);
		goto bailout;
	}
	tab->primheader = qfits_table_prim_header_default();
    return tab;

 bailout:
    if (tab) {
        fitstable_close(tab);
    }
    return NULL;
}

int fitstable_close(fitstable_t* tab) {
    int i;
    int rtn = 0;
    if (!tab) return 0;
    if (tab->fid) {
        if (fclose(tab->fid)) {
            SYSERROR("Failed to close output file %s", tab->fn);
            rtn = -1;
        }
    }
    if (tab->primheader)
        qfits_header_destroy(tab->primheader);
    if (tab->header)
        qfits_header_destroy(tab->header);
    if (tab->table)
        qfits_table_close(tab->table);
    free(tab->fn);
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        free(col->colname);
        free(col->units);
    }
    bl_free(tab->cols);
    if (tab->br) {
        buffered_read_free(tab->br);
        free(tab->br);
    }
    free(tab);
    return rtn;
}

static void fitstable_add_columns(fitstable_t* tab, fitscol_t* cols, int Ncols) {
    int i;
    for (i=0; i<Ncols; i++) {
        fitscol_t* col = bl_append(tab->cols, cols + i);
        col->csize = fits_get_atom_size(col->ctype);
        col->fitssize = fits_get_atom_size(col->fitstype);
    }
}

static void fitstable_add_column(fitstable_t* tab, fitscol_t* col) {
    fitstable_add_columns(tab, col, 1);
}

int fitstable_get_array_size(fitstable_t* tab, const char* name) {
    qfits_col* qcol;
    int colnum;
    colnum = fits_find_column(tab->table, name);
    if (colnum == -1)
        return -1;
    qcol = tab->table->col + colnum;
    return qcol->atom_nb;
}

int fitstable_get_type(fitstable_t* tab, const char* name) {
    qfits_col* qcol;
    int colnum;
    colnum = fits_find_column(tab->table, name);
    if (colnum == -1)
        return -1;
    qcol = tab->table->col + colnum;
    return qcol->atom_type;
}

int fitstable_open_next_extension(fitstable_t* tab) {
    tab->extension++;
    return fitstable_open_extension(tab, tab->extension);
}

int fitstable_open_extension(fitstable_t* tab, int ext) {
    if (tab->table) {
        qfits_table_close(tab->table);
    }
	tab->table = qfits_table_open(tab->fn, ext);
	if (!tab->table) {
		ERROR("FITS extension %i in file %s is not a table (or there was an error opening the file)", ext, tab->fn);
		return -1;
	}
    if (tab->header) {
        qfits_header_destroy(tab->header);
    }
    tab->header = qfits_header_readext(tab->fn, ext);
	if (!tab->header) {
		ERROR("Couldn't get header for FITS extension %i in file %s", ext, tab->fn);
		return -1;
	}
    return 0;
}

int fitstable_read_extension(fitstable_t* tab, int ext) {
    int i;
    int ok = 1;

    if (tab->table) {
        qfits_table_close(tab->table);
    }
	tab->table = qfits_table_open(tab->fn, ext);
	if (!tab->table) {
		ERROR("FITS extension %i in file %s is not a table (or there was an error opening the file)", ext, tab->fn);
		return -1;
	}
    if (tab->header) {
        qfits_header_destroy(tab->header);
    }
    tab->header = qfits_header_readext(tab->fn, ext);
	if (!tab->header) {
		ERROR("Couldn't get header for FITS extension %i in file %s", ext, tab->fn);
		return -1;
	}
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        qfits_col* qcol;

        // FIXME? set this here?
        col->csize = fits_get_atom_size(col->ctype);

        // Column found?
        col->col = fits_find_column(tab->table, col->colname);
        if (col->col == -1)
            continue;
        qcol = tab->table->col + col->col;

        // Type & array size correct?
        if (col->fitstype != fitscolumn_any_type() &&
            col->fitstype != qcol->atom_type) {
            col->col = -1;
            continue;
        }
        col->fitstype = qcol->atom_type;
        col->fitssize = fits_get_atom_size(col->fitstype);

        if (col->arraysize) {
            if (col->fitstype == TFITS_BIN_TYPE_X) {
                if (((col->arraysize + 7)/8) != qcol->atom_nb) {
                    col->col = -1;
                    continue;
                }
            } else {
                if (col->arraysize != qcol->atom_nb) {
                    col->col = -1;
                    continue;
                }
            }
        }
        if (col->fitstype == TFITS_BIN_TYPE_X) {
            col->arraysize = 8 * qcol->atom_nb;
        } else {
            col->arraysize = qcol->atom_nb;
        }
    }

    if (tab->br) {
        buffered_read_reset(tab->br);
        tab->br->ntotal = tab->table->nr;
    }

    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        if (col->col == -1 && col->required) {
            ok = 0;
            break;
        }
    }
	if (ok) return 0;
	return -1;
}

int fitstable_write_primary_header(fitstable_t* t) {
    return fitsfile_write_primary_header(t->fid, t->primheader,
                                         &t->end_header_offset, t->fn);
}

int fitstable_fix_primary_header(fitstable_t* t) {
    return fitsfile_fix_primary_header(t->fid, t->primheader,
                                       &t->end_header_offset, t->fn);
}

// Called just before starting to write a new field.
int fitstable_new_table(fitstable_t* t) {
    if (t->table) {
        qfits_table_close(t->table);
    }
    fitstable_create_table(t);
    if (t->header) {
        qfits_header_destroy(t->header);
    }
    t->header = qfits_table_ext_header_default(t->table);
    return 0;
}

int fitstable_write_header(fitstable_t* t) {
    if (!t->header) {
        if (fitstable_new_table(t)) {
            return -1;
        }
    }
    return fitsfile_write_header(t->fid, t->header,
                                 &t->table_offset, &t->end_table_offset,
                                 t->extension, t->fn);
}

int fitstable_fix_header(fitstable_t* t) {
    // update NAXIS2 to reflect the number of rows written.
    fits_header_mod_int(t->header, "NAXIS2", t->table->nr, NULL);

    if (fitsfile_fix_header(t->fid, t->header,
                            &t->table_offset, &t->end_table_offset,
                            t->extension, t->fn)) {
        return -1;
    }
    return fits_pad_file(t->fid);
}

void fitstable_close_table(fitstable_t* tab) {
    int i;
    if (tab->table) {
        qfits_table_close(tab->table);
        tab->table = NULL;
    }
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        col->col = -1;
        col->fitssize = 0;
        col->arraysize = 0;
        col->fitstype = fitscolumn_any_type();
    }
}

int fitstable_nrows(fitstable_t* t) {
    if (!t->table) return 0;
    return t->table->nr;
}

void fitstable_print_missing(fitstable_t* tab, FILE* f) {
    int i;
    fprintf(f, "Missing required rows: ");
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        if (col->col == -1 && col->required) {
            fprintf(f, "%s ", col->colname);
        }
    }
    //fprintf(f, "\n");
}

static void fitstable_create_table(fitstable_t* tab) {
    qfits_table* qt;
    int i;

    qt = qfits_table_new("", QFITS_BINTABLE, 0, ncols(tab), 0);
    tab->table = qt;

    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
		char* nil = "";
		assert(col->colname);
        fits_add_column(qt, i, col->fitstype, col->arraysize,
						col->units ? col->units : nil, col->colname);
    }
}

static int refill_buffer(void* userdata, void* buffer, uint offset, uint n) {
    fitstable_t* tab = userdata;
    if (fitstable_read_structs(tab, buffer, tab->br->elementsize, offset, n)) {
        ERROR("Error refilling FITS table read buffer");
        return -1;
    }
    return 0;
}

void fitstable_use_buffered_reading(fitstable_t* tab, int elementsize, int Nbuffer) {
    if (tab->br) {
        assert(tab->br->elementsize == elementsize);
        buffered_read_resize(tab->br, Nbuffer);
    } else {
        tab->br = buffered_read_new(elementsize, Nbuffer, 0, refill_buffer, tab);
    }
}

void fitstable_set_buffer_fill_function(fitstable_t* tab,
                                        int (*refill_buffer)(void* userdata, void* buffer, unsigned int offs, unsigned int nelems),
                                        void* userdata) {
    assert(tab->br);
    tab->br->refill_buffer = refill_buffer;
    tab->br->userdata = userdata;
}

void* fitstable_next_struct(fitstable_t* tab) {
    if (!tab->br) return NULL;
    return buffered_read(tab->br);
}

int fitstable_pushback(fitstable_t* tab) {
    if (!tab->br) return -1;
    buffered_read_pushback(tab->br);
    return 0;
}

