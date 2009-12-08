/*
  This file is part of the Astrometry.net suite.
  Copyright 2008, 2009 Dustin Lang.

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
#include <assert.h>
#include <stdarg.h>
#include <sys/param.h>
#include <errors.h>

#include "fitstable.h"
#include "fitsioutils.h"
#include "fitsfile.h"
#include "ioutils.h"
#include "an-endian.h"

#include "log.h"

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

    // column number of the FITS table.
    int col;
};
typedef struct fitscol_t fitscol_t;

// For in-memory: storage of previously-written extensions.
struct fitsext {
	qfits_header* header;
	qfits_table* table;
	bl* rows;
};
typedef struct fitsext fitsext_t;


static void fitstable_add_columns(fitstable_t* tab, fitscol_t* cols, int Ncols);
static void fitstable_add_column(fitstable_t* tab, fitscol_t* col);
static void fitstable_create_table(fitstable_t* tab);

static int ncols(const fitstable_t* t) {
    return bl_size(t->cols);
}
static fitscol_t* getcol(const fitstable_t* t, int i) {
    return bl_access(t->cols, i);
}

static off_t get_row_offset(const fitstable_t* table, int row) {
	assert(table->end_table_offset);
	assert(table->table);
	assert(table->table->tab_w);
	return table->end_table_offset + table->table->tab_w * row;
}

int fitscolumn_get_size(fitscol_t* col) {
	/*
	 if (col->fitstype == TFITS_BIN_TYPE_X) {
	 // UGLY HACK!
	 return (7 + col->fitssize * col->arraysize) / 8;
	 } else
	 */
	return col->fitssize * col->arraysize;
}

static int offset_of_column(const fitstable_t* table, int colnum) {
	int i;
	int offset = 0;
	assert(colnum <= ncols(table));
	for (i=0; i<ncols(table); i++) {
		fitscol_t* col;
		col = getcol(table, i);
		offset += fitscolumn_get_size(col);
	}
	return offset;
}

static bool is_writing(const fitstable_t* t) {
    return t->fid ? TRUE : FALSE;
	//return t->writing;
}

static void ensure_row_list_exists(fitstable_t* table) {
	if (!table->rows) {
		// how big are the rows?
		int rowsize = offset_of_column(table, bl_size(table->cols));
		table->rows = bl_new(1024, rowsize);
	}
}

static bool in_memory(const fitstable_t* t) {
	return t->inmemory;
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

int fitstable_ncols(fitstable_t* t) {
    return ncols(t);
}

int fitstable_read_row_data(fitstable_t* table, int row, void* dest) {
	int R;
	off_t off;
	assert(table);
	assert(row >= 0);
	assert(row < fitstable_nrows(table));
	assert(dest);
	R = fitstable_row_size(table);
	if (in_memory(table)) {
		memcpy(dest, bl_access(table->rows, row), R);
		return 0;
	}
	if (!table->readfid) {
		int start, end;
		table->readfid = fopen(table->fn, "rb");
		if (!table->readfid) {
			SYSERROR("Failed to open FITS table %s for reading", table->fn);
			return -1;
		}
		if (qfits_get_datinfo(table->fn, table->extension, &start, &end)) {
			ERROR("Failed to find start of table: %s", table->fn);
			return -1;
		}
		// end of table header is start of table data.
		table->end_table_offset = start;
	}
	off = get_row_offset(table, row);
	if (fseeko(table->readfid, off, SEEK_SET)) {
		SYSERROR("Failed to fseeko() to read a row");
		return -1;
	}
	if (fread(dest, 1, R, table->readfid) != R) {
		SYSERROR("Failed to read a row from %s", table->fn);
		return -1;
	}
	return 0;
}

int fitstable_write_row_data(fitstable_t* table, void* data) {
	int R;
	assert(table);
	assert(data);
	R = fitstable_row_size(table);
	if (in_memory(table)) {
		// Endian-flip here??  Or elsewhere...
		int i;
		int off = 0;
		char* cursor = data;
		for (i=0; i<ncols(table); i++) {
			int j;
			fitscol_t* col = getcol(table, i);
			for (j=0; j<col->arraysize; j++) {
				endian_swap(cursor, col->fitssize);
				cursor += col->fitssize;
			}
		}

		ensure_row_list_exists(table);
		bl_append(table->rows, data);
		// ?
		table->table->nr++;
		return 0;
	}
	if (fwrite(data, 1, R, table->fid) != R) {
		SYSERROR("Failed to write a row to %s", table->fn);
		return -1;
	}
    table->table->nr++;
	return 0;
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

char* fitstable_get_column_name(const fitstable_t* src, int i) {
	fitscol_t* col = getcol(src, i);
	return col->colname;
}

void fitstable_copy_columns(const fitstable_t* src, fitstable_t* dest) {
	int i;
	for (i=0; i<ncols(src); i++) {
		fitscol_t* col = getcol(src, i);
		char* name = col->colname;
		char* units = col->units;
		col->colname = strdup_safe(col->colname);
		col->units   = strdup_safe(col->units);
		fitstable_add_column(dest, col);
		col->colname = name;
		col->units = units;
	}
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

void fitstable_add_fits_columns_as_struct(fitstable_t* tab) {
	int i;
	int off = 0;
	for (i=0; i<tab->table->nc; i++) {
		qfits_col* qcol = tab->table->col + i;
		/*
		 if (qcol->atom_type == TFITS_BIN_TYPE_X) {
		 }
		 */
		fitstable_add_read_column_struct(tab, qcol->atom_type, qcol->atom_nb,
										 off, qcol->atom_type, qcol->tlabel, TRUE);
		off += fitscolumn_get_size(getcol(tab, ncols(tab)-1));
	}
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

int fitstable_remove_column(fitstable_t* tab, const char* name) {
    int i;
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        if (strcasecmp(name, col->colname) == 0) {
			free(col->colname);
			free(col->units);
            bl_remove_index(tab->cols, i);
            return 0;
        }
    }
    return -1;
}

void fitstable_print_columns(fitstable_t* tab) {
	int i;
	printf("Table columns:\n");
	for (i=0; i<ncols(tab); i++) {
		fitscol_t* col = getcol(tab, i);
		printf("  %i: %s: fits type %i, C type %i, arraysize %i, fitssize %i, C size %i, C offset %i, FITS column num: %i\n",
			   i, col->colname, col->fitstype, col->ctype, col->arraysize, col->fitssize, col->csize, col->coffset, col->col);
	}
}

int fitstable_read_structs(fitstable_t* tab, void* struc,
                           int strucstride, int offset, int N) {
    int i;
    void* tempdata = NULL;
    int highwater = 0;

    //logverb("fitstable_read_structs: stride %i, offset %i, N %i\n",strucstride, offset, N);

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
            int NB = fitscolumn_get_size(col) * N;
            if (NB > highwater) {
                free(tempdata);
                tempdata = malloc(NB);
                highwater = NB;
            }
            dest = tempdata;
            stride = fitscolumn_get_size(col);
        } else {
            dest = finaldest;
            stride = finalstride;
        }

		if (in_memory(tab)) {
			int j;
			int off = offset_of_column(tab, i);
			int sz;
			if (!tab->rows) {
				ERROR("No data has been written to this fitstable");
				return -1;
			}
			if (offset + N > bl_size(tab->rows)) {
				ERROR("Number of data items requested exceeds number of rows: offset %i, n %i, nrows %i", offset, N, bl_size(tab->rows));
				return -1;
			}

			//logverb("column %i: dest offset %i, stride %i, row offset %i, input offset %i, size %i (%ix%i)\n", i, (int)(dest - struc), stride, offset, off, fitscolumn_get_size(col), col->fitssize, col->arraysize);
			sz = fitscolumn_get_size(col);
			for (j=0; j<N; j++)
				memcpy(((char*)dest) + j * stride,
					   ((char*)bl_access(tab->rows, offset+j)) + off,
					   sz);
		} else {
			// Read from FITS file...
			qfits_query_column_seq_to_array(tab->table, col->col, offset, N, dest, stride);
		}

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

// One of "struc" or "ap" should be non-null.
static int write_one(fitstable_t* table, const void* struc, va_list ap) {
    int i;
    char* buf = NULL;
    int Nbuf = 0;
	int ret = 0;
	int nc = ncols(table);

	char* thisrow = NULL;
	int rowoff = 0;

	if (in_memory(table)) {
		ensure_row_list_exists(table);
		thisrow = calloc(1, bl_datasize(table->rows));
	}

    for (i=0; i<nc; i++) {
        fitscol_t* col;
        const char* columndata;
        col = getcol(table, i);
		if (col->in_struct) {
			if (struc)
				columndata = struc + col->coffset;
			else
				columndata = NULL;
		} else {
			if (struc)
				columndata = NULL;
			else
				columndata = va_arg(ap, void *);
		}
		// If "columndata" is NULL, fits_write_data_array
		// skips the required number of bytes.
		// This allows both structs and normal columns to coexist
		// (in theory -- is this ever used?)

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
		
		if (in_memory(table)) {
			int nb = fitscolumn_get_size(col);
			memcpy(thisrow + rowoff, columndata, nb);
			rowoff += nb;
		} else {
			ret = fits_write_data_array(table->fid, columndata,
										col->fitstype, col->arraysize);
			if (ret)
				break;
		}
    }
    free(buf);
	if (in_memory(table))
		bl_append(table->rows, thisrow);
	free(thisrow);
    table->table->nr++;
    return ret;
}


int fitstable_write_struct(fitstable_t* table, const void* struc) {
	va_list nil;
	//if (!table->table)
	//fitstable_create_table(table);
	return write_one(table, struc, nil);
}

int fitstable_write_structs(fitstable_t* table, const void* struc, int stride, int N) {
	int i;
	char* s = (char*)struc;
	//if (!table->table)
	//fitstable_create_table(table);
	for (i=0; i<N; i++) {
		if (fitstable_write_struct(table, s)) {
			return -1;
		}
		s += stride;
	}
	return 0;
}

int fitstable_write_row(fitstable_t* table, ...) {
	int ret;
	va_list ap;
	if (!table->table)
		fitstable_create_table(table);
	va_start(ap, table);
	ret = write_one(table, NULL, ap);
	va_end(ap);
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
	int off;

	off = offset_of_column(table, colnum);
	if (!in_memory(table)) {
		foffset = ftello(table->fid);
		// jump to row start...
		start = get_row_offset(table, rowoffset) + off;
		if (fseeko(table->fid, start, SEEK_SET)) {
			SYSERROR("Failed to fseeko() to the start of the file.");
			return -1;
		}
	}

    col = getcol(table, colnum);
    if (col->fitstype != col->ctype) {
		int sz = col->fitssize * col->arraysize * nrows;
        buf = malloc(sz);
        fits_convert_data(buf, col->fitssize * col->arraysize, col->fitstype,
                          src, src_stride, col->ctype,
                          col->arraysize, nrows);
        src = buf;
        src_stride = col->fitssize * col->arraysize;
    }

	if (in_memory(table)) {
		for (i=0; i<nrows; i++) {
			memcpy(((char*)bl_access(table->rows, rowoffset + i)) + off,
				   src, col->fitssize * col->arraysize);
			src = ((const char*)src) + src_stride;
		}
	} else {
		for (i=0; i<nrows; i++) {
			if (fseeko(table->fid, start + i * table->table->tab_w, SEEK_SET) ||
				fits_write_data_array(table->fid, src, col->fitstype, col->arraysize)) {
				SYSERROR("Failed to write row %i of column %i", rowoffset+i, colnum);
				return -1;
			}
			src = ((const char*)src) + src_stride;
		}
	}
    free(buf);

	if (!in_memory(table)) {
		if (fseeko(table->fid, foffset, SEEK_SET)) {
			SYSERROR("Failed to restore file offset.");
			return -1;
		}
	}
    return 0;
}

void fitstable_clear_table(fitstable_t* tab) {
    int i;
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        free(col->colname);
        free(col->units);
    }
    bl_remove_all(tab->cols);
}

static void* read_array_into(const fitstable_t* tab,
							 const char* colname, tfits_type ctype,
							 bool array_ok, int offset, int Nread,
							 void* dest, int deststride) {
    int colnum;
    qfits_col* col;
    int fitssize;
    int csize;
    int fitstype;
    int arraysize;

	char* tempdata = NULL;
    char* cdata;
	char* fitsdata;
	int cstride;
	int fitsstride;
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

	if (dest)
		cdata = dest;
	else
		cdata = calloc(csize, Nread * arraysize);

	if (dest && deststride > -1)
		cstride = deststride;
	else
		cstride = csize * arraysize;

	fitsstride = fitssize * arraysize;
	if (csize < fitssize) {
		// Need to allocate a bigger temp array and down-convert the data.
		// HACK - could set data=tempdata and realloc after (if 'dest' is NULL)
		tempdata = calloc(fitssize, Nread * arraysize);
		fitsdata = tempdata;
	} else {
		// We'll read the data into the first fraction of the output array.
		fitsdata = cdata;
	}

	if (in_memory(tab)) {
		int i;
		int off;
		int sz;
		if (!tab->rows) {
			ERROR("No data has been written to this fitstable");
			return NULL;
		}
		if (offset + Nread > bl_size(tab->rows)) {
			ERROR("Number of data items requested exceeds number of rows: offset %i, n %i, nrows %i", offset, Nread, bl_size(tab->rows));
			return NULL;
		}
		off = fits_offset_of_column(tab->table, colnum);
		sz = fitsstride;
		for (i=0; i<Nread; i++)
			memcpy(fitsdata + i * fitsstride,
				   ((char*)bl_access(tab->rows, offset+i)) + off,
				   sz);
	} else {
		qfits_query_column_seq_to_array(tab->table, colnum, offset, Nread,
										(unsigned char*)fitsdata, fitsstride);
	}

	if (fitstype != ctype) {
		if (csize <= fitssize) {
			// work forward
			fits_convert_data(cdata, cstride, ctype,
							  fitsdata, fitsstride, fitstype,
							  arraysize, Nread);
		} else {
			// work backward from the end of the array
            fits_convert_data(cdata + ((Nread*arraysize)-1) * csize,
							  -csize, ctype,
							  fitsdata + ((Nread*arraysize)-1) * fitssize,
							  -fitssize, fitstype,
							  1, Nread * arraysize);
		}
	}

	free(tempdata);
	return cdata;
}

static void* read_array(const fitstable_t* tab,
                        const char* colname, tfits_type ctype,
                        bool array_ok, int offset, int Nread) {
	return read_array_into(tab, colname, ctype, array_ok, offset, Nread, NULL, -1);
}

int fitstable_read_column_offset_into(const fitstable_t* tab,
									  const char* colname, tfits_type read_as_type,
									  void* dest, int stride, int start, int N) {
	void* res;
	res = read_array_into(tab, colname, read_as_type, FALSE, start, N, dest, stride);
	if (!res)
		return -1;
	return 0;
}

int fitstable_read_column_into(const fitstable_t* tab,
							   const char* colname, tfits_type read_as_type,
							   void* dest, int stride) {
	void* res;
	res = read_array_into(tab, colname, read_as_type, FALSE, 0, -1, dest, stride);
	if (!res)
		return -1;
	return 0;
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
	if (is_writing(tab))
        fits_pad_file(tab->fid);

	if (in_memory(tab)) {
		fitsext_t ext;
		if (!tab->table)
			return;
		// update NAXIS2
		fitstable_fix_header(tab);
		ext.table = tab->table;
		ext.header = tab->header;
		ext.rows = tab->rows;
		bl_append(tab->extensions, &ext);
		tab->rows = NULL;
	} else {
		qfits_table_close(tab->table);
		qfits_header_destroy(tab->header);
	}
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

fitstable_t* fitstable_open_in_memory() {
    fitstable_t* tab;
    tab = fitstable_new();
    if (!tab) {
		ERROR("Failed to allocate new FITS table structure");
        goto bailout;
	}
    tab->fn = NULL;
    tab->fid = NULL;
	tab->primheader = qfits_table_prim_header_default();
	tab->inmemory = TRUE;
	tab->extensions = bl_new(16, sizeof(fitsext_t));
    return tab;

 bailout:
    if (tab) {
        fitstable_close(tab);
    }
    return NULL;
}

int fitstable_switch_to_reading(fitstable_t* table) {
	assert(in_memory(table));
	// store the current extension.
	fitstable_next_extension(table);
	// This resets all the meta-data about the table, meaning a reader
	// can then re-add columns it is interested in.
	fitstable_clear_table(table);
	table->extension = 1;
	return fitstable_open_extension(table, table->extension);
}

fitstable_t* fitstable_open(const char* fn) {
    fitstable_t* tab;
    tab = fitstable_new();
    if (!tab) {
		ERROR("Failed to allocate new FITS table structure");
        goto bailout;
	}
    tab->extension = 1;
    tab->fn = strdup_safe(fn);
    tab->primheader = qfits_header_read(fn);
    if (!tab->primheader) {
        ERROR("Failed to read primary FITS header from %s", fn);
        goto bailout;
    }
    if (fitstable_open_extension(tab, tab->extension)) {
        ERROR("Failed to open extension %i in file %s", tab->extension, fn);
        goto bailout;
    }
	return tab;
 bailout:
    if (tab) {
        fitstable_close(tab);
    }
    return NULL;
}

static fitstable_t* open_for_writing(const char* fn, const char* mode) {
    fitstable_t* tab;
    tab = fitstable_new();
    if (!tab)
        goto bailout;
    tab->fn = strdup_safe(fn);
    tab->fid = fopen(fn, mode);
	if (!tab->fid) {
		SYSERROR("Couldn't open output file %s for writing", fn);
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
	fitstable_t* tab = open_for_writing(fn, "wb");
	if (!tab)
		return tab;
	tab->primheader = qfits_table_prim_header_default();
	return tab;
}

fitstable_t* fitstable_open_for_appending(const char* fn) {
	fitstable_t* tab = open_for_writing(fn, "r+b");
	if (!tab)
		return tab;
	if (fseeko(tab->fid, 0, SEEK_END)) {
		SYSERROR("Failed to seek to end of file");
		fitstable_close(tab);
		return NULL;
	}
    tab->primheader = qfits_header_read(fn);
    if (!tab->primheader) {
        ERROR("Failed to read primary FITS header from %s", fn);
		fitstable_close(tab);
		return NULL;
    }
	return tab;
}

int fitstable_close(fitstable_t* tab) {
    int i;
    int rtn = 0;
    if (!tab) return 0;
	if (is_writing(tab)) {
        if (fclose(tab->fid)) {
            SYSERROR("Failed to close output file %s", tab->fn);
            rtn = -1;
        }
    }
	if (tab->readfid) {
		fclose(tab->readfid);
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
	if (tab->rows) {
		bl_free(tab->rows);
	}
	if (tab->extensions) {
		for (i=0; i<bl_size(tab->extensions); i++) {
			fitsext_t* ext = bl_access(tab->extensions, i);
			if (ext->rows != tab->rows)
				bl_free(ext->rows);
			if (ext->header != tab->header)
				qfits_header_destroy(ext->header);
			if (ext->table != tab->table)
				qfits_table_close(ext->table);
		}
		bl_free(tab->extensions);
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
	if (in_memory(tab)) {
		fitsext_t* theext;
		if (ext > bl_size(tab->extensions)) {
			ERROR("Table has only %i extensions, but you requested #%i",
				  bl_size(tab->extensions), ext);
			return -1;
		}
		theext = bl_access(tab->extensions, ext-1);
		tab->table = theext->table;
		tab->header = theext->header;
		tab->rows = theext->rows;

	} else {
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
	}
    return 0;
}

int fitstable_read_extension(fitstable_t* tab, int ext) {
    int i;
    int ok = 1;

	if (fitstable_open_extension(tab, ext))
		return -1;

	if (tab->readfid) {
		// close FID so that table->end_table_offset gets refreshed.
		fclose(tab->readfid);
		tab->readfid = NULL;
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
            if (col->arraysize != qcol->atom_nb) {
                col->col = -1;
                continue;
            }
        }
		/*  This was causing problems with copying startree tag-along data (Tycho2 test case: FLAGS column)

		 if (col->fitstype == TFITS_BIN_TYPE_X) {
		 // ??? really??
		 col->arraysize = 8 * qcol->atom_nb;
		 } else {
		 col->arraysize = qcol->atom_nb;
		 }
		 */
		col->arraysize = qcol->atom_nb;
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
	if (in_memory(t)) return 0;
    return fitsfile_write_primary_header(t->fid, t->primheader,
                                         &t->end_header_offset, t->fn);
}

int fitstable_fix_primary_header(fitstable_t* t) {
	if (in_memory(t)) return 0;
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
	if (in_memory(t)) return 0;

    return fitsfile_write_header(t->fid, t->header,
                                 &t->table_offset, &t->end_table_offset,
                                 t->extension, t->fn);
}

int fitstable_pad_with(fitstable_t* t, char pad) {
    return fitsfile_pad_with(t->fid, pad);
}

int fitstable_fix_header(fitstable_t* t) {
    // update NAXIS2 to reflect the number of rows written.
    fits_header_mod_int(t->header, "NAXIS2", t->table->nr, NULL);

	if (in_memory(t)) return 0;

    if (fitsfile_fix_header(t->fid, t->header,
                            &t->table_offset, &t->end_table_offset,
                            t->extension, t->fn)) {
        return -1;
    }
    return 0; //fits_pad_file(t->fid);
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
    fprintf(f, "Missing required columns: ");
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        if (col->col == -1 && col->required) {
            fprintf(f, "%s ", col->colname);
        }
    }
}

void fitstable_error_report_missing(fitstable_t* tab) {
    int i;
    sl* missing = sl_new(4);
    char* mstr;
    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
        if (col->col == -1 && col->required)
            sl_append(missing, col->colname);
    }
    mstr = sl_join(missing, ", ");
    sl_free2(missing);
    ERROR("Missing required columns: %s", mstr);
    free(mstr);
}

static void fitstable_create_table(fitstable_t* tab) {
    qfits_table* qt;
    int i;

    qt = qfits_table_new("", QFITS_BINTABLE, 0, ncols(tab), 0);
    tab->table = qt;

    for (i=0; i<ncols(tab); i++) {
        fitscol_t* col = getcol(tab, i);
		char* nil = "";
        int arraysize;
		assert(col->colname);
        arraysize = col->arraysize;
        if (col->fitstype == TFITS_BIN_TYPE_X)
            arraysize = col->arraysize * 8;
        fits_add_column(qt, i, col->fitstype, arraysize,
						col->units ? col->units : nil, col->colname);
    }
}

static int refill_buffer(void* userdata, void* buffer, unsigned int offset, unsigned int n) {
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

