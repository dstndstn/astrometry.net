/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <errors.h>

#include "os-features.h"
#include "fitstable.h"
#include "fitsioutils.h"
#include "fitsfile.h"
#include "ioutils.h"
#include "an-endian.h"
#include "anqfits.h"

#include "log.h"

struct fitscol_t {
    char* colname;

    tfits_type fitstype;
    tfits_type ctype;
    char* units;
    int arraysize;

    anbool required;

    // size of one data item
    // computed: fits_sizeof({fits,c}type)
    int fitssize;
    int csize;

    // When being used to write to a C struct, the offset in the struct.
    anbool in_struct;
    int coffset;

    // column number of the FITS table.
    int col;
};
typedef struct fitscol_t fitscol_t;

// For in-memory: storage of previously-written extensions.
struct fitsext {
    qfits_header* header;
    qfits_table* table;
    // bl data size == row size.
    bl* rows;
};
typedef struct fitsext fitsext_t;



static anbool need_endian_flip() {
    return IS_BIG_ENDIAN == 0;
}

//static void fitstable_add_columns(fitstable_t* tab, fitscol_t* cols, int Ncols);
static fitscol_t* fitstable_add_column(fitstable_t* tab, fitscol_t* col);
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
    return table->end_table_offset + (off_t)(table->table->tab_w) * (off_t)(row);
}

int fitstable_n_extensions(const fitstable_t* t) {
    assert(t);
    assert(t->anq);
    return anqfits_n_ext(t->anq);
}

int fitscolumn_get_size(fitscol_t* col) {
    return col->fitssize * col->arraysize;
}

static int offset_of_column(const fitstable_t* table, int colnum) {
    int i;
    int offset = 0;
    assert(colnum <= ncols(table));
    for (i=0; i<colnum; i++) {
        fitscol_t* col;
        col = getcol(table, i);
        offset += fitscolumn_get_size(col);
    }
    return offset;
}

int fitstable_get_struct_size(const fitstable_t* table) {
    int rowsize = offset_of_column(table, bl_size(table->cols));
    return rowsize;
}

static anbool is_writing(const fitstable_t* t) {
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

static anbool in_memory(const fitstable_t* t) {
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

/*
 char fitscolumn_tfits_to_char(tfits_type t) {
 }
 */

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

/*
 const char* fitscolumn_format_string(tfits_type t) {
 switch (t) {
 case TFITS_BIN_TYPE_D:
 return "
 */

int fitstable_ncols(const fitstable_t* t) {
    return ncols(t);
}

int fitstable_read_row_data(fitstable_t* table, int row, void* dest) {
    return fitstable_read_nrows_data(table, row, 1, dest);
}

int fitstable_read_nrows_data(fitstable_t* table, int row0, int nrows,
                              void* dest) {
    int R;
    size_t nread;
    off_t off;
    assert(table);
    assert(row0 >= 0);
    assert((row0 + nrows) <= fitstable_nrows(table));
    assert(dest);
    R = fitstable_row_size(table);
    if (in_memory(table)) {
        int i;
        char* cdest = dest;
        for (i=0; i<nrows; i++)
            memcpy(cdest, bl_access(table->rows, row0 + i), R);
        return 0;
    }
    if (!table->readfid) {
        table->readfid = fopen(table->fn, "rb");
        if (!table->readfid) {
            SYSERROR("Failed to open FITS table %s for reading", table->fn);
            return -1;
        }

        assert(table->anq);
        off_t start;
        start = anqfits_data_start(table->anq, table->extension);
        table->end_table_offset = start;
    }
    off = get_row_offset(table, row0);
    if (fseeko(table->readfid, off, SEEK_SET)) {
        SYSERROR("Failed to fseeko() to read a row");
        return -1;
    }
    nread = (size_t)R * (size_t)nrows;
    if (fread(dest, 1, nread, table->readfid) != nread) {
        SYSERROR("Failed to read %i rows starting from %i, from %s", nrows, row0, table->fn);
        return -1;
    }
    return 0;
}

void fitstable_endian_flip_row_data(fitstable_t* table, void* data) {
    int i;
    char* cursor;
    if (!need_endian_flip())
        return;
    cursor = data;
    for (i=0; i<ncols(table); i++) {
        int j;
        fitscol_t* col = getcol(table, i);
        for (j=0; j<col->arraysize; j++) {
            /*
             if (col->fitssize == 8)
             printf("col '%s': d=%g, i=%li\n",
             col->colname, *((double*)cursor), *((uint64_t*)cursor));
             */
            endian_swap(cursor, col->fitssize);
            /*
             if (col->fitssize == 8)
             printf(" --> d=%g, i=%li\n",
             *((double*)cursor), *((uint64_t*)cursor));
             */

            cursor += col->fitssize;
        }
    }
}

static int write_row_data(fitstable_t* table, void* data, int R) {
    assert(table);
    assert(data);
    if (in_memory(table)) {
        ensure_row_list_exists(table);
        bl_append(table->rows, data);
        // ?
        table->table->nr++;
        return 0;
    }
    if (R == 0)
        R = fitstable_row_size(table);
    if (fwrite(data, 1, R, table->fid) != R) {
        SYSERROR("Failed to write a row to %s", table->fn);
        return -1;
    }
    assert(table->table);
    table->table->nr++;
    return 0;
}

int fitstable_write_row_data(fitstable_t* table, void* data) {
    return write_row_data(table, data, 0);
}

int fitstable_copy_rows_data(fitstable_t* intable, int* rows, int N, fitstable_t* outtable) {
    int R;
    char* buf = NULL;
    int i;
    // We need to endian-flip if we're going from FITS file <--> memory.
    anbool flip = need_endian_flip() && (in_memory(intable) != in_memory(outtable));
    R = fitstable_row_size(intable);
    buf = malloc(R);
    for (i=0; i<N; i++) {
        if (fitstable_read_row_data(intable, rows ? rows[i] : i, buf)) {
            ERROR("Failed to read data from input table");
            return -1;
        }
        // The in-memory side (input/output) does the flip.
        if (flip) {
            if (in_memory(intable))
                fitstable_endian_flip_row_data(intable, buf);
            else if (in_memory(outtable))
                fitstable_endian_flip_row_data(outtable, buf);
        }

        if (write_row_data(outtable, buf, R)) {
            ERROR("Failed to write data to output table");
            return -1;
        }
    }
    free(buf);
    return 0;
}

int fitstable_copy_row_data(fitstable_t* table, int row, fitstable_t* outtable) {
    return fitstable_copy_rows_data(table, &row, 1, outtable);
}

int fitstable_row_size(const fitstable_t* t) {
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
        fitscol_t* newcol = fitstable_add_column(dest, col);
        // fix string fields...
        newcol->colname = strdup_safe(newcol->colname);
        newcol->units   = strdup_safe(newcol->units);
    }
}

sl* fitstable_get_fits_column_names(const fitstable_t* t, sl* lst) {
    int i;
    if (!lst)
        lst = sl_new(16);
    for (i=0; i<t->table->nc; i++) {
        qfits_col* qcol = t->table->col + i;
        sl_append(lst, qcol->tlabel);
    }
    return lst;
}

int fitstable_get_N_fits_columns(const fitstable_t* t) {
    return t->table->nc;
}

const char* fitstable_get_fits_column_name(const fitstable_t* t, int i) {
    assert(i >= 0);
    assert(i < t->table->nc);
    return t->table->col[i].tlabel;
}

tfits_type fitstable_get_fits_column_type(const fitstable_t* t, int i) {
    assert(i >= 0);
    assert(i < t->table->nc);
    return t->table->col[i].atom_type;
}

int fitstable_get_fits_column_array_size(const fitstable_t* t, int i) {
    assert(i >= 0);
    assert(i < t->table->nc);
    return t->table->col[i].atom_nb;
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

int fitstable_n_fits_columns(const fitstable_t* tab) {
    return tab->table->nc;
}

int fitstable_add_fits_columns_as_struct4(const fitstable_t* intab,
					  fitstable_t* outtab,
					  const sl* colnames,
					  int c_offset,
                                          tfits_type fitstype) {
    int i, NC;
    int noc = ncols(outtab);
    NC = sl_size(colnames);
    for (i=0; i<NC; i++) {
        const qfits_col* qcol;
        fitscol_t* col;
        const char* name = sl_get_const(colnames, i);
        int j = fits_find_column(intab->table, name);
        int off;
        if (j == -1) {
            ERROR("Failed to find FITS column \"%s\"", name);
            return -1;
        }
        qcol = qfits_table_get_col(intab->table, j);
        // We give the offset of the column in the *input* table, so that
        // the resulting "outtab" can handle raw data from the "intab".
        off = fits_offset_of_column(intab->table, j);

        if (fitstype == TFITS_BIN_TYPE_UNKNOWN)
            fitstable_add_read_column_struct(outtab, qcol->atom_type, qcol->atom_nb,
                                             c_offset + off, qcol->atom_type, qcol->tlabel, TRUE);
        else
            fitstable_add_read_column_struct(outtab, qcol->atom_type, qcol->atom_nb,
                                             c_offset + off, fitstype, qcol->tlabel, TRUE);

        // set the FITS column number.
        col = getcol(outtab, ncols(outtab)-1);
        col->col = noc + i;
    }
    return 0;
}

int fitstable_add_fits_columns_as_struct3(const fitstable_t* intab,
					  fitstable_t* outtab,
					  const sl* colnames,
					  int c_offset) {
    return fitstable_add_fits_columns_as_struct4(intab, outtab, colnames,
                                                 c_offset, TFITS_BIN_TYPE_UNKNOWN);
}

void fitstable_add_fits_columns_as_struct2(const fitstable_t* intab,
                                           fitstable_t* outtab) {
    int i, NC;
    int off = 0;
    int noc = ncols(outtab);
    NC = fitstable_get_N_fits_columns(intab);
    for (i=0; i<NC; i++) {
        const qfits_col* qcol = qfits_table_get_col(intab->table, i);
        fitscol_t* col;
        /*
         if (qcol->atom_type == TFITS_BIN_TYPE_X) {
         }
         */
        fitstable_add_read_column_struct(outtab, qcol->atom_type, qcol->atom_nb,
                                         off, qcol->atom_type, qcol->tlabel, TRUE);
        // set the FITS column number.
        col = getcol(outtab, ncols(outtab)-1);
        col->col = noc + i;
        off += fitscolumn_get_size(col); //getcol(tab, ncols(tab)-1));
    }
}

void fitstable_add_fits_columns_as_struct(fitstable_t* tab) {
    int i;
    int off = 0;
    for (i=0; i<tab->table->nc; i++) {
        qfits_col* qcol = tab->table->col + i;
        fitscol_t* col;
        /*
         if (qcol->atom_type == TFITS_BIN_TYPE_X) {
         }
         */
        fitstable_add_read_column_struct(tab, qcol->atom_type, qcol->atom_nb,
                                         off, qcol->atom_type, qcol->tlabel, TRUE);
        // set the FITS column number.
        col = getcol(tab, ncols(tab)-1);
        col->col = i;

        off += fitscolumn_get_size(getcol(tab, ncols(tab)-1));
    }
}

int fitstable_find_fits_column(fitstable_t* tab, const char* colname,
                               char** units, tfits_type* type, int* arraysize) {
    int i;
    for (i=0; i<tab->table->nc; i++) {
        qfits_col* qcol = tab->table->col + i;
        if (!strcaseeq(colname, qcol->tlabel))
            continue;
        // found it!
        if (units)
            *units = qcol->tunit;
        if (type)
            *type = qcol->atom_type;
        if (arraysize)
            *arraysize = qcol->atom_nb;
        return 0;
    }
    return -1;
}


void fitstable_add_read_column_struct(fitstable_t* tab,
                                      tfits_type c_type,
                                      int arraysize,
                                      int structoffset,
                                      tfits_type fits_type,
                                      const char* name,
                                      anbool required) {
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
                                 anbool required) {
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
        printf("  %i: %s: fits type %i, C type %i, arraysize %i, fitssize %i, C size %i, C offset %i (if in a struct), FITS column num: %i\n",
               i, col->colname, col->fitstype, col->ctype, col->arraysize, col->fitssize, col->csize, col->coffset, col->col);
    }
}

int fitstable_read_structs(fitstable_t* tab, void* struc,
                           int strucstride, int offset, int N) {
    int i;
    void* tempdata = NULL;
    int highwater = 0;

    //printf("fitstable_read_structs: stride %i, offset %i, N %i\n",strucstride, offset, N);

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
                ERROR("Number of data items requested exceeds number of rows: offset %i, n %i, nrows %zu", offset, N, bl_size(tab->rows));
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
static int write_one(fitstable_t* table, const void* struc, anbool flip,
                     va_list* ap) {
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
                columndata = va_arg(*ap, void *);
        }
        // If "columndata" is NULL, fits_write_data_array
        // skips the required number of bytes.
        // This allows both structs and normal columns to coexist
        // (in theory -- is this ever used?)
        // (yes, by onefield.c when writing rdls and correspondence files
        //  with tag-along data...)

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
                                        col->fitstype, col->arraysize, flip);
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
    return write_one(table, struc, TRUE, NULL);
}

int fitstable_write_struct_noflip(fitstable_t* table, const void* struc) {
    return write_one(table, struc, FALSE, NULL);
}


int fitstable_write_structs(fitstable_t* table, const void* struc, int stride, int N) {
    int i;
    char* s = (char*)struc;
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
    ret = write_one(table, NULL, TRUE, &ap);
    va_end(ap);
    return ret;
}

int fitstable_write_row_noflip(fitstable_t* table, ...) {
    int ret;
    va_list ap;
    if (!table->table)
        fitstable_create_table(table);
    va_start(ap, table);
    ret = write_one(table, NULL, FALSE, &ap);
    va_end(ap);
    return ret;
}


int fitstable_write_one_column(fitstable_t* table, int colnum,
                               int rowoffset, int nrows,
                               const void* src, int src_stride) {
    anbool flip = TRUE;
    off_t foffset = 0;
    off_t start = 0;
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
                   src, (size_t)col->fitssize * (size_t)col->arraysize);
            src = ((const char*)src) + src_stride;
        }
    } else {
        for (i=0; i<nrows; i++) {
            if (fseeko(table->fid, start + (size_t)i * (size_t)table->table->tab_w, SEEK_SET) ||
                fits_write_data_array(table->fid, src, col->fitstype, col->arraysize, flip)) {
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

/**
 If "inds" is non-NULL, it's a list of indices to read.
 */
static void* read_array_into(const fitstable_t* tab,
                             const char* colname, tfits_type ctype,
                             anbool array_ok,
                             int offset, const int* inds, int Nread,
                             void* dest, int deststride,
                             int desired_arraysize,
                             int* p_arraysize) {
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
        ERROR("Column \"%s\" not found in FITS table %s", colname, tab->fn);
        return NULL;
    }
    col = tab->table->col + colnum;
    if (!array_ok && (col->atom_nb != 1)) {
        ERROR("Column \"%s\" in FITS table %s is an array of size %i, not a scalar",
              colname, tab->fn, col->atom_nb);
        return NULL;
    }

    arraysize = col->atom_nb;
    if (p_arraysize)
        *p_arraysize = arraysize;
    if (desired_arraysize && arraysize != desired_arraysize) {
        ERROR("Column \"%s\" has array size %i but you wanted %i", colname, arraysize, desired_arraysize);
        return NULL;
    }
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
        cdata = calloc((size_t)Nread * (size_t)arraysize, csize);

    if (dest && deststride > 0)
        cstride = deststride;
    else
        cstride = csize * arraysize;

    fitsstride = fitssize * arraysize;
    if (csize < fitssize) {
        // Need to allocate a bigger temp array and down-convert the data.
        // HACK - could set data=tempdata and realloc after (if 'dest' is NULL)
        tempdata = calloc((size_t)Nread * (size_t)arraysize, fitssize);
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
            ERROR("Number of data items requested exceeds number of rows: offset %i, n %i, nrows %zu", offset, Nread, bl_size(tab->rows));
            return NULL;
        }
        off = fits_offset_of_column(tab->table, colnum);
        sz = fitsstride;
        if (inds) {
            for (i=0; i<Nread; i++)
                memcpy(fitsdata + i * fitsstride,
                       ((char*)bl_access(tab->rows, inds[i])) + off,
                       sz);
        } else {
            for (i=0; i<Nread; i++)
                memcpy(fitsdata + i * fitsstride,
                       ((char*)bl_access(tab->rows, offset+i)) + off,
                       sz);
        }
    } else {
        int res;
        if (inds) {
            res = qfits_query_column_seq_to_array_inds(tab->table, colnum, inds, Nread,
                                                       (unsigned char*)fitsdata, fitsstride);
        } else {
            res = qfits_query_column_seq_to_array(tab->table, colnum, offset, Nread,
                                                  (unsigned char*)fitsdata, fitsstride);
        }
        if (res) {
            ERROR("Failed to read column from FITS file");
            // MEMLEAK!
            return NULL;
        }
    }

    if (fitstype != ctype) {
        if (csize <= fitssize) {
            // work forward
            fits_convert_data(cdata, cstride, ctype,
                              fitsdata, fitsstride, fitstype,
                              arraysize, Nread);
        } else {
            // work backward from the end of the array
            fits_convert_data(cdata + (((off_t)Nread*(off_t)arraysize)-1) * (off_t)csize,
                              -csize, ctype,
                              fitsdata + (((off_t)Nread*(off_t)arraysize)-1) * (off_t)fitssize,
                              -fitssize, fitstype,
                              1, (size_t)Nread * (size_t)arraysize);
        }
    }

    free(tempdata);
    return cdata;
}

static void* read_array(const fitstable_t* tab,
                        const char* colname, tfits_type ctype,
                        anbool array_ok, int offset, int Nread) {
    return read_array_into(tab, colname, ctype, array_ok, offset, NULL, Nread, NULL, 0, 0, NULL);
}

int fitstable_read_column_inds_into(const fitstable_t* tab,
                                    const char* colname, tfits_type read_as_type,
                                    void* dest, int stride, const int* inds, int N) {
    return (read_array_into(tab, colname, read_as_type, FALSE, 0, inds, N, dest, stride, 0, NULL)
            == NULL ? -1 : 0);
}

void* fitstable_read_column_inds(const fitstable_t* tab,
                                 const char* colname, tfits_type read_as_type,
                                 const int* inds, int N) {
    return read_array_into(tab, colname, read_as_type, FALSE, 0, inds, N, NULL, 0, 0, NULL);
}

int fitstable_read_column_array_inds_into(const fitstable_t* tab,
                                          const char* colname, tfits_type read_as_type,
                                          void* dest, int stride, int arraysize,
                                          const int* inds, int N) {
    return (read_array_into(tab, colname, read_as_type, TRUE, 0, inds, N, dest, stride, arraysize, NULL)
            == NULL ? -1 : 0);
}

void* fitstable_read_column_array_inds(const fitstable_t* tab,
                                       const char* colname, tfits_type read_as_type,
                                       const int* inds, int N, int* arraysize) {
    return read_array_into(tab, colname, read_as_type, TRUE, 0, inds, N, NULL, 0, 0, arraysize);
}

int fitstable_read_column_offset_into(const fitstable_t* tab,
                                      const char* colname, tfits_type read_as_type,
                                      void* dest, int stride, int start, int N) {
    return (read_array_into(tab, colname, read_as_type, FALSE, start, NULL, N, dest, stride, 0, NULL)
            == NULL ? -1 : 0);
}

int fitstable_read_column_into(const fitstable_t* tab,
                               const char* colname, tfits_type read_as_type,
                               void* dest, int stride) {
    return fitstable_read_column_offset_into(tab, colname, read_as_type, dest, stride, 0, -1);
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

static
fitstable_t* _fitstable_open(const char* fn) {
    fitstable_t* tab;
    tab = fitstable_new();
    if (!tab) {
        ERROR("Failed to allocate new FITS table structure");
        goto bailout;
    }
    tab->extension = 1;
    tab->fn = strdup_safe(fn);

    tab->anq = anqfits_open(fn);
    if (!tab->anq) {
        ERROR("Failed to open FITS file \"%s\"", fn);
        goto bailout;
    }
    tab->primheader = anqfits_get_header(tab->anq, 0);
    if (!tab->primheader) {
        ERROR("Failed to read primary FITS header from %s", fn);
        goto bailout;
    }
    return tab;
 bailout:
    if (tab) {
        fitstable_close(tab);
    }
    return NULL;
}

fitstable_t* fitstable_open(const char* fn) {
    fitstable_t* tab = _fitstable_open(fn);
    if (!tab) {
        return NULL;
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

fitstable_t* fitstable_open_mixed(const char* fn) {
    return _fitstable_open(fn);
}

fitstable_t* fitstable_open_extension_2(const char* fn, int ext) {
    fitstable_t* tab = _fitstable_open(fn);
    if (!tab)
        return tab;
    if (fitstable_open_extension(tab, ext)) {
        fitstable_close(tab);
        return NULL;
    }
    return tab;
}

static fitstable_t* open_for_writing(const char* fn, const char* mode, FILE* fid) {
    fitstable_t* tab;
    tab = fitstable_new();
    if (!tab)
        goto bailout;
    tab->fn = strdup_safe(fn);
    if (fid)
        tab->fid = fid;
    else {
        tab->fid = fopen(fn, mode);
        if (!tab->fid) {
            SYSERROR("Couldn't open output file %s for writing", fn);
            goto bailout;
        }
    }
    return tab;
 bailout:
    if (tab) {
        fitstable_close(tab);
    }
    return NULL;
}

fitstable_t* fitstable_open_for_writing(const char* fn) {
    fitstable_t* tab = open_for_writing(fn, "wb", NULL);
    if (!tab)
        return tab;
    tab->primheader = qfits_table_prim_header_default();
    return tab;
}

fitstable_t* fitstable_open_for_appending(const char* fn) {
    fitstable_t* tab = open_for_writing(fn, "r+b", NULL);
    if (!tab)
        return tab;
    if (fseeko(tab->fid, 0, SEEK_END)) {
        SYSERROR("Failed to seek to end of file");
        fitstable_close(tab);
        return NULL;
    }
    tab->primheader = anqfits_get_header2(fn, 0);
    if (!tab->primheader) {
        ERROR("Failed to read primary FITS header from %s", fn);
        fitstable_close(tab);
        return NULL;
    }
    return tab;
}

fitstable_t* fitstable_open_for_appending_to(FILE* fid) {
    fitstable_t* tab = open_for_writing(NULL, NULL, fid);
    if (!tab)
        return tab;
    if (fseeko(tab->fid, 0, SEEK_END)) {
        SYSERROR("Failed to seek to end of file");
        fitstable_close(tab);
        return NULL;
    }
    return tab;
}

int fitstable_append_to(fitstable_t* intable, FILE* fid) {
    fitstable_t* outtable;
    qfits_header* tmphdr;
    outtable = fitstable_open_for_appending_to(fid);
    fitstable_clear_table(intable);
    fitstable_add_fits_columns_as_struct(intable);
    fitstable_copy_columns(intable, outtable);
    outtable->table = fits_copy_table(intable->table);
    outtable->table->nr = 0;
    // swap in the input header.
    tmphdr = outtable->header;
    outtable->header = intable->header;
    if (fitstable_write_header(outtable)) {
        ERROR("Failed to write output table header");
        return -1;
    }
    if (fitstable_copy_rows_data(intable, NULL, fitstable_nrows(intable), outtable)) {
        ERROR("Failed to copy rows from input table to output");
        return -1;
    }
    if (fitstable_fix_header(outtable)) {
        ERROR("Failed to fix output table header");
        return -1;
    }
    outtable->header = tmphdr;
    // clear this so that fitstable_close() doesn't fclose() it.
    outtable->fid = NULL;
    fitstable_close(outtable);
    return 0;
}

int fitstable_close(fitstable_t* tab) {
    int i;
    int rtn = 0;
    if (!tab) return 0;
    if (is_writing(tab)) {
        if (tab->fid) {
            if (fclose(tab->fid)) {
                SYSERROR("Failed to close output file %s", tab->fn);
                rtn = -1;
            }
        }
    }
    if (tab->anq) {
        anqfits_close(tab->anq);
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

static fitscol_t* fitstable_add_column(fitstable_t* tab, fitscol_t* col) {
    col = bl_append(tab->cols, col);
    col->csize = fits_get_atom_size(col->ctype);
    col->fitssize = fits_get_atom_size(col->fitstype);
    return col;
}

/*
 static void fitstable_add_columns(fitstable_t* tab, fitscol_t* cols, int Ncols) {
 int i;
 for (i=0; i<Ncols; i++) {
 fitstable_add_column(tab, cols + i);
 }
 }
 */

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
            ERROR("Table has only %zu extensions, but you requested #%i",
                  bl_size(tab->extensions), ext);
            return -1;
        }
        theext = bl_access(tab->extensions, ext-1);
        tab->table = theext->table;
        tab->header = theext->header;
        tab->rows = theext->rows;
        tab->extension = ext;

    } else {
        if (tab->table) {
            qfits_table_close(tab->table);
            tab->table = NULL;
        }

        assert(tab->anq);
        if (ext >= anqfits_n_ext(tab->anq)) {
            ERROR("Requested FITS extension %i in file %s, but there are only %i extensions.\n", ext, tab->fn, anqfits_n_ext(tab->anq));
            return -1;
        }
        tab->table = anqfits_get_table(tab->anq, ext);

        if (!tab->table) {
            ERROR("FITS extension %i in file %s is not a table (or there was an error opening the file)", ext, tab->fn);
            return -1;
        }
        if (tab->header) {
            qfits_header_destroy(tab->header);
        }

        tab->header = anqfits_get_header(tab->anq, ext);
        if (!tab->header) {
            ERROR("Couldn't get header for FITS extension %i in file %s", ext, tab->fn);
            return -1;
        }
        tab->extension = ext;
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

    //printf("fitstable_fix_header: ext %i, table offset was %lu, fn %s\n",
    //t->extension, (long)t->table_offset, t->fn);
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

int fitstable_nrows(const fitstable_t* t) {
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
    //logverb("fitstable.c:refill_buffer: offset %i, n %i\n", offset, n);
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

