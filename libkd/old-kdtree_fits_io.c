/*
  This file is part of libkd.
  Copyright 2006-2008 Dustin Lang and Keir Mierle.

  libkd is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 2.

  libkd is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with libkd; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/param.h>

#include "kdtree_fits_io.h"
#include "kdtree_internal.h"
#include "kdtree_mem.h"
#include "fitsioutils.h"
#include "qfits.h"
#include "ioutils.h"
#include "errors.h"

#define KDTREE_FITS_VERSION 1

// is the given table name one of the above strings?
int kdtree_fits_column_is_kdtree(char* columnname) {
    return
        (strcmp(columnname, KD_STR_HEADER) == 0) ||
        starts_with(columnname, KD_STR_NODES) ||
        starts_with(columnname, KD_STR_LR   ) ||
        starts_with(columnname, KD_STR_PERM ) ||
        starts_with(columnname, KD_STR_BB   ) ||
        starts_with(columnname, KD_STR_SPLIT) ||
        starts_with(columnname, KD_STR_SPLITDIM) ||
        starts_with(columnname, KD_STR_DATA ) ||
        starts_with(columnname, KD_STR_RANGE);
}

kdtree_t* kdtree_fits_read(const char* fn, const char* treename, qfits_header** p_hdr) {
	return kdtree_fits_read_extras(fn, treename, p_hdr, NULL, 0);
}

static int is_tree_header_ok(qfits_header* header, int* ndim, int* ndata,
                             int* nnodes, unsigned int* treetype, int oldstyle) {
	unsigned int ext_type, int_type, data_type;
    char* str;
    if (oldstyle) {
        *ndim   = qfits_header_getint(header, "NDIM", -1);
        *ndata  = qfits_header_getint(header, "NDATA", -1);
        *nnodes = qfits_header_getint(header, "NNODES", -1);
    } else {
        *ndim   = qfits_header_getint(header, "KDT_NDIM", -1);
        *ndata  = qfits_header_getint(header, "KDT_NDAT", -1);
        *nnodes = qfits_header_getint(header, "KDT_NNOD", -1);
    }
    str = qfits_pretty_string(qfits_header_getstr(header, "KDT_EXT"));
    ext_type = kdtree_kdtype_parse_ext_string(str);
    str = qfits_pretty_string(qfits_header_getstr(header, "KDT_INT"));
    int_type = kdtree_kdtype_parse_tree_string(str);
    str = qfits_pretty_string(qfits_header_getstr(header, "KDT_DATA"));
    data_type = kdtree_kdtype_parse_data_string(str);

    // default: external world is doubles.
	if (ext_type == KDT_NULL)
		ext_type = KDT_EXT_DOUBLE;

	*treetype = kdtree_kdtypes_to_treetype(ext_type, int_type, data_type);

    if ((*ndim > -1) && (*ndata > -1) && (*nnodes > -1) &&
        (int_type != KDT_NULL) && (data_type != KDT_NULL) &&
        (fits_check_endian(header) == 0)) {
        return 1;
    }
    return 0;
}

// declarations
KD_DECLARE(kdtree_read_fits, int, (const char* fn, kdtree_t* kd, extra_table* uextras, int nuextras));

/**
   This function reads FITS headers to try to determine which kind of tree
   is contained in the file, then calls the appropriate (mangled) function
   kdtree_read_fits() (defined in kdtree_internal_fits.c).  This, in turn,
   calls kdtree_fits_common_read(), then does some extras processing and
   returns.
 */
kdtree_t* kdtree_fits_read_extras(const char* fn, const char* treename, qfits_header** p_hdr, extra_table* extras, int nextras) {
	qfits_header* header;
    int ndim, ndata, nnodes;
	unsigned int tt;
	kdtree_t* kdtree = NULL;
    int found = 0;
    int rtn = -1;

	if (!qfits_is_fits(fn)) {
		ERROR("Kdtree file %s doesn't look like a FITS file", fn);
		return NULL;
	}

    kdtree = CALLOC(1, sizeof(kdtree_t));
    if (!kdtree) {
		SYSERROR("Couldn't allocate kdtree");
		return NULL;
    }

    if (!treename) {
        // Look in the primary header...
        header = qfits_header_read(fn);
        if (!header) {
            ERROR("Couldn't read FITS header from %s", fn);
            free(kdtree);
            return NULL;
        }
        if (is_tree_header_ok(header, &ndim, &ndata, &nnodes, &tt, 1)) {
            found = 1;
        } else {
            qfits_header_destroy(header);
        }
    }
    if (!found) {
        int i, nexten;
        // scan the extension headers, looking for one that contains a matching KDT_NAME entry.
        nexten = qfits_query_n_ext(fn);
        header = NULL;
        for (i=1; i<=nexten; i++) {
            char* name;
            header = qfits_header_readext(fn, i);
            if (!header) {
                ERROR("Failed to read FITS header for extension %i in file %s", i, fn);
                return NULL;
            }
            name = fits_get_dupstring(header, "KDT_NAME");
            if (!name)
                continue;
            //printf("Found KDT_NAME entry \"%s\".\n", name);
            if (name && !name[0]) {
                // treat empty string as NULL.
                free(name);
                name = NULL;
            }
            // if the desired treename was specified and this one doesn't match...
            if (treename && strcmp(name, treename)) {
                free(name);
                continue;
            }

            if (is_tree_header_ok(header, &ndim, &ndata, &nnodes, &tt, 0)) {
                kdtree->name = name;
                break;
            }
            qfits_header_destroy(header);
        }
        if (i > nexten) {
            // Not found.
            ERROR("Kdtree named \"%s\" not found in file %s", treename, fn);
            FREE(kdtree);
            return NULL;
        }
    }

    kdtree->has_linear_lr = qfits_header_getboolean(header, "KDT_LINL", 0);

    if (p_hdr)
        *p_hdr = header;
    else
        qfits_header_destroy(header);

    kdtree->ndata  = ndata;
    kdtree->ndim   = ndim;
    kdtree->nnodes = nnodes;
	kdtree->nbottom = (nnodes+1)/2;
	kdtree->ninterior = nnodes - kdtree->nbottom;
    kdtree->nlevels = kdtree_nnodes_to_nlevels(nnodes);
	kdtree->treetype = tt;

	KD_DISPATCH(kdtree_read_fits, tt, rtn = , (fn, kdtree, extras, nextras));

    if (rtn) {
        FREE(kdtree->name);
        FREE(kdtree);
        return NULL;
    }

	return kdtree;
}

KD_DECLARE(kdtree_append_fits, int, (const kdtree_t* kd, const qfits_header* hdr, const extra_table* ue, int nue, FILE* out));

/**
   This function calls the appropriate (mangled) function
   kdtree_append_fits(), which in turn calls kdtree_fits_common_write()
   which does the actual writing.
 */
int kdtree_fits_append_extras(const kdtree_t* kd, const qfits_header* hdr,
                              const extra_table* extras, int nextras,
                              FILE* out) {
	int rtn = -1;
	KD_DISPATCH(kdtree_append_fits, kd->treetype, rtn = , (kd, hdr, extras, nextras, out));
	return rtn;
}

int kdtree_fits_append(const kdtree_t* kdtree, const qfits_header* hdr, FILE* out) {
    return kdtree_fits_append_extras(kdtree, hdr, NULL, 0, out);
}

FILE* kdtree_fits_write_primary_header(const char* fn) {
    FILE* fout;
    qfits_header* header;

    fout = fopen(fn, "wb");
    if (!fout) {
        SYSERROR("Failed to open file %s for writing", fn);
        return NULL;
    }

    header = qfits_table_prim_header_default();
    qfits_header_dump(header, fout);
    qfits_header_destroy(header);

    return fout;
}

int kdtree_fits_write_extras(const kdtree_t* kdtree, const char* fn, const qfits_header* hdr, const extra_table* extras, int nextras) {
    int rtn;
    FILE* fout;

    fout = kdtree_fits_write_primary_header(fn);

    rtn = kdtree_fits_append_extras(kdtree, hdr, extras, nextras, fout);
    if (rtn) {
        return rtn;
    }
    if (fclose(fout)) {
        SYSERROR("Failed to close file %s after writing", fn);
        return -1;
    }
    return 0;
}

int kdtree_fits_write(const kdtree_t* kdtree, const char* fn, const qfits_header* hdr) {
	return kdtree_fits_write_extras(kdtree, fn, hdr, NULL, 0);
}

int kdtree_fits_common_read(const char* fn, kdtree_t* kdtree, extra_table* extras, int nextras) {
	FILE* fid;
	int size;
	unsigned char* map;
	int i;

	for (i=0; i<nextras; i++) {
		extra_table* tab = extras + i;
		if (fits_find_table_column(fn, tab->name, &tab->offset, &tab->size, NULL)) {
			if (tab->required) {
				ERROR("Failed to find table %s in file %s", tab->name, fn);
				return -1;
			}
			tab->found = 0;
		} else
			tab->found = 1;
	}

	size = 0;
	for (i=0; i<nextras; i++) {
		extra_table* tab = extras + i;
		int tablesize;
		int col;
		qfits_table* table;
		int ds;

		if (!tab->found)
			continue;
		if (tab->compute_tablesize)
			tab->compute_tablesize(kdtree, tab);

		table = fits_get_table_column(fn, tab->name, &col);
		if (tab->nitems) {
			if (tab->nitems != table->nr) {
				ERROR("Table %s in file %s: expected %i data items, found %i",
						tab->name, fn, tab->nitems, table->nr);
				qfits_table_close(table);
				return -1;
			}
		} else {
			tab->nitems = table->nr;
		}
		ds = table->col[col].atom_nb * table->col[col].atom_size;
		if (tab->datasize) {
			if (tab->datasize != ds) {
				ERROR("Table %s in file %s: expected data size %i, found %i",
						tab->name, fn, tab->datasize, ds);
				qfits_table_close(table);
				return -1;
			}
		} else {
			tab->datasize = ds;
		}
		qfits_table_close(table);

		tablesize = tab->datasize * tab->nitems;
		if (fits_bytes_needed(tablesize) != tab->size) {
			ERROR("The size of table %s in file %s doesn't jive with what's expected: %i vs %i",
					tab->name, fn, fits_bytes_needed(tablesize), tab->size);
			return -1;
		}

		size = MAX(size, tab->offset + tab->size);
	}

	// launch!
	fid = fopen(fn, "rb");
	if (!fid) {
		SYSERROR("Couldn't open file %s to read kdtree", fn);
		return -1;
	}
	map = mmap(0, size, PROT_READ, MAP_SHARED, fileno(fid), 0);
	fclose(fid);
	if (map == MAP_FAILED) {
		SYSERROR("Couldn't mmap kdtree file %s", fn);
		return -1;
	}

	kdtree->mmapped = map;
	kdtree->mmapped_size = size;

	for (i=0; i<nextras; i++) {
		extra_table* tab = extras + i;
		if (!tab->found)
			continue;
		tab->ptr = (map + tab->offset);
	}

    kdtree_update_funcs(kdtree);

	return 0;
}

int kdtree_fits_common_write(const kdtree_t* kdtree, const qfits_header* inhdr, const extra_table* extras, int nextras, FILE* out) {
	int i;
    qfits_table* table;
    qfits_header* tablehdr;
    int ncols, nrows;
    int tablesize;

    nrows = tablesize = 0;
    ncols = 1;
    table = qfits_table_new("", QFITS_BINTABLE, tablesize, ncols, nrows);
    qfits_col_fill(table->col, 0, 0, 1, TFITS_BIN_TYPE_A,
                   KD_STR_HEADER, "", "", "", 0, 0, 0, 0, 0);
    tablehdr = qfits_table_ext_header_default(table);

	if (inhdr)
        // FIXME - don't copy headers that conflict with the ones used by this routine!
        fits_copy_all_headers(inhdr, tablehdr, NULL);
    fits_add_endian(tablehdr);
    fits_header_addf   (tablehdr, "KDT_NAME", "kdtree: name of this tree", "'%s'", kdtree->name ? kdtree->name : "");
    fits_header_add_int(tablehdr, "KDT_NDAT", kdtree->ndata,  "kdtree: number of data points");
    fits_header_add_int(tablehdr, "KDT_NDIM", kdtree->ndim,   "kdtree: number of dimensions");
    fits_header_add_int(tablehdr, "KDT_NNOD", kdtree->nnodes, "kdtree: number of nodes");
    fits_header_add_int(tablehdr, "KDT_VER",  KDTREE_FITS_VERSION, "kdtree: version number");

    qfits_header_dump(tablehdr, out);
    qfits_header_destroy(tablehdr);
    qfits_table_close(table);

	for (i=0; i<nextras; i++) {
		const extra_table* tab;
        int datasize;
        void* dataptr;

        tab = extras + i;
		if (tab->dontwrite)
			continue;
		datasize = tab->datasize;
		dataptr  = tab->ptr;
		ncols = 1;
		nrows = tab->nitems;
		tablesize = datasize * nrows * ncols;
		table = qfits_table_new("", QFITS_BINTABLE, tablesize, ncols, nrows);
		qfits_col_fill(table->col, datasize, 0, 1, TFITS_BIN_TYPE_A,
					   tab->name, "", "", "", 0, 0, 0, 0, 0);
		tablehdr = qfits_table_ext_header_default(table);
		qfits_header_dump(tablehdr, out);
		qfits_header_destroy(tablehdr);
		qfits_table_close(table);
		if ((fwrite(dataptr, 1, tablesize, out) != tablesize) ||
			fits_pad_file(out)) {
			SYSERROR("Failed to write kdtree table %s", tab->name);
			return -1;
		}
	}

    return 0;
}

void kdtree_fits_close(kdtree_t* kd) {
	if (!kd) return;
    FREE(kd->name);
	munmap(kd->mmapped, kd->mmapped_size);
	FREE(kd);
}
