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
#include <errno.h>
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
#include "fitsbin.h"

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

kdtree_t* kdtree_fits_read(const char* fn, const char* treename,
                           qfits_header** p_hdr) {
    kdtree_fits_t* io;
    kdtree_t* kd;
    io = kdtree_fits_open(fn);
	if (!io) {
        ERROR("Failed to open FITS file \"%s\"", fn);
        return NULL;
    }
    kd = kdtree_fits_read_tree(io, treename, p_hdr);
    if (!kd) {
        ERROR("Failed to read kdtree %s from file %s", treename, fn);
        kdtree_fits_io_close(io);
        return NULL;
    }
    kd->io = io;
    return kd;
}

int kdtree_fits_write(const kdtree_t* kd, const char* fn,
                      const qfits_header* hdr) {
    kdtree_fits_t* io;
    int rtn;
    io = kdtree_fits_open_for_writing(fn);
    if (!io) {
        ERROR("Failed to open file %s for writing", fn);
        return -1;
    }
    rtn = kdtree_fits_write_tree(io, kd, hdr);
    kdtree_fits_io_close(io);
    if (rtn) {
        ERROR("Failed to write kdtree to file %s", fn);
    }
    return rtn;
}

/*
bl* kdtree_fits_get_chunks(const kdtree_t* kd) {
    bl* chunks = bl_new(4, sizeof(fitsbin_chunk_t));
    fitsbin_chunk_t chunk;
    qfits_header* hdr;
    fitsbin_chunk_init(&chunk);
    chunk.tablename = "";
    hdr = fitsbin_get_chunk_header(NULL, &chunk);
    fits_add_endian(hdr);
    fits_header_addf   (hdr, "KDT_NAME", "kdtree: name of this tree", "'%s'", kd->name ? kd->name : "");
    fits_header_add_int(hdr, "KDT_NDAT", kd->ndata,  "kdtree: number of data points");
    fits_header_add_int(hdr, "KDT_NDIM", kd->ndim,   "kdtree: number of dimensions");
    fits_header_add_int(hdr, "KDT_NNOD", kd->nnodes, "kdtree: number of nodes");
    fits_header_add_int(hdr, "KDT_VER",  KDTREE_FITS_VERSION, "kdtree: version number");
    bl_append(chunks, &chunk);
}
 */

fitsbin_t* kdtree_fits_get_fitsbin(kdtree_fits_t* io) {
    return io;
}

kdtree_fits_t* kdtree_fits_open(const char* fn) {
    return fitsbin_open(fn);
}

kdtree_fits_t* kdtree_fits_open_for_writing(const char* fn) {
    return fitsbin_open_for_writing(fn);
}

qfits_header* kdtree_fits_get_primary_header(kdtree_fits_t* io) {
    return fitsbin_get_primary_header(kdtree_fits_get_fitsbin(io));
}

int kdtree_fits_read_chunk(kdtree_fits_t* io, fitsbin_chunk_t* chunk) {
    return fitsbin_read_chunk(io, chunk);
}

// declarations
KD_DECLARE(kdtree_read_fits, int, (kdtree_fits_t* io, kdtree_t* kd));
KD_DECLARE(kdtree_write_fits, int, (kdtree_fits_t* io, const kdtree_t* kd));

kdtree_t* kdtree_fits_read_tree(kdtree_fits_t* io, const char* treename,
                                qfits_header** p_hdr) {
    int ndim, ndata, nnodes;
	unsigned int tt;
	kdtree_t* kd = NULL;
    int found = 0;
    fitsbin_t* fb = io;
	qfits_header* header;
    int rtn;

    kd = CALLOC(1, sizeof(kdtree_t));
    if (!kd) {
		SYSERROR("Couldn't allocate kdtree");
		return NULL;
    }

    if (!treename) {
        // Look in the primary header...
        header = fitsbin_get_primary_header(fb);
        if (is_tree_header_ok(header, &ndim, &ndata, &nnodes, &tt, 1)) {
            found = 1;
            header = qfits_header_copy(header);
        }
    }
    if (!found) {
        int i, nexten;
        char* fn = fb->filename;
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
                goto next;
            //printf("Found KDT_NAME entry \"%s\".\n", name);
            if (name && !name[0]) {
                // treat empty string as NULL.
                free(name);
                name = NULL;
            }
            // if the desired treename was specified and this one doesn't match...
            if (treename && strcmp(name, treename)) {
                free(name);
                goto next;
            }

            if (is_tree_header_ok(header, &ndim, &ndata, &nnodes, &tt, 0)) {
                kd->name = name;
                found = 1;
                break;
            }
        next:
            qfits_header_destroy(header);
        }
        if (!found) {
            // Not found.
            ERROR("Kdtree named \"%s\" not found in file %s", treename, fn);
            FREE(kd);
            return NULL;
        }
    }

    kd->has_linear_lr = qfits_header_getboolean(header, "KDT_LINL", 0);

    if (p_hdr)
        *p_hdr = header;
    else
        qfits_header_destroy(header);

    kd->ndata  = ndata;
    kd->ndim   = ndim;
    kd->nnodes = nnodes;
	kd->nbottom = (nnodes+1)/2;
	kd->ninterior = nnodes - kd->nbottom;
    kd->nlevels = kdtree_nnodes_to_nlevels(nnodes);
	kd->treetype = tt;

	KD_DISPATCH(kdtree_read_fits, tt, rtn = , (io, kd));

    if (rtn) {
        FREE(kd->name);
        FREE(kd);
        return NULL;
    }

    kdtree_update_funcs(kd);

	return kd;
    
}

int kdtree_fits_write_chunk(kdtree_fits_t* io, fitsbin_chunk_t* chunk) {
    fitsbin_chunk_t* ch;
    fitsbin_t* fb = kdtree_fits_get_fitsbin(io);
    ch = fitsbin_add_chunk(fb, chunk);
    if (fitsbin_write_chunk(fb, ch)) {
        ERROR("Failed to write kdtree extra chunk");
        return -1;
    }
    return 0;
}

int kdtree_fits_write_tree(kdtree_fits_t* io, const kdtree_t* kd,
                           const qfits_header* inhdr) {
    fitsbin_chunk_t chunk;
    fitsbin_t* fb = kdtree_fits_get_fitsbin(io);
    qfits_header* hdr;
    int rtn;

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "";
    hdr = fitsbin_get_chunk_header(fb, &chunk);

    if (inhdr)
        fits_copy_all_headers(inhdr, hdr, NULL);
    fits_add_endian(hdr);
    fits_header_addf   (hdr, "KDT_NAME", "kdtree: name of this tree", "'%s'", kd->name ? kd->name : "");
    fits_header_add_int(hdr, "KDT_NDAT", kd->ndata,  "kdtree: number of data points");
    fits_header_add_int(hdr, "KDT_NDIM", kd->ndim,   "kdtree: number of dimensions");
    fits_header_add_int(hdr, "KDT_NNOD", kd->nnodes, "kdtree: number of nodes");
    fits_header_add_int(hdr, "KDT_VER",  KDTREE_FITS_VERSION, "kdtree: version number");

    // kdtree header is an empty fitsbin_chunk.
    fitsbin_write_chunk(fb, &chunk);
    fitsbin_chunk_clean(&chunk);

	KD_DISPATCH(kdtree_write_fits, kd->treetype, rtn = , (io, kd));
    return rtn;
}

int kdtree_fits_io_close(kdtree_fits_t* io) {
    fitsbin_t* fb;
    fb = kdtree_fits_get_fitsbin(io);
    return fitsbin_close(fb);
}

int kdtree_fits_close(kdtree_t* kd) {
    kdtree_fits_t* io;
    if (!kd) return 0;
    io = kd->io;
	if (!io) return 0;
    kdtree_fits_io_close(io);
    FREE(kd->name);
	FREE(kd);
    return 0;
}
