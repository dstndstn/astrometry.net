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

// is the given table name one of the above strings?
int kdtree_fits_column_is_kdtree(char* columnname) {
    return
        starts_with(columnname, KD_STR_HEADER) ||
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

const fitsbin_t* get_fitsbin_const(const kdtree_fits_t* io) {
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
KD_DECLARE(kdtree_write_fits, int, (kdtree_fits_t* io, const kdtree_t* kd,
                                    const qfits_header* inhdr));
/*
 sl* kdtree_fits_list_trees(kdtree_fits_t* io) {
    sl* s = sl_new(4);
    fitsbin_t* fb = kdtree_fits_get_fitsbin(io);
	qfits_header* header;
    int ndim, ndata, nnodes;
	unsigned int tt;
    char* fn = fb->filename;
    int i, N;

    // Look in the primary header...
    header = fitsbin_get_primary_header(fb);
    if (is_tree_header_ok(header, &ndim, &ndata, &nnodes, &tt, 1)) {
        sl_append(s, NULL);
    }
    N = qfits_query_n_ext(fn);
    for (i=1; i<=N; i++) {
        char* name;
        header = qfits_header_readext(fn, i);
        if (!header) {
            ERROR("Failed to read FITS header for extension %i in file %s", i, fn);
            continue;
        }
        name = fits_get_dupstring(header, "KDT_NAME");
        if (!name)
            goto next;
        if (!is_tree_header_ok(header, &ndim, &ndata, &nnodes, &tt, 0)) {
            free(name);
            goto next;
        }
        sl_append(s, name);
    next:
        qfits_header_destroy(header);
    }
    return s;
}
 */

static qfits_header* find_tree(const char* treename, const fitsbin_t* fb,
                               int* ndim, int* ndata, int* nnodes,
                               unsigned int* tt, char** realname) {
    qfits_header* header;
    int i, nexten;
    char* fn = fb->filename;

    if (!treename) {
        // Look in the primary header...
        header = fitsbin_get_primary_header(fb);
        if (is_tree_header_ok(header, ndim, ndata, nnodes, tt, 1)) {
            header = qfits_header_copy(header);
            *realname = NULL;
            return header;
        }
    }
    // scan the extension headers, looking for one that contains a matching KDT_NAME entry.
    nexten = qfits_query_n_ext(fn);
    header = NULL;
    for (i=1; i<=nexten; i++) {
        char* name;
        header = qfits_header_readext(fn, i);
        if (!header) {
            ERROR("Failed to read FITS header for extension %i in file %s", i, fn);
            goto next;
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
        if (is_tree_header_ok(header, ndim, ndata, nnodes, tt, 0)) {
            *realname = name;
            return header;
        }
    next:
        qfits_header_destroy(header);
    }
    return NULL;
}

int kdtree_fits_contains_tree(const kdtree_fits_t* io, const char* treename) {
    int ndim, ndata, nnodes;
	unsigned int tt;
    qfits_header* hdr;
    int rtn;
    const fitsbin_t* fb = get_fitsbin_const(io);
    char* realname;
    hdr = find_tree(treename, fb, &ndim, &ndata, &nnodes, &tt, &realname);
    rtn = (hdr != NULL);
    if (hdr != NULL)
        qfits_header_destroy(hdr);
    return rtn;
}

kdtree_t* kdtree_fits_read_tree(kdtree_fits_t* io, const char* treename,
                                qfits_header** p_hdr) {
    int ndim, ndata, nnodes;
	unsigned int tt;
	kdtree_t* kd = NULL;
    fitsbin_t* fb = kdtree_fits_get_fitsbin(io);
	qfits_header* header;
    int rtn;
    char* fn = fb->filename;

    kd = CALLOC(1, sizeof(kdtree_t));
    if (!kd) {
		SYSERROR("Couldn't allocate kdtree");
		return NULL;
    }

    header = find_tree(treename, fb, &ndim, &ndata, &nnodes, &tt, &kd->name);
    if (!header) {
        // Not found.
        ERROR("Kdtree matching \"%s\" not found in file %s", treename, fn);
        FREE(kd);
        return NULL;
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

    kd->io = io;

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

// just writes the tree, no primary header.
int kdtree_fits_append_tree(kdtree_fits_t* io, const kdtree_t* kd,
                            const qfits_header* inhdr) {
    int rtn;
	KD_DISPATCH(kdtree_write_fits, kd->treetype, rtn = , (io, kd, inhdr));
    return rtn;
}

// just writes the tree, no primary header.
int kdtree_fits_write_primary_header(kdtree_fits_t* io,
                                     const qfits_header* inhdr) {
    fitsbin_t* fb;
    qfits_header* hdr;
    fb = kdtree_fits_get_fitsbin(io);
    if (inhdr) {
        hdr = fitsbin_get_primary_header(fb);
        fits_copy_all_headers(inhdr, hdr, NULL);
    }
    return fitsbin_write_primary_header(fb);
}


int kdtree_fits_write_tree(kdtree_fits_t* io, const kdtree_t* kd,
                           const qfits_header* inhdr) {
    return (kdtree_fits_write_primary_header(io, NULL) ||
            kdtree_fits_append_tree(io, kd, inhdr));
}

int kdtree_fits_io_close(kdtree_fits_t* io) {
    fitsbin_t* fb;
    fb = kdtree_fits_get_fitsbin(io);
    return fitsbin_close(fb);
}

int kdtree_fits_close(kdtree_t* kd) {
    if (!kd) return 0;
    // FIXME - erm, this will double-free if we try to read and then free
    // multiple kdtrees from one file...  reference count??
	if (kd->io)
        kdtree_fits_io_close(kd->io);
    FREE(kd->name);
	FREE(kd);
    return 0;
}
