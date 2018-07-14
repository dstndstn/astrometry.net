/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

#include "kdtree_fits_io.h"
#include "kdtree_internal.h"
#include "kdtree_mem.h"
#include "fitsioutils.h"
#include "anqfits.h"
#include "ioutils.h"
#include "errors.h"
#include "fitsbin.h"
#include "tic.h"
#include "log.h"

// is the given table name one of the above strings?
int kdtree_fits_column_is_kdtree(char* columnname) {
    return
        starts_with(columnname, KD_STR_HEADER) ||
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
    char str[FITS_LINESZ+1];

    if (oldstyle) {
        *ndim   = qfits_header_getint(header, "NDIM", -1);
        *ndata  = qfits_header_getint(header, "NDATA", -1);
        *nnodes = qfits_header_getint(header, "NNODES", -1);
    } else {
        *ndim   = qfits_header_getint(header, "KDT_NDIM", -1);
        *ndata  = qfits_header_getint(header, "KDT_NDAT", -1);
        *nnodes = qfits_header_getint(header, "KDT_NNOD", -1);
    }
    qfits_pretty_string_r(qfits_header_getstr(header, "KDT_EXT"), str);
    ext_type = kdtree_kdtype_parse_ext_string(str);
    qfits_pretty_string_r(qfits_header_getstr(header, "KDT_INT"), str);
    int_type = kdtree_kdtype_parse_tree_string(str);
    qfits_pretty_string_r(qfits_header_getstr(header, "KDT_DATA"), str);
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
        if (treename)
            ERROR("Failed to read kdtree named \"%s\" from file %s", treename, fn);
        else
            ERROR("Failed to read kdtree from file %s", fn);
        kdtree_fits_io_close(io);
        return NULL;
    }
    return kd;
}

static int write_convenience(const kdtree_t* kd, const char* fn,
                             const qfits_header* hdr, anbool flipped) {
    kdtree_fits_t* io;
    int rtn;
    io = kdtree_fits_open_for_writing(fn);
    if (!io) {
        ERROR("Failed to open file %s for writing", fn);
        return -1;
    }
    if (flipped)
        rtn = kdtree_fits_write_tree_flipped(io, kd, hdr);
    else
        rtn = kdtree_fits_write_tree(io, kd, hdr);
    kdtree_fits_io_close(io);
    if (rtn) {
        ERROR("Failed to write kdtree to file %s", fn);
    }
    return rtn;
}

int kdtree_fits_write_flipped(const kdtree_t* kdtree, const char* fn,
                              const qfits_header* hdr) {
    return write_convenience(kdtree, fn, hdr, TRUE);
}

int kdtree_fits_write(const kdtree_t* kdtree, const char* fn,
                      const qfits_header* hdr) {
    return write_convenience(kdtree, fn, hdr, FALSE);
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

kdtree_fits_t* kdtree_fits_open_fits(anqfits_t* fits) {
    return fitsbin_open_fits(fits);
}

kdtree_fits_t* kdtree_fits_open_for_writing(const char* fn) {
    return fitsbin_open_for_writing(fn);
}

qfits_header* kdtree_fits_get_primary_header(kdtree_fits_t* io) {
    return fitsbin_get_primary_header(kdtree_fits_get_fitsbin(io));
}

int kdtree_fits_read_chunk(kdtree_fits_t* io, fitsbin_chunk_t* chunk) {
    int rtn;
    //double t0 = timenow();
    rtn = fitsbin_read_chunk(io, chunk);
    //debug("kdtree_fits_read_chunk(%s) took %g ms\n", chunk->tablename, 1000. * (timenow() - t0));
    return rtn;
}

// declarations
KD_DECLARE(kdtree_read_fits, int, (kdtree_fits_t* io, kdtree_t* kd));
KD_DECLARE(kdtree_write_fits, int, (kdtree_fits_t* io, const kdtree_t* kd,
                                    const qfits_header* inhdr, anbool flip_endian,
                                    FILE* fid));

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

    // treat empty treename as NULL...
    if (treename && !treename[0])
        treename = NULL;

    // scan the extension headers, looking for one that contains a matching KDT_NAME entry.
    nexten = fitsbin_n_ext(fb);
    header = NULL;
    for (i=1; i<nexten; i++) {
        char* name;
        header = fitsbin_get_header(fb, i);
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

	// if the desired treename was specified, it must match;
	// bail out if this is not the case.
	// (if treename is NULL then anything matches.)
	if (treename && !(name && (strcmp(name, treename) == 0))) {
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
    char* realname = NULL;
    hdr = find_tree(treename, fb, &ndim, &ndata, &nnodes, &tt, &realname);
    free(realname);
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
    int rtn = 0;
    char* fn = fb->filename;
    //double t0;

    kd = CALLOC(1, sizeof(kdtree_t));
    if (!kd) {
        SYSERROR("Couldn't allocate kdtree");
        return NULL;
    }

    header = find_tree(treename, fb, &ndim, &ndata, &nnodes, &tt, &kd->name);
    if (!header) {
        // Not found.
        if (treename)
            ERROR("Kdtree header for a tree named \"%s\" was not found in file %s", treename, fn);
        else
            ERROR("Kdtree header was not found in file %s", fn);

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

    //t0 = timenow();
    KD_DISPATCH(kdtree_read_fits, tt, rtn = , (io, kd));
    //debug("kdtree_read_fits(%s) took %g ms\n", fn, 1000. * (timenow() - t0));

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
    fitsbin_t* fb = kdtree_fits_get_fitsbin(io);
    if (fitsbin_write_chunk(fb, chunk)) {
        ERROR("Failed to write kdtree extra chunk");
        return -1;
    }
    return 0;
}

int kdtree_fits_write_chunk_to(fitsbin_chunk_t* chunk, FILE* fid) {
    if (fitsbin_write_chunk_to(NULL, chunk, fid)) {
        ERROR("Failed to write kdtree extra chunk");
        return -1;
    }
    return 0;
}

int kdtree_fits_write_chunk_flipped(kdtree_fits_t* io, fitsbin_chunk_t* chunk,
                                    int wordsize) {
    fitsbin_t* fb = kdtree_fits_get_fitsbin(io);
    if (fitsbin_write_chunk_flipped(fb, chunk, wordsize)) {
        ERROR("Failed to write (flipped) kdtree extra chunk");
        return -1;
    }
    return 0;
}

// just writes the tree, no primary header.
int kdtree_fits_append_tree(kdtree_fits_t* io, const kdtree_t* kd,
                            const qfits_header* inhdr) {
    int rtn = -1;
    KD_DISPATCH(kdtree_write_fits, kd->treetype, rtn = , (io, kd, inhdr, FALSE, NULL));
    return rtn;
}

int kdtree_fits_append_tree_to(kdtree_t* kd,
                               const qfits_header* inhdr,
                               FILE* fid) {
    int rtn = -1;
    KD_DISPATCH(kdtree_write_fits, kd->treetype, rtn = , (NULL, kd, inhdr, FALSE, fid));
    return rtn;
}

int kdtree_fits_append_tree_flipped(kdtree_fits_t* io, const kdtree_t* kd,
                                    const qfits_header* inhdr) {
    int rtn = -1;
    KD_DISPATCH(kdtree_write_fits, kd->treetype, rtn = , (io, kd, inhdr, TRUE, NULL));
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

int kdtree_fits_write_tree_flipped(kdtree_fits_t* io, const kdtree_t* kd,
                                   const qfits_header* inhdr) {
    return (kdtree_fits_write_primary_header(io, NULL) ||
            kdtree_fits_append_tree_flipped(io, kd, inhdr));
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
