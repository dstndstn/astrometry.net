/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "qfits_header.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "ioutils.h"
#include "qidxfile.h"
#include "errors.h"

#define CHUNK_QIDX 0

static int callback_read_header(fitsbin_t* fb, fitsbin_chunk_t* chunk) {
    qfits_header* primheader = fitsbin_get_primary_header(fb);
    qidxfile* qf = chunk->userdata;

    if (fits_check_endian(primheader)) {
        ERROR("qidx file was written with wrong endianness");
        return -1;
    }
    qf->numstars = qfits_header_getint(primheader, "NSTARS", -1);
    qf->numquads = qfits_header_getint(primheader, "NQUADS", -1);
    qf->dimquads = qfits_header_getint(primheader, "DIMQUADS", 4);
    if ((qf->numstars == -1) || (qf->numquads == -1)) {
        ERROR("Couldn't find NSTARS or NQUADS entries in FITS header");
        return -1;
    }

    chunk->nrows = qf->numstars * 2 + qf->numquads * qf->dimquads;
    return 0;
}

static qidxfile* new_qidxfile(const char* fn, anbool writing) {
    qidxfile* qf;
    fitsbin_chunk_t chunk;

    qf = calloc(1, sizeof(qidxfile));
    if (!qf) {
        SYSERROR("Couldn't malloc a qidxfile struct");
        return NULL;
    }

    // default
    qf->dimquads = 4;

    if (writing)
        qf->fb = fitsbin_open_for_writing(fn);
    else
        qf->fb = fitsbin_open(fn);
    if (!qf->fb) {
        ERROR("Failed to create fitsbin");
        return NULL;
    }

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "qidx";
    chunk.required = 1;
    chunk.callback_read_header = callback_read_header;
    chunk.userdata = qf;
    chunk.itemsize = sizeof(uint32_t);
    fitsbin_add_chunk(qf->fb, &chunk);
    fitsbin_chunk_clean(&chunk);

    return qf;
}

qidxfile* qidxfile_open(const char* fn) {
    qidxfile* qf = NULL;

    qf = new_qidxfile(fn, FALSE);
    if (!qf)
        goto bailout;

    if (fitsbin_read(qf->fb)) {
        ERROR("Failed to find qidx table.\n");
        goto bailout;
    }

    qf->index = fitsbin_get_chunk(qf->fb, CHUNK_QIDX)->data;
    qf->heap  = qf->index + 2 * qf->numstars;
    return qf;

 bailout:
    if (qf)
        qidxfile_close(qf);
    return NULL;
}

int qidxfile_close(qidxfile* qf) {
    int rtn;
    if (!qf) return 0;
    if (fitsbin_get_fid(qf->fb))
        fits_pad_file(fitsbin_get_fid(qf->fb));
    rtn = fitsbin_close(qf->fb);
    free(qf);
    return rtn;
}



qidxfile* qidxfile_open_for_writing(const char* fn, int nstars, int nquads) {
    qidxfile* qf;
    qfits_header* hdr;
    qf = new_qidxfile(fn, TRUE);
    if (!qf)
        goto bailout;
    qf->numstars = nstars;
    qf->numquads = nquads;

    hdr = fitsbin_get_primary_header(qf->fb);
    fits_add_endian(hdr);
    fits_header_add_int(hdr, "NSTARS", qf->numstars, "Number of stars used.");
    fits_header_add_int(hdr, "NQUADS", qf->numquads, "Number of quads used.");
    qfits_header_add(hdr, "AN_FILE", "QIDX", "This is a quad index file.", NULL);
    qfits_header_add(hdr, "COMMENT", "The data table of this file has two parts:", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", " -the index", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", " -the heap", NULL, NULL);
    fits_add_long_comment(hdr, "The index contains two uint32 values for each star: the offset and "
                          "length, in the heap, of the list of quads to which it belongs.  "
                          "The offset and length are in units of uint32s, not bytes.  "
                          "Offset 0 is the first uint32 in the heap.  "
                          "The heap is ordered and tightly packed.  "
                          "The heap is a flat list of quad indices (uint32s).");
    return qf;

 bailout:
    if (qf)
        qidxfile_close(qf);
    return NULL;
}

int qidxfile_write_header(qidxfile* qf) {
    fitsbin_t* fb = qf->fb;
    fitsbin_chunk_t* chunk;
    chunk = fitsbin_get_chunk(fb, CHUNK_QIDX);
    chunk->nrows = 2 * qf->numstars + qf->dimquads * qf->numquads;
    if (fitsbin_write_primary_header(fb) ||
        fitsbin_write_chunk_header(fb, chunk)) {
        ERROR("Failed to write qidxfile header");
        return -1;
    }
    qf->cursor_index = 0;
    qf->cursor_heap  = 0;
    return 0;
}

int qidxfile_write_star(qidxfile* qf, int* quads, int nquads) {
    fitsbin_t* fb = qf->fb;
    FILE* fid;
    uint32_t nq;
    int i;
    fitsbin_chunk_t* chunk;

    fid = fitsbin_get_fid(fb);
    chunk = fitsbin_get_chunk(fb, CHUNK_QIDX);

    // Write the offset & size:
    if (fseeko(fid, fitsbin_get_data_start(fb, chunk) + qf->cursor_index * 2 * sizeof(uint32_t), SEEK_SET)) {
        ERROR("qidxfile_write_star: failed to fseek");
        return -1;
    }
    nq = nquads;
    if (fitsbin_write_item(fb, chunk, &qf->cursor_heap) ||
        fitsbin_write_item(fb, chunk, &nq)) {
        ERROR("qidxfile_write_star: failed to write a qidx offset/size");
        return -1;
    }
    // Write the quads.
    if (fseeko(fid, fitsbin_get_data_start(fb, chunk) + qf->numstars * 2 * sizeof(uint32_t) +
               qf->cursor_heap * sizeof(uint32_t), SEEK_SET)) {
        SYSERROR("qidxfile_write_star: failed to fseek");
        return -1;
    }

    for (i=0; i<nquads; i++) {
        // (in case uint != uint32_t)
        uint32_t q = quads[i];
        if (fitsbin_write_item(fb, chunk, &q)) {
            ERROR("qidxfile_write_star: failed to write quads");
            return -1;
        }
    }

    qf->cursor_index++;
    qf->cursor_heap += nquads;
    return 0;
}

int qidxfile_get_quads(const qidxfile* qf, int starid, uint32_t** quads, int* nquads) {
    int heapindex = qf->index[2*starid];
    *nquads = qf->index[2*starid + 1];
    *quads = qf->heap + heapindex;
    return 0;
}

qfits_header* qidxfile_get_header(const qidxfile* qf) {
    return fitsbin_get_primary_header(qf->fb);
}
