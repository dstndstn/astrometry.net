/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>

#include "qfits.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "ioutils.h"
#include "qidxfile.h"

#define CHUNK_QIDX 0

static int callback_read_header(qfits_header* primheader, qfits_header* header,
								size_t* expected, char** errstr,
								void* userdata) {
	qidxfile* qf = userdata;

	if (fits_check_endian(primheader)) {
		if (errstr) *errstr = "File was written with wrong endianness.";
		return -1;
	}
	qf->numstars = qfits_header_getint(primheader, "NSTARS", -1);
	qf->numquads = qfits_header_getint(primheader, "NQUADS", -1);
    qf->dimquads = qfits_header_getint(primheader, "DIMQUADS", 4);
	if ((qf->numstars == -1) || (qf->numquads == -1)) {
		if (errstr) *errstr = "Couldn't find NSTARS or NQUADS entries in FITS header.";
		return -1;
	}

    *expected = qf->numstars * 2 * sizeof(uint32_t) +
		qf->numquads * qf->dimquads * sizeof(uint32_t);
	return 0;
}

static qidxfile* new_qidxfile() {
	qidxfile* qf;
	fitsbin_chunk_t* chunk;

    qf = calloc(1, sizeof(qidxfile));
	if (!qf) {
		fprintf(stderr, "Couldn't malloc a qidxfile struct: %s\n", strerror(errno));
		return NULL;
	}

    // default
    qf->dimquads = 4;

    qf->fb = fitsbin_new(1);

    chunk = qf->fb->chunks + CHUNK_QIDX;
    chunk->tablename = strdup("qidx");
    chunk->required = 1;
    chunk->callback_read_header = callback_read_header;
    chunk->userdata = qf;
	chunk->itemsize = sizeof(uint32_t);

	return qf;
}

qidxfile* qidxfile_open(const char* fn) {
	qidxfile* qf = NULL;

	qf = new_qidxfile();
	if (!qf)
		goto bailout;

    fitsbin_set_filename(qf->fb, fn);
    if (fitsbin_read(qf->fb))
        goto bailout;

	qf->index = qf->fb->chunks[CHUNK_QIDX].data;
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
	rtn = fitsbin_close(qf->fb);
	free(qf);
    return rtn;
}

qidxfile* qidxfile_open_for_writing(const char* fn, uint nstars, uint nquads) {
	qidxfile* qf;
	qfits_header* hdr;

	qf = new_qidxfile();
	if (!qf)
		goto bailout;
	qf->numstars = nstars;
	qf->numquads = nquads;
    fitsbin_set_filename(qf->fb, fn);

    if (fitsbin_start_write(qf->fb))
        goto bailout;

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
	fb->chunks[CHUNK_QIDX].nrows = 2 * qf->numstars + qf->dimquads * qf->numquads;
	if (fitsbin_write_primary_header(fb) ||
		fitsbin_write_header(fb)) {
		fprintf(stderr, "Failed to write qidxfile header.\n");
		return -1;
	}
	qf->cursor_index = 0;
	qf->cursor_heap  = 0;
	return 0;
}

int qidxfile_write_star(qidxfile* qf, uint* quads, uint nquads) {
	fitsbin_t* fb = qf->fb;
	FILE* fid;
	uint32_t nq;
	int i;

    fid = fitsbin_get_fid(fb);

	// Write the offset & size:
	if (fseeko(fid, fitsbin_get_data_start(fb, CHUNK_QIDX) + qf->cursor_index * 2 * sizeof(uint32_t), SEEK_SET)) {
		fprintf(stderr, "qidxfile_write_star: failed to fseek: %s\n", strerror(errno));
		return -1;
	}
	nq = nquads;
	if (fitsbin_write_item(fb, CHUNK_QIDX, &qf->cursor_heap) ||
        fitsbin_write_item(fb, CHUNK_QIDX, &nq)) {
		fprintf(stderr, "qidxfile_write_star: failed to write a qidx offset/size.\n");
		return -1;
	}
	// Write the quads.
	if (fseeko(fid, fitsbin_get_data_start(fb, CHUNK_QIDX) + qf->numstars * 2 * sizeof(uint32_t) +
			   qf->cursor_heap * sizeof(uint32_t), SEEK_SET)) {
		fprintf(stderr, "qidxfile_write_star: failed to fseek: %s\n", strerror(errno));
		return -1;
	}

	for (i=0; i<nquads; i++) {
        // (in case uint != uint32_t)
		uint32_t q = quads[i];
        if (fitsbin_write_item(fb, CHUNK_QIDX, &q)) {
            fprintf(stderr, "qidxfile_write_star: failed to write quads.\n");
            return -1;
        }
    }

	qf->cursor_index++;
	qf->cursor_heap += nquads;
	return 0;
}

int qidxfile_get_quads(const qidxfile* qf, uint starid, uint32_t** quads, uint* nquads) {
	uint heapindex = qf->index[2*starid];
	*nquads = qf->index[2*starid + 1];
	*quads = qf->heap + heapindex;
	return 0;
}

qfits_header* qidxfile_get_header(const qidxfile* qf) {
	return fitsbin_get_primary_header(qf->fb);
}
