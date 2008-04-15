/*
  This file is part of the Astrometry.net suite.
  Copyright 2007-2008 Dustin Lang.

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

#include <stdarg.h>
#include <stdio.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include "keywords.h"
#include "fitsbin.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "fitsfile.h"

/*
 static void
 ATTRIB_FORMAT(printf,2,3)
 seterr(char** errstr, const char* format, ...) {
 va_list va;
 if (!errstr) return;
 va_start(va, format);
 vasprintf(errstr, format, va);
 va_end(va);
 // Hack
 fprintf(stderr, "%s", *errstr);
 }
 */

FILE* fitsbin_get_fid(fitsbin_t* fb) {
    return fb->fid;
}

off_t fitsbin_get_data_start(fitsbin_t* fb, int chunk) {
    assert(chunk < fb->nchunks);
    return fb->chunks[chunk].header_end;
}

void fitsbin_set_filename(fitsbin_t* fb, const char* fn) {
    free(fb->filename);
    fb->filename = strdup(fn);
}

fitsbin_t* fitsbin_new(int nchunks) {
	fitsbin_t* fb;
	fb = calloc(1, sizeof(fitsbin_t));
	if (!fb)
		return NULL;
    fb->chunks = calloc(nchunks, sizeof(fitsbin_chunk_t));
    fb->nchunks = nchunks;
	return fb;
}

static void free_chunk(fitsbin_chunk_t* chunk) {
    if (!chunk) return;
	free(chunk->tablename);
    if (chunk->header)
        qfits_header_destroy(chunk->header);
	if (chunk->map) {
		if (munmap(chunk->map, chunk->mapsize)) {
			fprintf(stderr, "Failed to munmap fitsbin: %s\n", strerror(errno));
		}
	}
}

int fitsbin_close(fitsbin_t* fb) {
    int i;
    int rtn = 0;
	if (!fb) return rtn;
    if (fb->fid) {
		fits_pad_file(fb->fid);
		if (fclose(fb->fid)) {
			fprintf(stderr, "Error closing fitsbin file: %s\n", strerror(errno));
            rtn = -1;
        }
    }
    if (fb->primheader)
        qfits_header_destroy(fb->primheader);
    for (i=0; i<fb->nchunks; i++)
        free_chunk(fb->chunks + i);
    free(fb->filename);
    free(fb->chunks);
	free(fb);
    return rtn;
}

int fitsbin_write_primary_header(fitsbin_t* fb) {
    return fitsfile_write_primary_header(fb->fid, fb->primheader,
                                         &fb->primheader_end, fb->filename);
}

qfits_header* fitsbin_get_primary_header(fitsbin_t* fb) {
    return fb->primheader;
}

int fitsbin_fix_primary_header(fitsbin_t* fb) {
    return fitsfile_fix_primary_header(fb->fid, fb->primheader,
                                       &fb->primheader_end, fb->filename);
}

int fitsbin_write_header(fitsbin_t* fb) {
    return fitsbin_write_chunk_header(fb, 0);
}

qfits_header* fitsbin_get_chunk_header(fitsbin_t* fb, int chunknum) {
    fitsbin_chunk_t* chunk;
    qfits_table* table;
    int tablesize;
    qfits_header* hdr;
    int ncols = 1;

    chunk = fb->chunks + chunknum;
	// the table header
	tablesize = chunk->itemsize * chunk->nrows * ncols;
	table = qfits_table_new(fb->filename, QFITS_BINTABLE, tablesize, ncols, chunk->nrows);
	assert(table);
    qfits_col_fill(table->col, chunk->itemsize, 0, 1, TFITS_BIN_TYPE_A,
				   chunk->tablename, "", "", "", 0, 0, 0, 0, 0);
    hdr = qfits_table_ext_header_default(table);
    qfits_table_close(table);
    chunk->header = hdr;
    return hdr;
}

int fitsbin_write_chunk_header(fitsbin_t* fb, int chunknum) {
    qfits_header* hdr;
    fitsbin_chunk_t* chunk;

    assert(chunknum < fb->nchunks);
    assert(chunknum >= 0);
    chunk = fb->chunks + chunknum;
    hdr = fitsbin_get_chunk_header(fb, chunknum);
    if (fitsfile_write_header(fb->fid, chunk->header,
                              &chunk->header_start, &chunk->header_end,
                              chunknum, fb->filename)) {
        return -1;
    }
	return 0;
}

int fitsbin_fix_header(fitsbin_t* fb) {
    return fitsbin_fix_chunk_header(fb, 0);
}

int fitsbin_fix_chunk_header(fitsbin_t* fb, int chunknum) {
    fitsbin_chunk_t* chunk;
    assert(chunknum < fb->nchunks);
    assert(chunknum >= 0);
    chunk = fb->chunks + chunknum;
    if (fitsfile_fix_header(fb->fid, chunk->header,
                            &chunk->header_start, &chunk->header_end,
                            chunknum, fb->filename)) {
        return -1;
    }
	return 0;
}

int fitsbin_write_items(fitsbin_t* fb, int chunk, void* data, int N) {
    if (fwrite(data, fb->chunks[chunk].itemsize, N, fb->fid) != N) {
        fprintf(stderr, "Failed to write %i items: %s\n", N, strerror(errno));
        return -1;
    }
    return 0;
}

int fitsbin_write_item(fitsbin_t* fb, int chunk, void* data) {
    return fitsbin_write_items(fb, chunk, data, 1);
}

int fitsbin_start_write(fitsbin_t* fb) {
    return 0;
}

fitsbin_t* fitsbin_open_for_writing(const char* fn, const char* tablename) {
	fitsbin_t* fb;
    fitsbin_chunk_t* chunk;

	fb = fitsbin_new(1);
	if (!fb)
        return NULL;

	fb->fid = fopen(fn, "wb");
	if (!fb->fid) {
		fprintf(stderr, "Couldn't open file \"%s\" for output: %s\n", fn, strerror(errno));
        fitsbin_close(fb);
        return NULL;
	}

	fb->filename = strdup(fn);
    fb->primheader = qfits_header_default();

    chunk = fb->chunks;
	chunk->tablename = strdup(tablename);

	return fb;
}

fitsbin_t* fitsbin_open(const char* fn, const char* tablename,
						int (*callback_read_header)(qfits_header* primheader, qfits_header* header, size_t* expected, char** errstr, void* userdata),
						void* userdata) {
    fitsbin_t* fb;
    fitsbin_chunk_t* chunk;
    int rtn;

    fb = fitsbin_new(1);
    if (!fb)
        return fb;

    fb->primheader = qfits_header_read(fn);
    if (!fb->primheader) {
        fprintf(stderr, "Couldn't read FITS header from file \"%s\".", fn);
        fitsbin_close(fb);
        return NULL;
    }

	fb->filename = strdup(fn);

    chunk = fb->chunks;
    chunk->tablename = strdup(tablename);
    chunk->callback_read_header = callback_read_header;
    chunk->userdata = userdata;

    rtn = fitsbin_read(fb);
    if (rtn) {
        fitsbin_close(fb);
        return NULL;
    }
    return fb;
}

int fitsbin_read(fitsbin_t* fb) {
	FILE* fid = NULL;
    int tabstart, tabsize, ext;
    size_t expected = 0;
	int mode, flags;
	off_t mapstart;
	int mapoffset;
    char* fn;
    int i;
    char* errstr;

    fn = fb->filename;

	if (!qfits_is_fits(fn)) {
        fprintf(stderr, "File \"%s\" is not FITS format.", fn);
        goto bailout;
	}

    // HACK .... what is the interaction between this func and fitsbin_open()?
    if (!fb->primheader) {
        fb->primheader = qfits_header_read(fn);
        if (!fb->primheader) {
            fprintf(stderr, "Couldn't read FITS header from file \"%s\".", fn);
            //fitsbin_close(fb);
            return -1;
        }
    }

	fid = fopen(fn, "rb");
	if (!fid) {
		fprintf(stderr, "Failed to open file \"%s\": %s.", fn, strerror(errno));
        goto bailout;
	}

    for (i=0; i<fb->nchunks; i++) {
        fitsbin_chunk_t* chunk = fb->chunks + i;

        if (fits_find_table_column(fn, chunk->tablename, &tabstart, &tabsize, &ext)) {
            if (chunk->required) {
                fprintf(stderr, "Couldn't find table \"%s\" in file \"%s\".", chunk->tablename, fn);
                goto bailout;
            } else {
                continue;
            }
        }

        chunk->header = qfits_header_readext(fn, ext);
        if (!chunk->header) {
            fprintf(stderr, "Couldn't read FITS header from file \"%s\" extension %i.", fn, ext);
            goto bailout;
        }

        if (chunk->callback_read_header &&
            chunk->callback_read_header(fb->primheader, chunk->header, &expected, &errstr, chunk->userdata)) {
            fprintf(stderr, "fitsbin callback failed: %s\n", errstr);
            goto bailout;
        }

        if (expected && (fits_bytes_needed(expected) != tabsize)) {
            fprintf(stderr, "Expected table size (%i => %i FITS blocks) is not equal to size of table \"%s\" (%i FITS blocks).",
                   (int)expected, fits_blocks_needed(expected), chunk->tablename, tabsize / FITS_BLOCK_SIZE);
            goto bailout;
        }

        mode = PROT_READ;
        flags = MAP_SHARED;

        get_mmap_size(tabstart, tabsize, &mapstart, &(chunk->mapsize), &mapoffset);

        chunk->map = mmap(0, chunk->mapsize, mode, flags, fileno(fid), mapstart);
        if (chunk->map == MAP_FAILED) {
            fprintf(stderr, "Couldn't mmap file \"%s\": %s", fn, strerror(errno));
            chunk->map = NULL;
            goto bailout;
        }
        chunk->data = chunk->map + mapoffset;
    }
    fclose(fid);
    fid = NULL;

    return 0;

 bailout:
    return -1;
}
