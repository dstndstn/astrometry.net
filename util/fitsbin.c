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
#include "errors.h"

FILE* fitsbin_get_fid(fitsbin_t* fb) {
    return fb->fid;
}

static int nchunks(fitsbin_t* fb) {
    return bl_size(fb->chunks);
}

static fitsbin_chunk_t* get_chunk(fitsbin_t* fb, int i) {
    if (i >= bl_size(fb->chunks)) {
        ERROR("Attempt to get chunk %i from a fitsbin with only %i chunks",
              i, bl_size(fb->chunks));
        return NULL;
    }
    if (i < 0) {
        ERROR("Attempt to get fitsbin chunk %i", i);
        return NULL;
    }
    return bl_access(fb->chunks, i);
}

static fitsbin_t* new_fitsbin(const char* fn) {
	fitsbin_t* fb;
	fb = calloc(1, sizeof(fitsbin_t));
	if (!fb)
		return NULL;
    fb->chunks = bl_new(4, sizeof(fitsbin_chunk_t));
    fb->filename = strdup(fn);
	return fb;
}

static void free_chunk(fitsbin_chunk_t* chunk) {
    if (!chunk) return;
	free(chunk->tablename);
    if (chunk->header)
        qfits_header_destroy(chunk->header);
	if (chunk->map) {
		if (munmap(chunk->map, chunk->mapsize)) {
			SYSERROR("Failed to munmap fitsbin");
		}
	}
}

fitsbin_chunk_t* fitsbin_get_chunk(fitsbin_t* fb, int chunk) {
    return get_chunk(fb, chunk);
}

void fitsbin_add_chunk(fitsbin_t* fb, fitsbin_chunk_t* chunk) {
    chunk->tablename = strdup(chunk->tablename);
    bl_append(fb->chunks, chunk);
}

off_t fitsbin_get_data_start(fitsbin_t* fb, int chunk) {
    return get_chunk(fb, chunk)->header_end;
}

int fitsbin_close(fitsbin_t* fb) {
    int i;
    int rtn = 0;
	if (!fb) return rtn;
    if (fb->fid) {
		fits_pad_file(fb->fid);
		if (fclose(fb->fid)) {
			SYSERROR("Error closing fitsbin file");
            rtn = -1;
        }
    }
    if (fb->primheader)
        qfits_header_destroy(fb->primheader);
    for (i=0; i<nchunks(fb); i++)
        free_chunk(get_chunk(fb, i));
    free(fb->filename);
    if (fb->chunks)
        bl_free(fb->chunks);
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

qfits_header* fitsbin_get_chunk_header(fitsbin_t* fb, int chunknum) {
    fitsbin_chunk_t* chunk;
    qfits_table* table;
    int tablesize;
    qfits_header* hdr;
    int ncols = 1;

    chunk = get_chunk(fb, chunknum);
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

    chunk = get_chunk(fb, chunknum);
    hdr = fitsbin_get_chunk_header(fb, chunknum);
    if (fitsfile_write_header(fb->fid, chunk->header,
                              &chunk->header_start, &chunk->header_end,
                              chunknum, fb->filename)) {
        return -1;
    }
	return 0;
}

int fitsbin_fix_chunk_header(fitsbin_t* fb, int chunknum) {
    fitsbin_chunk_t* chunk;
    chunk = get_chunk(fb, chunknum);
    if (fitsfile_fix_header(fb->fid, chunk->header,
                            &chunk->header_start, &chunk->header_end,
                            chunknum, fb->filename)) {
        return -1;
    }
	return 0;
}

int fitsbin_write_items(fitsbin_t* fb, int chunk, void* data, int N) {
    if (fwrite(data, get_chunk(fb, chunk)->itemsize, N, fb->fid) != N) {
        SYSERROR("Failed to write %i items", N);
        return -1;
    }
    return 0;
}

int fitsbin_write_item(fitsbin_t* fb, int chunk, void* data) {
    return fitsbin_write_items(fb, chunk, data, 1);
}

int fitsbin_read(fitsbin_t* fb) {
	FILE* fid = NULL;
    int tabstart, tabsize, ext;
    size_t expected = 0;
	int mode, flags;
	off_t mapstart;
	int mapoffset;
    int i;

    for (i=0; i<nchunks(fb); i++) {
        fitsbin_chunk_t* chunk = get_chunk(fb, i);

        if (fits_find_table_column(fb->filename, chunk->tablename, &tabstart, &tabsize, &ext)) {
            if (!chunk->required)
                continue;
            ERROR("Couldn't find table \"%s\" in file \"%s\"", chunk->tablename, fb->filename);
            goto bailout;
        }

        chunk->header = qfits_header_readext(fb->filename, ext);
        if (!chunk->header) {
            ERROR("Couldn't read FITS header from file \"%s\" extension %i", fb->filename, ext);
            goto bailout;
        }

        if (chunk->callback_read_header &&
            chunk->callback_read_header(fb->primheader, chunk->header, &expected, chunk->userdata)) {
            ERROR("fitsbin callback failed");
            goto bailout;
        }

        if (expected && (fits_bytes_needed(expected) != tabsize)) {
            ERROR("Expected table size (%i => %i FITS blocks) is not equal to size of table \"%s\" (%i FITS blocks).",
                   (int)expected, fits_blocks_needed(expected), chunk->tablename, tabsize / FITS_BLOCK_SIZE);
            goto bailout;
        }

        mode = PROT_READ;
        flags = MAP_SHARED;

        get_mmap_size(tabstart, tabsize, &mapstart, &(chunk->mapsize), &mapoffset);

        chunk->map = mmap(0, chunk->mapsize, mode, flags, fileno(fid), mapstart);
        if (chunk->map == MAP_FAILED) {
            SYSERROR("Couldn't mmap file \"%s\"", fb->filename);
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

fitsbin_t* fitsbin_open(const char* fn) {
    fitsbin_t* fb;
	if (!qfits_is_fits(fn)) {
        ERROR("File \"%s\" is not FITS format.", fn);
        return NULL;
	}
    fb = new_fitsbin(fn);
    if (!fb)
        return fb;
	fb->fid = fopen(fn, "rb");
	if (!fb->fid) {
		SYSERROR("Failed to open file \"%s\"", fn);
        goto bailout;
	}
    fb->primheader = qfits_header_read(fn);
    if (!fb->primheader) {
        ERROR("Couldn't read FITS header from file \"%s\"", fn);
        goto bailout;
    }
    return fb;
 bailout:
    fitsbin_close(fb);
    return NULL;
}

fitsbin_t* fitsbin_single_open_for_writing(const char* fn) {
    fitsbin_t* fb;

    fb = new_fitsbin(fn);
    if (!fb)
        return NULL;
    fb->primheader = qfits_header_default();
	fb->fid = fopen(fb->filename, "wb");
	if (!fb->fid) {
		SYSERROR("Couldn't open file \"%s\" for output", fb->filename);
        fitsbin_close(fb);
        return NULL;
	}
    return fb;
}

