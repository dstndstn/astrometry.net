/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang.

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

#ifndef FITSBIN_H
#define FITSBIN_H

#include <stdio.h>

#include "qfits.h"
#include "bl.h"

struct fitsbin_t;

struct fitsbin_chunk_t {
	char* tablename;

    // internal use: pointer to strdup'd name.
    char* tablename_copy;

	// The data (NULL if the table was not found)
	void* data;

	// The size of a single row in bytes.
	int itemsize;

	// The number of items (rows)
	int nrows;

	// abort if this table isn't found?
	int required;

    // Reading:
    //int (*callback_read_header)(qfits_header* primheader, qfits_header* header, size_t* expected, void* userdata);
    int (*callback_read_header)(struct fitsbin_t* fb, struct fitsbin_chunk_t* chunk);
    void* userdata;

    qfits_header* header;

    // Writing:
    off_t header_start;
    off_t header_end;

	// Internal use:
	// The mmap'ed address
	char* map;
	// The mmap'ed size.
	size_t mapsize;
};
typedef struct fitsbin_chunk_t fitsbin_chunk_t;


struct fitsbin_t {
	char* filename;

    bl* chunks;

    // Writing:
    FILE* fid;

    // The primary FITS header
    qfits_header* primheader;
    off_t primheader_end;

    // for use by callback_read_header().
    void* userdata;
};
typedef struct fitsbin_t fitsbin_t;

/**
 Typical usage patterns:

 -Reading:
 fitsbin_open();
 fitsbin_add_chunk();
 fitsbin_add_chunk();
 ...
 fitsbin_read();
 ...
 fitsbin_close();

 OR:
 fb = fitsbin_open();
 fitsbin_chunk_init(&chunk);
 chunk.tablename = "hello";
 fitsbin_read_chunk(fb, &chunk);
 // chunk.data;
 //NO fitsbin_add_chunk(fb, &chunk);
 fitsbin_close();

 -Writing:
 fitsbin_open_for_writing();
 fitsbin_add_chunk();
 fitsbin_add_chunk();
 ...
 fitsbin_write_primary_header();
 ...
 fitsbin_write_chunk_header();
 fitsbin_write_items();
 ...
 fitsbin_fix_chunk_header();

 fitsbin_write_chunk_header();
 fitsbin_write_items();
 ...
 fitsbin_fix_chunk_header();
 ...
 fitsbin_fix_primary_header();
 fitsbin_close();

 OR:

 fb = fitsbin_open_for_writing();
 fitsbin_write_primary_header();

 fitsbin_chunk_init(&chunk);
 chunk.tablename = "whatever";
 chunk.data = ...;
 chunk.itemsize = 4;
 chunk.nrows = 1000;
 fitsbin_write_chunk(fb, &chunk);
 fitsbin_chunk_clean(&chunk);

 fitsbin_fix_primary_header();
 fitsbin_close(fb);

 */

// Initializes a chunk to default values
void fitsbin_chunk_init(fitsbin_chunk_t* chunk);

// Frees contents of this chunk.
void fitsbin_chunk_clean(fitsbin_chunk_t* chunk);

// clean + init
void fitsbin_chunk_reset(fitsbin_chunk_t* chunk);




fitsbin_t* fitsbin_open(const char* fn);

fitsbin_t* fitsbin_open_for_writing(const char* fn);

int fitsbin_read(fitsbin_t* fb);

fitsbin_chunk_t* fitsbin_get_chunk(fitsbin_t* fb, int chunk);

off_t fitsbin_get_data_start(fitsbin_t* fb, fitsbin_chunk_t* chunk);

int fitsbin_n_chunks(fitsbin_t* fb);

/**
 Appends the given chunk -- makes a copy of the contents of "chunk" and
 returns a pointer to the stored location.
 */
fitsbin_chunk_t* fitsbin_add_chunk(fitsbin_t* fb, fitsbin_chunk_t* chunk);

/**
 Immediately tries to read a chunk.  If the chunk is not found, -1 is returned
 and the chunk is not added to this fitsbin's list.  If it's found, 0 is
 returned, a copy of the chunk is stored, and the results are placed in
 "chunk".
 */
int fitsbin_read_chunk(fitsbin_t* fb, fitsbin_chunk_t* chunk);

FILE* fitsbin_get_fid(fitsbin_t* fb);

int fitsbin_close(fitsbin_t* fb);

qfits_header* fitsbin_get_primary_header(fitsbin_t* fb);

// (pads to FITS block size)
int fitsbin_write_primary_header(fitsbin_t* fb);

// (pads to FITS block size)
int fitsbin_fix_primary_header(fitsbin_t* fb);

qfits_header* fitsbin_get_chunk_header(fitsbin_t* fb, fitsbin_chunk_t* chunk);

int fitsbin_write_chunk(fitsbin_t* fb, fitsbin_chunk_t* chunk);

// (pads to FITS block size)
int fitsbin_write_chunk_header(fitsbin_t* fb, fitsbin_chunk_t* chunk);

// (pads to FITS block size)
int fitsbin_fix_chunk_header(fitsbin_t* fb, fitsbin_chunk_t* chunk);

int fitsbin_write_item(fitsbin_t* fb, fitsbin_chunk_t* chunk, void* data);

int fitsbin_write_items(fitsbin_t* fb, fitsbin_chunk_t* chunk, void* data, int N);

#endif
