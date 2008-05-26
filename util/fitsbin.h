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

struct fitsbin_chunk_t {
	char* tablename;

	// The data (NULL if the table was not found)
	void* data;

	// The size of a single row in bytes.
	int itemsize;

	// The number of items (rows)
	int nrows;

	// abort if this table isn't found?
	int required;

    // Reading:
    int (*callback_read_header)(qfits_header* primheader, qfits_header* header, size_t* expected, void* userdata);
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

 */

fitsbin_t* fitsbin_open(const char* fn);

fitsbin_t* fitsbin_open_for_writing(const char* fn);

int fitsbin_read(fitsbin_t* fb);

off_t fitsbin_get_data_start(fitsbin_t* fb, int chunk);

fitsbin_chunk_t* fitsbin_get_chunk(fitsbin_t* fb, int chunk);

void fitsbin_add_chunk(fitsbin_t* fb, fitsbin_chunk_t* chunk);

FILE* fitsbin_get_fid(fitsbin_t* fb);

int fitsbin_close(fitsbin_t* fb);

qfits_header* fitsbin_get_primary_header(fitsbin_t* fb);

// (pads to FITS block size)
int fitsbin_write_primary_header(fitsbin_t* fb);

// (pads to FITS block size)
int fitsbin_fix_primary_header(fitsbin_t* fb);

qfits_header* fitsbin_get_chunk_header(fitsbin_t* fb, int chunk);

// (pads to FITS block size)
int fitsbin_write_chunk_header(fitsbin_t* fb, int chunk);

// (pads to FITS block size)
int fitsbin_fix_chunk_header(fitsbin_t* fb, int chunk);

int fitsbin_write_item(fitsbin_t* fb, int chunk, void* data);

int fitsbin_write_items(fitsbin_t* fb, int chunk, void* data, int N);

#endif
