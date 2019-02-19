/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef FITSBIN_H
#define FITSBIN_H

#include <stdio.h>

#include "astrometry/anqfits.h"
#include "astrometry/bl.h"
#include "astrometry/an-bool.h"
/**
 "fitsbin" is our abuse of FITS binary tables to hold raw binary data,
 *without endian flips*, by storing the data as characters/bytes.
 This has the advantage that they can be directly mmap()'d, but of
 course means that they aren't endian-independent.  We accept that
 tradeoff in the interest of speed and the recognition that x86 is
 pretty much all we care about.


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
    int (*callback_read_header)(struct fitsbin_t* fb, struct fitsbin_chunk_t* chunk);
    void* userdata;

    qfits_header* header;

    // Writing:
    off_t header_start;
    off_t header_end;

    // on output, force a type other than A?
    tfits_type forced_type;

    // Internal use:
    // The mmap'ed address
    char* map;
    // The mmap'ed size.
    size_t mapsize;
};
typedef struct fitsbin_chunk_t fitsbin_chunk_t;


struct fitsbin_t {
    char* filename;

    anqfits_t* fits;

    bl* chunks;

    // Writing:
    FILE* fid;

    // only used for in_memory():
    anbool inmemory;
    bl* items;
    bl* extensions;

    // The primary FITS header
    qfits_header* primheader;
    off_t primheader_end;

    // for use when reading (not in_memory()): cache the tables in this FITS file.
    // ideally this would be pushed down to the qfits layer...
    qfits_table** tables;
    // number of extensions, include the primary; extensions < Next are valid.
    int Next;

    // for use by callback_read_header().
    void* userdata;
};
typedef struct fitsbin_t fitsbin_t;

// Initializes a chunk to default values
void fitsbin_chunk_init(fitsbin_chunk_t* chunk);

// Frees contents of this chunk.
void fitsbin_chunk_clean(fitsbin_chunk_t* chunk);

// clean + init
void fitsbin_chunk_reset(fitsbin_chunk_t* chunk);

char* fitsbin_get_filename(const fitsbin_t* fb);

// Reading: returns a new copy of the given FITS extension header.
// (-> *qfits_get_header)
qfits_header* fitsbin_get_header(const fitsbin_t* fb, int ext);

// Reading: how many extensions in this file?  (-> *qfits_query_n_ext)
int fitsbin_n_ext(const fitsbin_t* fb);

fitsbin_t* fitsbin_open(const char* fn);

fitsbin_t* fitsbin_open_fits(anqfits_t* fits);

fitsbin_t* fitsbin_open_for_writing(const char* fn);

fitsbin_t* fitsbin_open_in_memory(void);

int fitsbin_close_fd(fitsbin_t* fb);

int fitsbin_switch_to_reading(fitsbin_t* fb);

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

qfits_header* fitsbin_get_primary_header(const fitsbin_t* fb);

void fitsbin_set_primary_header(fitsbin_t* fb, const qfits_header* hdr);

// (pads to FITS block size)
int fitsbin_write_primary_header(fitsbin_t* fb);

// (pads to FITS block size)
int fitsbin_fix_primary_header(fitsbin_t* fb);

qfits_header* fitsbin_get_chunk_header(fitsbin_t* fb, fitsbin_chunk_t* chunk);

int fitsbin_write_chunk(fitsbin_t* fb, fitsbin_chunk_t* chunk);

int fitsbin_write_chunk_flipped(fitsbin_t* fb, fitsbin_chunk_t* chunk,
                                int wordsize);

// (pads to FITS block size)
int fitsbin_write_chunk_header(fitsbin_t* fb, fitsbin_chunk_t* chunk);

// (pads to FITS block size)
int fitsbin_fix_chunk_header(fitsbin_t* fb, fitsbin_chunk_t* chunk);

int fitsbin_write_item(fitsbin_t* fb, fitsbin_chunk_t* chunk, void* data);

int fitsbin_write_items(fitsbin_t* fb, fitsbin_chunk_t* chunk, void* data, int N);


// direct FILE* output:

int fitsbin_write_primary_header_to(fitsbin_t* fb, FILE* fid);

int fitsbin_write_chunk_header_to(fitsbin_t* fb, fitsbin_chunk_t* chunk, FILE* fid);

int fitsbin_write_items_to(fitsbin_chunk_t* chunk, void* data, int N, FILE* fid);

int fitsbin_write_chunk_to(fitsbin_t* fb, fitsbin_chunk_t* chunk, FILE* fid);

#endif
