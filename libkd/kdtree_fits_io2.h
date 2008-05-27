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

#ifndef KDTREE_FITS_IO_H
#define KDTREE_FITS_IO_H

#include <stdio.h>

#include "kdtree.h"
#include "fitsbin.h"

/*
// convenience function
kdtree_t* kdtree_fits_read(const char* fn, const char* treename, qfits_header** p_hdr);

kdtree_t* kdtree_fits_read_extras(const char* fn, const char* treename, qfits_header** p_hdr, extra_table* extras, int nextras);

int kdtree_fits_write(const kdtree_t* kdtree, const char* fn, const qfits_header* hdr);

int kdtree_fits_write_extras(const kdtree_t* kdtree, const char* fn, const qfits_header* hdr, const extra_table* extras, int nextras);

void kdtree_fits_close(kdtree_t* kd);

FILE* kdtree_fits_write_primary_header(const char* fn);

int kdtree_fits_append(const kdtree_t* kdtree, const qfits_header* hdr, FILE* out);

int kdtree_fits_append_extras(const kdtree_t* kdtree, const qfits_header* hdr, const extra_table* extras, int nextras, FILE* out);
 */


/**
 Usage patterns:

 kdtree_io_t* io = kdtree_io_open("in.kd.fits");
 kdtree_t* kd = kdtree_io_read_tree(io, "mytree");
 // kd contains the tree that was read.
 // io->fb->primheader is the primary header

 fitsbin_chunk_t chunk;
 chunk.tablename = "my_extra_data";
 chunk.itemsize = sizeof(int32_t);
 chunk.nrows = kd->ndata;
 kdtree_io_read_chunk(io, &chunk);

 // chunk->header
 // chunk->data

 kdtree_io_close();




 kdtree_io_t* io = kdtree_io_open_for_writing("out.kd.fits");

 kdtree_t* mytree = ...;
 kdtree_io_write_tree(io, mytree);

 fitsbin_chunk_t chunk;
 chunk.tablename = "my_extra";
 chunk.data = ...;
 chunk.itemsize = sizeof(int32_t);
 chunk.nrows = mytree->ndata;
 kdtree_io_write_chunk(io, &chunk)

 kdtree_io_close();

 */

kdtree_io_t* kdtree_io_open(const char* fn);

kdtree_t* kdtree_io_read_tree(kdtree_io_t* io, const char* treename);

int kdtree_io_read_chunk(kdtree_io_t* io, fitsbin_chunk_t* chunk);

qfits_header* kdtree_io_get_primary_header(kdtree_io_t* io);



kdtree_io_t* kdtree_io_open_for_writing(const char* fn);

kdtree_io_write_tree(kdtree_io_t* io, kdtree_t* kd);

fitsbin_t* kdtree_io_get_fitsbin(kdtree_io_t* io);

int kdtree_io_write_chunk(kdtree_io_t* io, fitsbin_chunk_t* chunk);


kdtree_io_close(kdtree_io_t* io);

/*
 struct kdtree_io_s {
 fitsbin_t* fb;
 kdtree_t* kd;
 };
 typedef struct kdtree_io_s kdtree_io_t;
 */

typedef fitsbin_t kdtree_io_t;

/*
 struct extra_table_info {
 fitsbin_chunk_t chunk;
 
 // this gets called after the kdtree size has been discovered.
 void (*compute_tablesize)(kdtree_t* kd, struct extra_table_info* thistable);
 
 // when writing: don't write this one.
 int dontwrite;
 };
 */

// names (actually prefixes) of FITS tables.
#define KD_STR_HEADER    "kdtree_header"
#define KD_STR_NODES     "kdtree_nodes"
#define KD_STR_LR        "kdtree_lr"
#define KD_STR_PERM      "kdtree_perm"
#define KD_STR_BB        "kdtree_bb"
#define KD_STR_SPLIT     "kdtree_split"
#define KD_STR_SPLITDIM  "kdtree_splitdim"
#define KD_STR_DATA      "kdtree_data"
#define KD_STR_RANGE     "kdtree_range"

// is the given column name one of the above strings?
int kdtree_fits_column_is_kdtree(char* columnname);

#endif
