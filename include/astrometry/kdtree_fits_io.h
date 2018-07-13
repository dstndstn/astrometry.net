/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef KDTREE_FITS_IO_H
#define KDTREE_FITS_IO_H

#include <stdio.h>

#include "astrometry/kdtree.h"
#include "astrometry/fitsbin.h"
#include "astrometry/anqfits.h"

/**
 Usage patterns:

 kdtree_fits_t* io = kdtree_fits_open("in.kd.fits");
 kdtree_t* kd = kdtree_fits_read_tree(io, "mytree");
 // kd contains the tree that was read.
 // io->fb->primheader is the primary header

 fitsbin_chunk_t chunk;
 chunk.tablename = "my_extra_data";
 chunk.itemsize = sizeof(int32_t);
 chunk.nrows = kd->ndata;
 kdtree_fits_read_chunk(io, &chunk);

 // chunk->header
 // chunk->data

 kdtree_fits_close();




 kdtree_fits_t* io = kdtree_fits_open_for_writing("out.kd.fits");

 kdtree_t* mytree = ...;
 kdtree_fits_write_tree(io, mytree);

 fitsbin_chunk_t chunk;
 chunk.tablename = "my_extra";
 chunk.data = ...;
 chunk.itemsize = sizeof(int32_t);
 chunk.nrows = mytree->ndata;
 kdtree_fits_write_chunk(io, &chunk)

 kdtree_fits_close();

 */
typedef fitsbin_t kdtree_fits_t;

kdtree_fits_t* kdtree_fits_open(const char* fn);

kdtree_fits_t* kdtree_fits_open_fits(anqfits_t* fits);

// convenience...
kdtree_t* kdtree_fits_read(const char* fn, const char* treename,
                           qfits_header** p_hdr);

int kdtree_fits_write(const kdtree_t* kdtree, const char* fn,
                      const qfits_header* hdr);

//sl* kdtree_fits_list_trees(kdtree_fits_t* io);

int kdtree_fits_contains_tree(const kdtree_fits_t* io, const char* treename);

fitsbin_t* kdtree_fits_get_fitsbin(kdtree_fits_t* io);

kdtree_t* kdtree_fits_read_tree(kdtree_fits_t* io, const char* treename,
                                qfits_header** p_hdr);

int kdtree_fits_read_chunk(kdtree_fits_t* io, fitsbin_chunk_t* chunk);

qfits_header* kdtree_fits_get_primary_header(kdtree_fits_t* io);



kdtree_fits_t* kdtree_fits_open_for_writing(const char* fn);

// writes the primary header and the tree.
int kdtree_fits_write_tree(kdtree_fits_t* io, const kdtree_t* kd,
                           const qfits_header* add_headers);

// just writes the tree, no primary header.
int kdtree_fits_append_tree(kdtree_fits_t* io, const kdtree_t* kd,
                            const qfits_header* add_headers);


int kdtree_fits_append_tree_to(kdtree_t* kd,
                               const qfits_header* inhdr,
                               FILE* fid);


int kdtree_fits_write_primary_header(kdtree_fits_t* io,
                                     const qfits_header* add_headers);

int kdtree_fits_write_chunk(kdtree_fits_t* io, fitsbin_chunk_t* chunk);

int kdtree_fits_write_chunk_to(fitsbin_chunk_t* chunk, FILE* fid);

int kdtree_fits_close(kdtree_t* io);

int kdtree_fits_io_close(kdtree_fits_t* io);

// flipped-endian writing...
int kdtree_fits_write_chunk_flipped(kdtree_fits_t* io, fitsbin_chunk_t* chunk,
                                    int wordsize);

int kdtree_fits_write_flipped(const kdtree_t* kdtree, const char* fn,
                              const qfits_header* hdr);
int kdtree_fits_write_tree_flipped(kdtree_fits_t* io, const kdtree_t* kd,
                                   const qfits_header* inhdr);
int kdtree_fits_append_tree_flipped(kdtree_fits_t* io, const kdtree_t* kd,
                                    const qfits_header* inhdr);


// names (actually prefixes) of FITS tables.
#define KD_STR_HEADER    "kdtree_header"
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
