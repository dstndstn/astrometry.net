/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef QIDXFILE_H
#define QIDXFILE_H

#include <sys/types.h>
#include <stdint.h>

#include "astrometry/qfits_header.h"
#include "astrometry/fitsbin.h"

struct qidxfile {
    int numstars;
    int numquads;

    int dimquads;

    fitsbin_t* fb;

    // when reading:
    uint32_t* index;
    uint32_t* heap;

    uint32_t cursor_index;
    uint32_t cursor_heap;
};
typedef struct qidxfile qidxfile;

int qidxfile_close(qidxfile* qf);

// Sets "quads" to a pointer within the qidx's data block.
// DO NOT free this pointer!
// It is valid until the qidxfile is closed.
int qidxfile_get_quads(const qidxfile* qf, int starid, uint32_t** quads, int* nquads);

int qidxfile_write_star(qidxfile* qf, int* quads, int nquads);

int qidxfile_write_header(qidxfile* qf);

qidxfile* qidxfile_open(const char* fname);

qidxfile* qidxfile_open_for_writing(const char* qidxfname,
                                    int nstars, int nquads);

qfits_header* qidxfile_get_header(const qidxfile* qf);

#endif
