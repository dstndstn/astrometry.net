/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef BUILD_INDEX_H
#define BUILD_INDEX_H

#include "astrometry/fitstable.h"
#include "astrometry/index.h"
#include "astrometry/an-bool.h"

struct index_params {
    // catalog:
    const char* racol;
    const char* deccol;
    // in arcsec
    double jitter;

    // uniformization:
    const char* sortcol;
    anbool sortasc;

    double brightcut;
    int bighp;
    int bignside;
    int sweeps;
    double dedup;
    int margin;
    int UNside;

    // drop RA,Dec from the tagalong table?
    anbool drop_radec;

    // hpquads:
    int Nside;

    void* hpquads_sort_data;
    int (*hpquads_sort_func)(const void*, const void*);
    int hpquads_sort_size;

    // quad size range, in arcmin
    double qlo; double qhi;
    int passes;
    int Nreuse; int Nloosen;
    anbool scanoccupied;
    int dimquads;
    int indexid;

    // general options
    anbool inmemory;
    anbool delete_tempfiles;
    const char* tempdir;
    char** args;
    int argc;
};
typedef struct index_params index_params_t;

void build_index_defaults(index_params_t* params);

int build_index_files(const char* catalogfn, int extension,
                      const char* indexfn,
                      index_params_t* params);

int build_index(fitstable_t* catalog, index_params_t* p,
                index_t** p_index, const char* indexfn);

int build_index_shared_skdt(const char* starkdfn, startree_t* starkd,
                            index_params_t* p,
                            index_t** p_index, const char* indexfn);

int build_index_shared_skdt_files(const char* starkdfn, const char* indexfn,
                                  index_params_t* p);

#endif
