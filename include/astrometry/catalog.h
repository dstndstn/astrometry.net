/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef CATUTILS_H
#define CATUTILS_H

#include <sys/types.h>
#include <stdint.h>

#include "astrometry/fitsbin.h"
#include "astrometry/bl.h"
#include "astrometry/an-bool.h"

#define AN_FILETYPE_CATALOG "OBJS"

struct catalog {
    int numstars;

    int healpix;
    int hpnside;

    double* stars;

    // optional table: star magnitudes and mag errors.
    float* mag;
    float* mag_err;

    // optional tables: positional error ellipses, proper motions
    float* sigma_radec;   // sigma_ra, sigma_dec
    float* proper_motion; // motion_ra, motion_dec
    float* sigma_pm;      // sigma_motion_ra, sigma_motion_dec

    // optional table: star IDs
    uint64_t* starids;

    // while writing: storage for the extra fields.
    fl* maglist;
    fl* magerrlist;
    fl* siglist;
    fl* pmlist;
    fl* sigpmlist;
    bl* idlist;

    fitsbin_t* fb;
};
typedef struct catalog catalog;

catalog* catalog_open(char* catfn);

catalog* catalog_open_for_writing(char* catfn);

double* catalog_get_star(catalog* cat, int sid);

double* catalog_get_base(catalog* cat);

int catalog_write_star(catalog* cat, double* star);

int catalog_close(catalog* cat);

//void catalog_compute_radecminmax(catalog* cat);

int catalog_write_header(catalog* cat);

qfits_header* catalog_get_header(catalog* cat);

int catalog_fix_header(catalog* cat);

anbool catalog_has_mag(const catalog* cat);

void catalog_add_mag(catalog* cat, float mag);
void catalog_add_mag_err(catalog* cat, float magerr);
void catalog_add_sigmas(catalog* cat, float sra, float sdec);
void catalog_add_pms(catalog* cat, float sra, float sdec);
void catalog_add_sigma_pms(catalog* cat, float sra, float sdec);
void catalog_add_id(catalog* cat, uint64_t id);

/*
 This should be called after writing all the star positions and
 calling catalog_fix_header().  It appends the data in "cat->mags"
 to the file as an extra FITS table.
 */
int catalog_write_mags(catalog* cat);
int catalog_write_mag_errs(catalog* cat);
int catalog_write_sigmas(catalog* cat);
int catalog_write_pms(catalog* cat);
int catalog_write_sigma_pms(catalog* cat);
int catalog_write_ids(catalog* cat);

#endif
