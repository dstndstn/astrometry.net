/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "catalog.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "starutil.h"
#include "mathutil.h"
#include "errors.h"

#define CHUNK_XYZ    0
#define CHUNK_MAG    1
#define CHUNK_MAG_ERR 2
#define CHUNK_SIG    3
#define CHUNK_PM     4
#define CHUNK_SIGPM  5
#define CHUNK_STARID 6

static fitsbin_chunk_t* xyz_chunk(catalog* cat) {
    return fitsbin_get_chunk(cat->fb, CHUNK_XYZ);
}
static fitsbin_chunk_t* mag_chunk(catalog* cat) {
    return fitsbin_get_chunk(cat->fb, CHUNK_MAG);
}
static fitsbin_chunk_t* mag_err_chunk(catalog* cat) {
    return fitsbin_get_chunk(cat->fb, CHUNK_MAG_ERR);
}
static fitsbin_chunk_t* get_chunk(catalog* cat, int i) {
    return fitsbin_get_chunk(cat->fb, i);
}

static int callback_read_header(fitsbin_t* fb, fitsbin_chunk_t* chunk) {
    qfits_header* primheader = fitsbin_get_primary_header(fb);
    catalog* cat = chunk->userdata;
    cat->numstars = qfits_header_getint(primheader, "NSTARS", -1);
    cat->healpix = qfits_header_getint(primheader, "HEALPIX", -1);
    cat->hpnside = qfits_header_getint(primheader, "HPNSIDE", 1);
    if (fits_check_endian(primheader)) {
        ERROR("Catalog file was written with wrong endianness");
        return -1;
    }
    if (cat->numstars == -1) {
        ERROR("Couldn't find NSTARS header in catalog file.");
        return -1;
    }

    chunk->nrows = cat->numstars;
    return 0;
}

static int callback_read_tagalong(fitsbin_t* fb, fitsbin_chunk_t* chunk) {
    catalog* cat = chunk->userdata;
    chunk->nrows = cat->numstars;
    return 0;
}

static catalog* new_catalog(const char* fn, anbool writing) {
    catalog* cat;
    fitsbin_chunk_t chunk;

    cat = calloc(1, sizeof(catalog));
    if (!cat) {
        fprintf(stderr, "catalog_open: malloc failed.\n");
    }

    if (writing)
        cat->fb = fitsbin_open_for_writing(fn);
    else
        cat->fb = fitsbin_open(fn);
    if (!cat->fb) {
        ERROR("Failed to create fitsbin");
        return NULL;
    }

    memset(&chunk, 0, sizeof(fitsbin_chunk_t));

    // NOTE -- the order these are added MUST match the CHUNK_XYZ, CHUNK_MAG, etc
    // ordering.

    // Star positions
    fitsbin_chunk_init(&chunk);
    chunk.tablename = "xyz";
    chunk.required = 1;
    chunk.callback_read_header = callback_read_header;
    chunk.userdata = cat;
    chunk.itemsize = DIM_STARS * sizeof(double);
    fitsbin_add_chunk(cat->fb, &chunk);
    fitsbin_chunk_reset(&chunk);

    // Star magnitudes
    chunk.tablename = "mag";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_tagalong;
    chunk.userdata = cat;
    chunk.itemsize = sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);
    fitsbin_chunk_reset(&chunk);

    // Star magnitude errors
    chunk.tablename = "mag_err";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_tagalong;
    chunk.userdata = cat;
    chunk.itemsize = sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);
    fitsbin_chunk_reset(&chunk);

    // Sigmas
    chunk.tablename = "sigma_radec";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_tagalong;
    chunk.userdata = cat;
    chunk.itemsize = 2 * sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);
    fitsbin_chunk_reset(&chunk);

    // Motions
    chunk.tablename = "proper_motion";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_tagalong;
    chunk.userdata = cat;
    chunk.itemsize = 2 * sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);
    fitsbin_chunk_reset(&chunk);

    // Sigma Motions
    chunk.tablename = "sigma_pm";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_tagalong;
    chunk.userdata = cat;
    chunk.itemsize = 2 * sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);
    fitsbin_chunk_reset(&chunk);

    // Ids
    chunk.tablename = "starid";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_tagalong;
    chunk.userdata = cat;
    chunk.itemsize = sizeof(uint64_t);
    fitsbin_add_chunk(cat->fb, &chunk);
    fitsbin_chunk_clean(&chunk);

    return cat;
}

anbool catalog_has_mag(const catalog* cat) {
    return (cat->mag != NULL);
}

catalog* catalog_open(char* fn) {
    catalog* cat = NULL;

    cat = new_catalog(fn, FALSE);
    if (!cat)
        goto bailout;
    if (fitsbin_read(cat->fb)) {
        ERROR("catalog: fitsbin_read() failed");
        goto bailout;
    }
    cat->stars   = xyz_chunk(cat)->data;
    cat->mag     = mag_chunk(cat)->data;
    cat->mag_err = mag_err_chunk(cat)->data;
    cat->sigma_radec   = get_chunk(cat, CHUNK_SIG   )->data;
    cat->proper_motion = get_chunk(cat, CHUNK_PM    )->data;
    cat->sigma_pm      = get_chunk(cat, CHUNK_SIGPM )->data;
    cat->starids       = get_chunk(cat, CHUNK_STARID)->data;
    return cat;

 bailout:
    if (cat)
        catalog_close(cat);
    return NULL;
}

static void add_default_header_vals(catalog* cat) {
    qfits_header* hdr;
    hdr = catalog_get_header(cat);
    qfits_header_add(hdr, "AN_FILE", "OBJS", "This file has a list of object positions.", NULL);
    fits_add_endian(hdr);
    fits_add_double_size(hdr);
    qfits_header_add(hdr, "NSTARS", "0", "Number of stars in this file.", NULL);
    qfits_header_add(hdr, "HEALPIX", "-1", "Healpix covered by this catalog, with Nside=HPNSIDE", NULL);
    qfits_header_add(hdr, "HPNSIDE", "-1", "Nside of HEALPIX.", NULL);
    qfits_header_add(hdr, "COMMENT", "This is a flat array of XYZ for each catalog star.", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "  (ie, star position on the unit sphere)", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "  (stored as three native-{endian,size} doubles)", NULL, NULL);
}

catalog* catalog_open_for_writing(char* fn)  {
    catalog* cat;

    cat = new_catalog(fn, TRUE);
    if (!cat)
        goto bailout;
    cat->hpnside = 1;
    add_default_header_vals(cat);
    return cat;

 bailout:
    if (cat)
        catalog_close(cat);
    return NULL;
}

qfits_header* catalog_get_header(catalog* cat) {
    return fitsbin_get_primary_header(cat->fb);
}

int catalog_write_header(catalog* cat) {
    fitsbin_t* fb = cat->fb;

    if (fitsbin_write_primary_header(fb) ||
        fitsbin_write_chunk_header(fb, xyz_chunk(cat))) {
        ERROR("Failed to write catalog header");
        return -1;
    }
    return 0;
}

int catalog_fix_header(catalog* cat) {
    qfits_header* hdr;
    fitsbin_t* fb = cat->fb;

    hdr = catalog_get_header(cat);
    // fill in the real values...
    fits_header_mod_int(hdr, "NSTARS", cat->numstars, "Number of stars.");
    fits_header_mod_int(hdr, "HEALPIX", cat->healpix, "Healpix covered by this catalog, with Nside=HPNSIDE");
    fits_header_mod_int(hdr, "HPNSIDE", cat->hpnside, "Nside of HEALPIX.");

    if (fitsbin_fix_primary_header(fb) ||
        fitsbin_fix_chunk_header(fb, xyz_chunk(cat))) {
        ERROR("Failed to fix catalog header");
        return -1;
    }
    return 0;
}

double* catalog_get_base(catalog* cat) {
    return cat->stars;
}

double* catalog_get_star(catalog* cat, int sid) {
    if (sid >= cat->numstars) {
        fflush(stdout);
        fprintf(stderr, "catalog: asked for star %u, but catalog size is only %u.\n",
                sid, cat->numstars);
        return NULL;
    }
    return cat->stars + sid * 3;
}

int catalog_write_star(catalog* cat, double* star) {
    if (fitsbin_write_item(cat->fb, xyz_chunk(cat), star)) {
        fprintf(stderr, "Failed to write catalog data to file %s: %s\n",
                cat->fb->filename, strerror(errno));
        return -1;
    }
    cat->numstars++;
    return 0;
}

int write_floats(catalog* cat, int chunknum, 
                 const char* name, fl* list, int nperstar) {
    int i;
    int B = 1000;
    fitsbin_chunk_t* chunk = get_chunk(cat, chunknum);
    if (!list || (fl_size(list) != cat->numstars * nperstar)) {
        ERROR("Number of %ss (%i) doesn't match number of stars (%i)",
              name, list ? fl_size(list) : 0, cat->numstars);
        return -1;
    }

    if (fitsbin_write_chunk_header(cat->fb, chunk)) {
        ERROR("Failed to write %ss header", name);
        return -1;
    }
    for (i=0; i<cat->numstars; i+=B) {
        float data[nperstar * B];
        int n = MIN(i+B, cat->numstars) - i;
        fl_copy(list, i*nperstar, nperstar*n, data);
        if (fitsbin_write_items(cat->fb, chunk, data, n)) {
            ERROR("Failed to write %s for stars %i to %i", name, i, i+n-1);
            return -1;
        }
    }
    if (fitsbin_fix_chunk_header(cat->fb, chunk)) {
        ERROR("Failed to fix %ss header", name);
        return -1;
    }
    return 0;
}

int catalog_write_mags(catalog* cat) {
    return write_floats(cat, CHUNK_MAG, "magnitude", cat->maglist, 1);
}

int catalog_write_mag_errs(catalog* cat) {
    return write_floats(cat, CHUNK_MAG_ERR, "magnitude errors", cat->magerrlist, 1);
}

int catalog_write_sigmas(catalog* cat) {
    return write_floats(cat, CHUNK_SIG, "sigma", cat->siglist, 2);
}

int catalog_write_pms(catalog* cat) {
    return write_floats(cat, CHUNK_PM, "proper motion", cat->pmlist, 2);
}

int catalog_write_sigma_pms(catalog* cat) {
    return write_floats(cat, CHUNK_SIGPM, "sigma proper motion", cat->sigpmlist, 2);
}

int catalog_write_ids(catalog* cat) {
    int i;
    char* name = "id";
    int chunknum = CHUNK_STARID;
    fitsbin_chunk_t* chunk = get_chunk(cat, chunknum);
    if (!cat->idlist || (bl_size(cat->idlist) != cat->numstars)) {
        ERROR("Number of %ss (%i) doesn't match number of stars (%i)",
              name, cat->idlist ? bl_size(cat->idlist) : 0, cat->numstars);
        return -1;
    }

    if (fitsbin_write_chunk_header(cat->fb, chunk)) {
        ERROR("Failed to write %ss header", name);
        return -1;
    }
    for (i=0; i<cat->numstars; i++)
        if (fitsbin_write_item(cat->fb, chunk, bl_access(cat->idlist, i))) {
            ERROR("Failed to write %s for star %i", name, i);
            return -1;
        }
    if (fitsbin_fix_chunk_header(cat->fb, chunk)) {
        ERROR("Failed to fix %ss header", name);
        return -1;
    }
    return 0;
}

void addfloat(fl** list, float val) {
    if (!(*list))
        *list = fl_new(256);
    fl_append(*list, val);
}

void catalog_add_mag(catalog* cat, float mag) {
    addfloat(&(cat->maglist), mag);
}
void catalog_add_mag_err(catalog* cat, float magerr) {
    addfloat(&(cat->magerrlist), magerr);
}
void catalog_add_sigmas(catalog* cat, float sra, float sdec) {
    addfloat(&(cat->siglist), sra);
    addfloat(&(cat->siglist), sdec);
}
void catalog_add_pms(catalog* cat, float sra, float sdec) {
    addfloat(&(cat->pmlist), sra);
    addfloat(&(cat->pmlist), sdec);
}
void catalog_add_sigma_pms(catalog* cat, float sra, float sdec) {
    addfloat(&(cat->sigpmlist), sra);
    addfloat(&(cat->sigpmlist), sdec);
}
void catalog_add_id(catalog* cat, uint64_t id) {
    if (!cat->idlist)
        cat->idlist = bl_new(256, sizeof(uint64_t));
    bl_append(cat->idlist, &id);
}

int catalog_close(catalog* cat) {
    int rtn;
    if (!cat) return 0;
    rtn = fitsbin_close(cat->fb);
    fl_free(cat->maglist);
    fl_free(cat->magerrlist);
    fl_free(cat->siglist);
    fl_free(cat->pmlist);
    fl_free(cat->sigpmlist);
    bl_free(cat->idlist);
    free(cat);
    return rtn;
}

