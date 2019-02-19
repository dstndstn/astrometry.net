/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "quadfile.h"
#include "qfits_header.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "ioutils.h"
#include "errors.h"
#include "an-endian.h"

#define CHUNK_QUADS 0

static fitsbin_chunk_t* quads_chunk(quadfile_t* qf) {
    return fitsbin_get_chunk(qf->fb, CHUNK_QUADS);
}

static int callback_read_header(fitsbin_t* fb, fitsbin_chunk_t* chunk) {
    qfits_header* primheader = fitsbin_get_primary_header(fb);
    quadfile_t* qf = chunk->userdata;

    qf->dimquads = qfits_header_getint(primheader, "DIMQUADS", 4);
    qf->numquads = qfits_header_getint(primheader, "NQUADS", -1);
    qf->numstars = qfits_header_getint(primheader, "NSTARS", -1);
    qf->index_scale_upper = qfits_header_getdouble(primheader, "SCALE_U", -1.0);
    qf->index_scale_lower = qfits_header_getdouble(primheader, "SCALE_L", -1.0);
    qf->indexid = qfits_header_getint(primheader, "INDEXID", 0);
    qf->healpix = qfits_header_getint(primheader, "HEALPIX", -1);
    qf->hpnside = qfits_header_getint(primheader, "HPNSIDE", 1);

    if ((qf->numquads == -1) || (qf->numstars == -1) ||
        (qf->index_scale_upper == -1.0) || (qf->index_scale_lower == -1.0)) {
        ERROR("Couldn't find NQUADS or NSTARS or SCALE_U or SCALE_L entries in FITS header");
        return -1;
    }
    if (fits_check_endian(primheader)) {
        ERROR("Quad file was written with the wrong endianness");
        return -1;
    }

    chunk->itemsize = qf->dimquads * sizeof(uint32_t);
    chunk->nrows = qf->numquads;
    return 0;
}

static quadfile_t* new_quadfile(const char* fn, anqfits_t* fits, anbool writing) {
    quadfile_t* qf;
    fitsbin_chunk_t chunk;
    qf = calloc(1, sizeof(quadfile_t));
    if (!qf) {
        SYSERROR("Couldn't malloc a quadfile struct");
        return NULL;
    }
    qf->healpix = -1;
    qf->hpnside = 1;

    if (writing)
        if (fn) {
            qf->fb = fitsbin_open_for_writing(fn);
        } else {
            qf->fb = fitsbin_open_in_memory();
        }
    else {
        if (fits)
            qf->fb = fitsbin_open_fits(fits);
        else
            qf->fb = fitsbin_open(fn);
    }
    if (!qf->fb) {
        ERROR("Failed to create fitsbin");
        return NULL;
    }

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "quads";
    chunk.required = 1;
    chunk.callback_read_header = callback_read_header;
    chunk.userdata = qf;
    fitsbin_add_chunk(qf->fb, &chunk);
    fitsbin_chunk_clean(&chunk);
    
    return qf;
}

int quadfile_check(const quadfile_t* qf) {
    int q, i;
    if (qf->dimquads < 3 || qf->dimquads > DQMAX) {
        ERROR("Dimquads has illegal value %i", qf->dimquads);
        return -1;
    }
    for (q=0; q<qf->numquads; q++) {
        unsigned int stars[DQMAX];
        if (quadfile_get_stars(qf, q, stars)) {
            ERROR("Failed to get quad %i of %i", q, qf->numquads);
            return -1;
        }
        for (i=0; i<qf->dimquads; i++) {
            if (stars[i] >= qf->numstars) {
                ERROR("Star ID %i is out of bounds: num stars %i", stars[i], qf->numstars);
                return -1;
            }
        }
    }
    return 0;
}

int quadfile_dimquads(const quadfile_t* qf) {
    return qf->dimquads;
}

int quadfile_nquads(const quadfile_t* qf) {
    return qf->numquads;
}

qfits_header* quadfile_get_header(const quadfile_t* qf) {
    return fitsbin_get_primary_header(qf->fb);
}

static quadfile_t* my_open(const char* fn, anqfits_t* fits) {
    quadfile_t* qf = NULL;
    fitsbin_chunk_t* chunk;

    qf = new_quadfile(fn, fits, FALSE);
    if (!qf)
        goto bailout;
    if (fitsbin_read(qf->fb)) {
        ERROR("Failed to open quads file");
        goto bailout;
    }
    chunk = quads_chunk(qf);
    qf->quadarray = chunk->data;

    // close fd.
    if (qf->fb->fid) {
        if (fclose(qf->fb->fid)) {
            ERROR("Failed to close quadfile FID");
            goto bailout;
        }
        qf->fb->fid = NULL;
    }

    return qf;

 bailout:
    if (qf)
        quadfile_close(qf);
    return NULL;
}

char* quadfile_get_filename(const quadfile_t* qf) {
    return fitsbin_get_filename(qf->fb);
}

quadfile_t* quadfile_open_fits(anqfits_t* fits) {
    return my_open(NULL, fits);
}

quadfile_t* quadfile_open(const char* fn) {
    return my_open(fn, NULL);
}

int quadfile_close(quadfile_t* qf) {
    int rtn;
    if (!qf) return 0;
    rtn = fitsbin_close(qf->fb);
    free(qf);
    return rtn;
}

static quadfile_t* open_for_writing(const char* fn) {
    quadfile_t* qf;
    qfits_header* hdr;
    qf = new_quadfile(fn, NULL, TRUE);
    if (!qf)
        goto bailout;
    qf->dimquads = 4;
    // add default values to header
    hdr = fitsbin_get_primary_header(qf->fb);
    fits_add_endian(hdr);
    qfits_header_add(hdr, "AN_FILE", "QUAD", "This file lists, for each quad, its stars.", NULL);
    qfits_header_add(hdr, "DIMQUADS", "0", "", NULL);
    qfits_header_add(hdr, "NQUADS", "0", "", NULL);
    qfits_header_add(hdr, "NSTARS", "0", "", NULL);
    qfits_header_add(hdr, "SCALE_U", "0.0", "", NULL);
    qfits_header_add(hdr, "SCALE_L", "0.0", "", NULL);
    qfits_header_add(hdr, "INDEXID", "0", "", NULL);
    qfits_header_add(hdr, "HEALPIX", "-1", "", NULL);
    qfits_header_add(hdr, "HPNSIDE", "1", "", NULL);
    fits_add_long_comment(hdr, "The first extension contains the quads "
                          "stored as %i 32-bit native-endian unsigned ints.", qf->dimquads);
    return qf;

 bailout:
    if (qf)
        quadfile_close(qf);
    return NULL;
}

quadfile_t* quadfile_open_for_writing(const char* fn) {
    if (!fn) {
        ERROR("Non-NULL filename required");
        return NULL;
    }
    return open_for_writing(fn);
}

quadfile_t* quadfile_open_in_memory() {
    return open_for_writing(NULL);
}

int quadfile_switch_to_reading(quadfile_t* qf) {
    if (quadfile_fix_header(qf)) {
        ERROR("Failed to fix quads header");
        return -1;
    }
    if (fitsbin_switch_to_reading(qf->fb)) {
        ERROR("Failed to switch to read mode");
        return -1;
    }
    if (fitsbin_read(qf->fb)) {
        ERROR("Failed to open quads file");
        return -1;
    }
    qf->quadarray = quads_chunk(qf)->data;
    return 0;
}

static void add_to_header(qfits_header* hdr, quadfile_t* qf) {
    fits_header_mod_int(hdr, "DIMQUADS", qf->dimquads, "Number of stars in a quad.");
    fits_header_mod_int(hdr, "NQUADS", qf->numquads, "Number of quads.");
    fits_header_mod_int(hdr, "NSTARS", qf->numstars, "Number of stars.");
    fits_header_mod_double(hdr, "SCALE_U", qf->index_scale_upper, "Upper-bound index scale (radians).");
    fits_header_mod_double(hdr, "SCALE_L", qf->index_scale_lower, "Lower-bound index scale (radians).");
    fits_header_mod_int(hdr, "INDEXID", qf->indexid, "Index unique ID.");
    fits_header_mod_int(hdr, "HEALPIX", qf->healpix, "Healpix of this index.");
    fits_header_mod_int(hdr, "HPNSIDE", qf->hpnside, "Nside of the healpixelization");
}

int quadfile_write_header(quadfile_t* qf) {
    fitsbin_t* fb = qf->fb;
    fitsbin_chunk_t* chunk = quads_chunk(qf);
    qfits_header* hdr;
    chunk->itemsize = qf->dimquads * sizeof(uint32_t);
    chunk->nrows = qf->numquads;

    hdr = fitsbin_get_primary_header(fb);
    add_to_header(hdr, qf);

    if (fitsbin_write_primary_header(fb) ||
        fitsbin_write_chunk_header(fb, chunk)) {
        ERROR("Failed to write quadfile header");
        return -1;
    }
    return 0;
}

int quadfile_write_header_to(quadfile_t* qf, FILE* fid) {
    fitsbin_t* fb = qf->fb;
    fitsbin_chunk_t* chunk = quads_chunk(qf);
    qfits_header* hdr;
    chunk->itemsize = qf->dimquads * sizeof(uint32_t);
    chunk->nrows = qf->numquads;
    hdr = fitsbin_get_primary_header(fb);
    add_to_header(hdr, qf);

    if (fitsbin_write_primary_header_to(fb, fid) ||
        fitsbin_write_chunk_header_to(fb, chunk, fid)) {
        ERROR("Failed to write quadfile header");
        return -1;
    }
    return 0;
}

int quadfile_write_quad(quadfile_t* qf, unsigned int* stars) {
    uint32_t* data;
    uint32_t ustars[qf->dimquads];
    int i;
    fitsbin_chunk_t* chunk = quads_chunk(qf);

    if (sizeof(uint32_t) == sizeof(uint)) {
        data = stars;
    } else {
        data = ustars;
        for (i=0; i<qf->dimquads; i++)
            ustars[i] = stars[i];
    }
    if (fitsbin_write_item(qf->fb, chunk, data)) {
        ERROR("Failed to write a quad");
        return -1;
    }
    qf->numquads++;
    return 0;
}

int quadfile_write_all_quads_to(quadfile_t* qf, FILE* fid) {
    fitsbin_chunk_t* chunk = quads_chunk(qf);
    if (fitsbin_write_items_to(chunk, qf->quadarray, quadfile_nquads(qf), fid)) {
        ERROR("Failed to write %i quads", quadfile_nquads(qf));
        return -1;
    }
    return 0;
}

int quadfile_fix_header(quadfile_t* qf) {
    qfits_header* hdr;
    fitsbin_t* fb = qf->fb;
    fitsbin_chunk_t* chunk = quads_chunk(qf);

    chunk->itemsize = qf->dimquads * sizeof(uint32_t);
    chunk->nrows = qf->numquads;

    hdr = fitsbin_get_primary_header(fb);
    add_to_header(hdr, qf);

    if (fitsbin_fix_primary_header(fb) ||
        fitsbin_fix_chunk_header(fb, chunk)) {
        ERROR("Failed to fix quad header");
        return -1;
    }
    return 0;
}

double quadfile_get_index_scale_upper_arcsec(const quadfile_t* qf) {
    return rad2arcsec(qf->index_scale_upper);
}

double quadfile_get_index_scale_lower_arcsec(const quadfile_t* qf) {
    return rad2arcsec(qf->index_scale_lower);
}

int quadfile_get_stars(const quadfile_t* qf, unsigned int quadid, unsigned int* stars) {
    int i;
    if (quadid >= qf->numquads) {
        ERROR("Requested quadid %i, but number of quads is %i",	quadid, qf->numquads);
        assert(quadid < qf->numquads);
        return -1;
    }

    for (i=0; i<qf->dimquads; i++) {
        stars[i] = qf->quadarray[quadid * qf->dimquads + i];
    }
    return 0;
}



