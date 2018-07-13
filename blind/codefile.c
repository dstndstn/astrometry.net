/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "codefile.h"
#include "starutil.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "errors.h"
#include "quad-utils.h"

void quad_write(codefile_t* codes, quadfile_t* quads,
                unsigned int* quad, startree_t* starkd,
                int dimquads, int dimcodes) {
    double code[DCMAX];
    quad_compute_code(quad, dimquads, starkd, code);
    quad_enforce_invariants(quad, code, dimquads, dimcodes);
    codefile_write_code(codes, code);
    quadfile_write_quad(quads, quad);
}

void quad_write_const(codefile_t* codes, quadfile_t* quads,
                      const unsigned int* quad, startree_t* starkd,
                      int dimquads, int dimcodes) {
    int k;
    unsigned int quadcopy[DQMAX];
    for (k=0; k<dimquads; k++)
        quadcopy[k] = quad[k];
    quad_write(codes, quads, quadcopy, starkd, dimquads, dimcodes);
}

void codefile_compute_field_code(const double* xy, double* code, int dimquads) {
    double Ax, Ay;
    double Bx, By;
    double ABx, ABy;
    double scale, invscale;
    double costheta, sintheta;
    int i;

    Ax = xy[2*0 + 0];
    Ay = xy[2*0 + 1];
    Bx = xy[2*1 + 0];
    By = xy[2*1 + 1];
    ABx = Bx - Ax;
    ABy = By - Ay;
    scale = (ABx * ABx) + (ABy * ABy);
    invscale = 1.0 / scale;
    costheta = (ABy + ABx) * invscale;
    sintheta = (ABy - ABx) * invscale;

    // starting with star C...
    for (i=2; i<dimquads; i++) {
        double Cx, Cy;
        double x, y;
        Cx = xy[2*i + 0];
        Cy = xy[2*i + 1];
        Cx -= Ax;
        Cy -= Ay;
        x =  Cx * costheta + Cy * sintheta;
        y = -Cx * sintheta + Cy * costheta;
        code[2*(i-2)+0] = x;
        code[2*(i-2)+1] = y;
    }
}

void codefile_compute_star_code(const double* starxyz, double* code, int dimquads) {
    quad_compute_star_code(starxyz, code, dimquads);
}


#define CHUNK_CODES 0

static fitsbin_chunk_t* codes_chunk(codefile_t* cf) {
    return fitsbin_get_chunk(cf->fb, CHUNK_CODES);
}

static int callback_read_header(fitsbin_t* fb, fitsbin_chunk_t* chunk) {
    qfits_header* primheader = fitsbin_get_primary_header(fb);
    codefile_t* cf = chunk->userdata;

    cf->dimcodes = qfits_header_getint(primheader, "DIMCODES", 4);
    cf->numcodes = qfits_header_getint(primheader, "NCODES", -1);
    cf->numstars = qfits_header_getint(primheader, "NSTARS", -1);
    cf->index_scale_upper = qfits_header_getdouble(primheader, "SCALE_U", -1.0);
    cf->index_scale_lower = qfits_header_getdouble(primheader, "SCALE_L", -1.0);
    cf->indexid = qfits_header_getint(primheader, "INDEXID", 0);
    cf->healpix = qfits_header_getint(primheader, "HEALPIX", -1);
    cf->hpnside = qfits_header_getint(primheader, "HPNSIDE", 1);

    if ((cf->numcodes == -1) || (cf->numstars == -1) ||
        (cf->index_scale_upper == -1.0) || (cf->index_scale_lower == -1.0)) {
        ERROR("Couldn't find NCODES or NSTARS or SCALE_U or SCALE_L entries in FITS header");
        return -1;
    }
    if (fits_check_endian(primheader)) {
        ERROR("File was written with the wrong endianness");
        return -1;
    }
    chunk->itemsize = cf->dimcodes * sizeof(double);
    chunk->nrows = cf->numcodes;
    return 0;
}

static codefile_t* new_codefile(const char* fn, anbool writing, anbool inmem) {
    fitsbin_chunk_t chunk;
    codefile_t* cf = calloc(1, sizeof(codefile_t));
    if (!cf) {
        SYSERROR("Couldn't calloc a codefile struct");
        return NULL;
    }
    cf->healpix = -1;
    cf->hpnside = 1;

    if (inmem) {
        cf->fb = fitsbin_open_in_memory();
    } else {
        if (writing)
            cf->fb = fitsbin_open_for_writing(fn);
        else
            cf->fb = fitsbin_open(fn);
    }
    if (!cf->fb) {
        ERROR("Failed to create fitsbin");
        return NULL;
    }

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "codes";
    chunk.required = 1;
    chunk.callback_read_header = callback_read_header;
    chunk.userdata = cf;
    fitsbin_add_chunk(cf->fb, &chunk);
    fitsbin_chunk_clean(&chunk);

    return cf;
}

void codefile_get_code(const codefile_t* cf, int codeid, double* code) {
    int i;
    if (codeid >= cf->numcodes) {
        ERROR("Requested codeid %i, but number of codes is %i", codeid, cf->numcodes);
        assert(codeid < cf->numcodes);
        // just carry on - we'll probably segfault.
    }
    for (i=0; i<cf->dimcodes; i++)
        code[i] = cf->codearray[codeid * cf->dimcodes + i];
}

int codefile_close(codefile_t* cf) {
    int rtn;
    if (!cf) return 0;
    rtn = fitsbin_close(cf->fb);
    free(cf);
    return rtn;
}

codefile_t* codefile_open(const char* fn) {
    codefile_t* cf = NULL;

    cf = new_codefile(fn, FALSE, FALSE);
    if (!cf)
        goto bailout;
    if (fitsbin_read(cf->fb)) {
        ERROR("Failed to open codes file");
        goto bailout;
    }
    cf->codearray = codes_chunk(cf)->data;
    return cf;

 bailout:
    if (cf)
        codefile_close(cf);
    return NULL;
}

static codefile_t* open_for_writing(const char* fn) {
    codefile_t* cf;
    qfits_header* hdr;
    if (fn)
        cf = new_codefile(fn, TRUE, FALSE);
    else
        cf = new_codefile(fn, TRUE, TRUE);
    if (!cf)
        goto bailout;
    // default
    cf->dimcodes = 4;

    // add default values to header
    hdr = codefile_get_header(cf);
    fits_add_endian(hdr);
    qfits_header_add(hdr, "AN_FILE", "CODE", "This file lists the code for each quad.", NULL);
    qfits_header_add(hdr, "NCODES", "0", "", NULL);
    qfits_header_add(hdr, "NSTARS", "0", "", NULL);
    fits_header_add_int(hdr, "DIMCODES", cf->dimcodes, "");
    qfits_header_add(hdr, "SCALE_U", "0.0", "", NULL);
    qfits_header_add(hdr, "SCALE_L", "0.0", "", NULL);
    qfits_header_add(hdr, "INDEXID", "0", "", NULL);
    qfits_header_add(hdr, "HEALPIX", "-1", "", NULL);
    qfits_header_add(hdr, "HPNSIDE", "1", "", NULL);
    fits_add_long_comment(hdr, "The first extension contains the codes "
                          "stored as %i native-endian doubles.  "
                          "(the quad location in %i-D code space)", cf->dimcodes, cf->dimcodes);
    return cf;
 bailout:
    if (cf)
        codefile_close(cf);
    return NULL;
}

codefile_t* codefile_open_for_writing(const char* fn) {
    if (!fn) {
        ERROR("Non-NULL filename required");
        return NULL;
    }
    return open_for_writing(fn);
}

codefile_t* codefile_open_in_memory() {
    return open_for_writing(NULL);
}

int codefile_switch_to_reading(codefile_t* cf) {
    if (codefile_fix_header(cf)) {
        ERROR("Failed to fix codes header");
        return -1;
    }
    if (fitsbin_switch_to_reading(cf->fb)) {
        ERROR("Failed to switch to read mode");
        return -1;
    }
    if (fitsbin_read(cf->fb)) {
        ERROR("Failed to open codes file");
        return -1;
    }
    cf->codearray = codes_chunk(cf)->data;
    return 0;
}

int codefile_write_header(codefile_t* cf) {
    fitsbin_t* fb = cf->fb;
    fitsbin_chunk_t* chunk = codes_chunk(cf);
    chunk->itemsize = cf->dimcodes * sizeof(double);
    chunk->nrows = cf->numcodes;

    if (fitsbin_write_primary_header(fb) ||
        fitsbin_write_chunk_header(fb, chunk)) {
        ERROR("Failed to write codefile header");
        return -1;
    }
    return 0;
}

int codefile_fix_header(codefile_t* cf) {
    qfits_header* hdr;
    fitsbin_t* fb = cf->fb;
    fitsbin_chunk_t* chunk = codes_chunk(cf);
    chunk->itemsize = cf->dimcodes * sizeof(double);
    chunk->nrows = cf->numcodes;

    hdr = codefile_get_header(cf);

    // fill in the real values...
    fits_header_mod_int(hdr, "DIMCODES", cf->dimcodes, "Number of values in a code.");
    fits_header_mod_int(hdr, "NCODES", cf->numcodes, "Number of codes.");
    fits_header_mod_int(hdr, "NSTARS", cf->numstars, "Number of stars.");
    fits_header_mod_double(hdr, "SCALE_U", cf->index_scale_upper, "Upper-bound index scale (radians).");
    fits_header_mod_double(hdr, "SCALE_L", cf->index_scale_lower, "Lower-bound index scale (radians).");
    fits_header_mod_int(hdr, "INDEXID", cf->indexid, "Index unique ID.");
    fits_header_mod_int(hdr, "HEALPIX", cf->healpix, "Healpix of this index.");
    fits_header_mod_int(hdr, "HPNSIDE", cf->hpnside, "Nside of the healpixelization");

    if (fitsbin_fix_primary_header(fb) ||
        fitsbin_fix_chunk_header(fb, chunk)) {
        ERROR("Failed to fix code header");
        return -1;
    }
    return 0;
}

int codefile_write_code(codefile_t* cf, double* code) {
    fitsbin_chunk_t* chunk = codes_chunk(cf);
    if (fitsbin_write_item(cf->fb, chunk, code)) {
        ERROR("Failed to write code");
        return -1;
    }
    cf->numcodes++;
    return 0;
}

qfits_header* codefile_get_header(const codefile_t* cf) {
    return fitsbin_get_primary_header(cf->fb);
}

int codefile_dimcodes(const codefile_t* cf) {
    return cf->dimcodes;
}
