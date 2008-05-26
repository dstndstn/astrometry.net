/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "catalog.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "mathutil.h"
#include "errors.h"

#define CHUNK_XYZ    0
#define CHUNK_MAG    1
#define CHUNK_SIG    2
#define CHUNK_PM     3
#define CHUNK_SIGPM  4
#define CHUNK_STARID 5

static fitsbin_chunk_t* xyz_chunk(catalog* cat) {
    return fitsbin_get_chunk(cat->fb, CHUNK_XYZ);
}
static fitsbin_chunk_t* mag_chunk(catalog* cat) {
    return fitsbin_get_chunk(cat->fb, CHUNK_MAG);
}
static fitsbin_chunk_t* get_chunk(catalog* cat, int i) {
    return fitsbin_get_chunk(cat->fb, i);
}

static int callback_read_header(qfits_header* primheader, qfits_header* header,
								size_t* expected, void* userdata) {
	catalog* cat = userdata;

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

    *expected = cat->numstars * DIM_STARS * sizeof(double);
	return 0;
}

static int callback_read_mags(qfits_header* primheader, qfits_header* header,
                              size_t* expected, void* userdata) {
	catalog* cat = userdata;
    *expected = cat->numstars * sizeof(float);
	return 0;
}

static int callback_read_sigmas(qfits_header* primheader, qfits_header* header,
                                size_t* expected, void* userdata) {
	catalog* cat = userdata;
    *expected = cat->numstars * 2 * sizeof(float);
	return 0;
}
#define callback_read_pms callback_read_sigmas
#define callback_read_sigma_pms callback_read_sigmas

static int callback_read_ids(qfits_header* primheader, qfits_header* header,
                                size_t* expected, void* userdata) {
	catalog* cat = userdata;
    *expected = cat->numstars * sizeof(uint64_t);
	return 0;
}

static catalog* new_catalog(const char* fn, bool writing) {
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

    // Star positions
    chunk.tablename = "xyz";
    chunk.required = 1;
    chunk.callback_read_header = callback_read_header;
    chunk.userdata = cat;
	chunk.itemsize = DIM_STARS * sizeof(double);
    fitsbin_add_chunk(cat->fb, &chunk);

    // Star magnitudes
    chunk.tablename = "mags";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_mags;
    chunk.userdata = cat;
    chunk.itemsize = sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);

    // Sigmas
    chunk.tablename = "sigma_radec";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_sigmas;
    chunk.userdata = cat;
    chunk.itemsize = 2 * sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);

    // Motions
    chunk.tablename = "proper_motion";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_pms;
    chunk.userdata = cat;
    chunk.itemsize = 2 * sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);

    // Sigma Motions
    chunk.tablename = "sigma_pm";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_pms;
    chunk.userdata = cat;
    chunk.itemsize = 2 * sizeof(float);
    fitsbin_add_chunk(cat->fb, &chunk);

    // Ids
    chunk.tablename = "starids";
    chunk.required = 0;
    chunk.callback_read_header = callback_read_ids;
    chunk.userdata = cat;
    chunk.itemsize = sizeof(uint64_t);
    fitsbin_add_chunk(cat->fb, &chunk);

    return cat;
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
	cat->stars = xyz_chunk(cat)->data;
    cat->mags  = mag_chunk(cat)->data;
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
    xyz_chunk(cat)->nrows = cat->numstars;

	if (fitsbin_write_primary_header(fb) ||
		fitsbin_write_chunk_header(fb, CHUNK_XYZ)) {
		ERROR("Failed to write catalog header");
		return -1;
	}
	return 0;
}

int catalog_fix_header(catalog* cat) {
	qfits_header* hdr;
	fitsbin_t* fb = cat->fb;
    xyz_chunk(cat)->nrows = cat->numstars;

	hdr = catalog_get_header(cat);
	// fill in the real values...
    fits_header_mod_int(hdr, "NSTARS", cat->numstars, "Number of stars.");
    fits_header_mod_int(hdr, "HEALPIX", cat->healpix, "Healpix covered by this catalog, with Nside=HPNSIDE");
    fits_header_mod_int(hdr, "HPNSIDE", cat->hpnside, "Nside of HEALPIX.");

	if (fitsbin_fix_primary_header(fb) ||
		fitsbin_fix_chunk_header(fb, CHUNK_XYZ)) {
        ERROR("Failed to fix catalog header");
		return -1;
	}
	return 0;
}

/*
void catalog_compute_radecminmax(catalog* cat) {
	double ramin, ramax, decmin, decmax;
	int i;
	ramin = HUGE_VAL;
	ramax = -HUGE_VAL;
	decmin = HUGE_VAL;
	decmax = -HUGE_VAL;
	for (i = 0; i < cat->numstars; i++) {
		double* xyz;
		double ra, dec;
		xyz = catalog_get_star(cat, i);
		ra = xy2ra(xyz[0], xyz[1]);
		dec = z2dec(xyz[2]);
		if (ra > ramax)
			ramax = ra;
		if (ra < ramin)
			ramin = ra;
		if (dec > decmax)
			decmax = dec;
		if (dec < decmin)
			decmin = dec;
	}
	cat->ramin = ramin;
	cat->ramax = ramax;
	cat->decmin = decmin;
	cat->decmax = decmax;
}
 */

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
	if (fitsbin_write_item(cat->fb, CHUNK_XYZ, star)) {
		fprintf(stderr, "Failed to write catalog data to file %s: %s\n",
                cat->fb->filename, strerror(errno));
		return -1;
	}
	cat->numstars++;
	return 0;
}

int write_floats(catalog* cat, int chunknum, 
                  const char* name, fl* list, int nperstar) {
	fitsbin_chunk_t* chunk;
    int i, k;
    if (!list || (fl_size(list) != cat->numstars * nperstar)) {
        ERROR("Number of %ss (%i) doesn't match number of stars (%i)",
              name, list ? fl_size(list) : 0, cat->numstars);
        return -1;
    }
    chunk = get_chunk(cat, chunknum);
    chunk->nrows = cat->numstars;

    if (fitsbin_write_chunk_header(cat->fb, chunknum)) {
        ERROR("Failed to write %ss header", name);
        return -1;
    }
    for (i=0; i<chunk->nrows; i++)
        for (k=0; k<nperstar; k++)
            if (fitsbin_write_item(cat->fb, chunknum, fl_access(list, i*nperstar + k))) {
                ERROR("Failed to write %s for star %i", name, i);
                return -1;
            }
    if (fitsbin_fix_chunk_header(cat->fb, chunknum)) {
        ERROR("Failed to fix %ss header", name);
        return -1;
    }
	return 0;
}

int catalog_write_mags(catalog* cat) {
    return write_floats(cat, CHUNK_MAG, "magnitude", cat->maglist, 1);
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
	fitsbin_chunk_t* chunk;
    int i;
    char* name = "id";
    int chunknum = CHUNK_STARID;
    if (!cat->idlist || (bl_size(cat->idlist) != cat->numstars)) {
        ERROR("Number of %ss (%i) doesn't match number of stars (%i)",
              name, cat->idlist ? bl_size(cat->idlist) : 0, cat->numstars);
        return -1;
    }
    chunk = get_chunk(cat, chunknum);
    chunk->nrows = cat->numstars;

    if (fitsbin_write_chunk_header(cat->fb, chunknum)) {
        ERROR("Failed to write %ss header", name);
        return -1;
    }
    for (i=0; i<chunk->nrows; i++)
        if (fitsbin_write_item(cat->fb, chunknum, bl_access(cat->idlist, i))) {
            ERROR("Failed to write %s for star %i", name, i);
            return -1;
        }
    if (fitsbin_fix_chunk_header(cat->fb, chunknum)) {
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
	free(cat);
    fl_free(cat->maglist);
    fl_free(cat->siglist);
    fl_free(cat->pmlist);
    fl_free(cat->sigpmlist);
    bl_free(cat->idlist);
    return rtn;
}

