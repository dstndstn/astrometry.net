/*
 This file was added by the Astrometry.net team.
 Copyright 2007-2013 Dustin Lang.

 Licensed under GPL v2 or later.
 */

#ifndef ANQFITS_H
#define ANQFITS_H

#include <stdint.h>

#include "astrometry/qfits_header.h"
#include "astrometry/qfits_table.h"
#include "astrometry/qfits_keywords.h"
#include "astrometry/qfits_std.h"
#include "astrometry/qfits_image.h"
#include "astrometry/qfits_tools.h"
#include "astrometry/qfits_time.h"


int fits_get_atom_size(tfits_type type);


/**
 Converts data between different FITS types.

 Does NO checking, rounding, or anything smart - just uses C casts.

 ASSUMES the data have already been flipped to the local host's endianness.
 */
int fits_convert_data(void* dest, int deststride, tfits_type desttype,
                      const void* src, int srcstride, tfits_type srctype,
                      int arraysize, size_t N);

int fits_convert_data_2(void* vdest, int deststride, tfits_type desttype,
                        const void* vsrc, int srcstride, tfits_type srctype,
                        int arraysize, size_t N,
                        double bzero, double bscale);




typedef struct {
    int naxis;
    off_t width;
    off_t height;
    off_t planes;
    int bpp;
    int bitpix;
    double bscale;
    double bzero;
} anqfits_image_t;

anqfits_image_t* anqfits_image_new(void);
void anqfits_image_free(anqfits_image_t*);


// Everything we know about a FITS extension.
typedef struct {
    // Offsets to header, in FITS blocks
    // --> int works for ~12 TB files.
    int hdr_start;
    // Header size
    int hdr_size;
    // Offsets to data
    int data_start;
    // Data size
    int data_size;
    qfits_header* header;
    qfits_table* table;
    anqfits_image_t* image;
} anqfits_ext_t;

typedef struct {
    char* filename;
    int Nexts;    // # of extensions in file
    anqfits_ext_t* exts;
    off_t filesize ; // File size in FITS blocks
} anqfits_t;


anqfits_t* anqfits_open(const char* filename);

// Open the given file, but only parse up to the given HDU number.
// Attempts to get headers or data beyond that HDU will fail, and the
// number of HDUs the file is reported to contain will be hdu+1.
anqfits_t* anqfits_open_hdu(const char* filename, int hdu);

void anqfits_close(anqfits_t* qf);

int anqfits_n_ext(const anqfits_t* qf);

// In BYTES
off_t anqfits_header_start(const anqfits_t* qf, int ext);

// In BYTES
off_t anqfits_header_size(const anqfits_t* qf, int ext);

// In BYTES
off_t anqfits_data_start(const anqfits_t* qf, int ext);

// In BYTES
off_t anqfits_data_size(const anqfits_t* qf, int ext);

int anqfits_get_data_start_and_size(const anqfits_t* qf, int ext,
                                    off_t* pstart, off_t* psize);
int anqfits_get_header_start_and_size(const anqfits_t* qf, int ext,
                                      off_t* pstart, off_t* psize);

int anqfits_is_table(const anqfits_t* qf, int ext);

qfits_header* anqfits_get_header(const anqfits_t* qf, int ext);
qfits_header* anqfits_get_header2(const char* fn, int ext);

qfits_header* anqfits_get_header_only(const char* fn, int ext);

const qfits_header* anqfits_get_header_const(const anqfits_t* qf, int ext);

// Returns a newly-allocated array containing the raw header bytes for the
// given extension.  (Plus a zero-terminator.)  Places the number of
// bytes returned in *Nbytes (not including the zero-terminator).
char* anqfits_header_get_data(const anqfits_t* qf, int ext, int* Nbytes);

qfits_table* anqfits_get_table(const anqfits_t* qf, int ext);

const qfits_table* anqfits_get_table_const(const anqfits_t* qf, int ext);

anqfits_image_t* anqfits_get_image(const anqfits_t* qf, int ext);

const anqfits_image_t* anqfits_get_image_const(const anqfits_t* qf, int ext);


void* anqfits_readpix(const anqfits_t* qf, int ext,
                      /** Pixel window coordinates (0 for whole image);
                       THESE ARE ZERO-INDEXED, unlike qfits_loadpix,
                       and (x1,y1) or NON-INCLUSIVE. **/
                      int x0, int x1, int y0, int y1,
                      /** The plane you want, from 0 to planes-1 */
                      int            pnum,
                      /** Pixel type you want
                       (PTYPE_FLOAT, PTYPE_INT or PTYPE_DOUBLE) */
                      int            ptype,
                      void* output,
                      int* W, int* H);

/*
 Deprecated // ?
 int anqfits_is_table_2(const anqfits_t* qf, int ext);
 Deprecated // ?
 char* anqfits_query_ext_2(const anqfits_t* qf, int ext, const char* keyword);
 */

#endif

