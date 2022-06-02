/*
 This file was added by the Astrometry.net team.
 Copyright 2007,2010 Dustin Lang.
 Copyright 2013 Dustin Lang.

 Licensed under GPL v2 or later.
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <sys/mman.h>

#include "anqfits.h"
#include "qfits_std.h"
#include "qfits_error.h"
#include "qfits_tools.h"
#include "qfits_table.h"
#include "qfits_memory.h"
#include "qfits_rw.h"
#include "qfits_card.h"

#include "qfits_image.h"
#include "qfits_convert.h"
#include "qfits_byteswap.h"

#include "ioutils.h"
#include "errors.h"

#if 0
//#if 1
#define qdebug( code ) { code }
#define debug printf
#else
#define qdebug( code ) {}
#define debug(args...)
#endif

int fits_get_atom_size(tfits_type type) {
	int atomsize = -1;
	switch (type) {
	case TFITS_BIN_TYPE_A:
	case TFITS_BIN_TYPE_X:
	case TFITS_BIN_TYPE_L:
	case TFITS_BIN_TYPE_B:
		atomsize = 1;
		break;
	case TFITS_BIN_TYPE_I:
		atomsize = 2;
		break;
	case TFITS_BIN_TYPE_J:
	case TFITS_BIN_TYPE_E:
		atomsize = 4;
		break;
	case TFITS_BIN_TYPE_K:
	case TFITS_BIN_TYPE_D:
		atomsize = 8;
		break;
	default:
		break;
	}
	return atomsize;
}

int fits_convert_data_2(void* vdest, int deststride, tfits_type desttype,
                        const void* vsrc, int srcstride, tfits_type srctype,
                        int arraysize, size_t N,
                        double bzero, double bscale) {
	size_t i;
    int j;
    char* dest = vdest;
    const char* src = vsrc;
    int destatomsize = fits_get_atom_size(desttype);
    int srcatomsize = fits_get_atom_size(srctype);
    anbool scaling = (bzero != 0.0) || (bscale != 1.0);

    // this loop is over rows of data
    for (i=0; i<N; i++) {
        // store local pointers so we can stride over the array, without
        // affecting the row stride.
        char* adest = dest;
        const char* asrc = src;
        int64_t ival = 0;
        double  dval = 0;

        // this loop is over elements of the array, if the column contains an array.
        // (ie, for scalar columns, arraysize is 1.)
        for (j=0; j<arraysize; j++) {
            anbool src_is_int = TRUE;
            switch (srctype) {
            case TFITS_BIN_TYPE_A:
            case TFITS_BIN_TYPE_X:
            case TFITS_BIN_TYPE_B:
                ival = *((uint8_t*)asrc);
                break;
            case TFITS_BIN_TYPE_L:
                // these are actually the characters 'T' and 'F'.
                ival = *((uint8_t*)asrc);
                if (ival == 'T')
                    ival = 1;
                else
                    ival = 0;
                break;
            case TFITS_BIN_TYPE_I:
                ival = *((int16_t*)asrc);
                break;
            case TFITS_BIN_TYPE_J:
                ival = *((int32_t*)asrc);
                break;
            case TFITS_BIN_TYPE_K:
                ival = *((int64_t*)asrc);
                break;
            case TFITS_BIN_TYPE_E:
                dval = *((float*)asrc);
                src_is_int = FALSE;
                break;
            case TFITS_BIN_TYPE_D:
                dval = *((double*)asrc);
                src_is_int = FALSE;
                break;
            default:
                fprintf(stderr, "fits_convert_data: unknown source type %i\n", srctype);
                assert(0);
                return -1;
            }

            if (scaling) {
                if (src_is_int) {
                    src_is_int = FALSE;
                    dval = ival;
                }
                dval = (bzero + dval * bscale);
            }

            switch (desttype) {
            case TFITS_BIN_TYPE_A:
            case TFITS_BIN_TYPE_X:
            case TFITS_BIN_TYPE_B:
                *((uint8_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_L:
                *((char*)adest) = (src_is_int ? ival : dval) ? 'T' : 'F';
                break;
            case TFITS_BIN_TYPE_I:
                *((int16_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_J:
                *((int32_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_K:
                *((int64_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_E:
                *((float*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_D:
                *((double*)adest) = (src_is_int ? ival : dval);
                break;
            default:
                fprintf(stderr, "fits_convert_data: unknown destination type %i\n", desttype);
                assert(0);
                return -1;
            }

            asrc  += srcatomsize;
            adest += destatomsize;
        }

        dest += deststride;
        src  +=  srcstride;
    }
    return 0;
}


int fits_convert_data(void* vdest, int deststride, tfits_type desttype,
                      const void* vsrc, int srcstride, tfits_type srctype,
                      int arraysize, size_t N) {
    return fits_convert_data_2(vdest, deststride, desttype,
                               vsrc, srcstride, srctype,
                               arraysize, N, 0.0, 1.0);
}




// from ioutils.c
/*
static
void get_mmap_size(size_t start, size_t size, off_t* mapstart, size_t* mapsize, int* pgap) {
	int ps = getpagesize();
	int gap = start % ps;
	// start must be a multiple of pagesize.
	*mapstart = start - gap;
	*mapsize  = size  + gap;
	*pgap = gap;
}
// from fitsioutils.c:
static int fits_get_atom_size(tfits_type type) {
	int atomsize = -1;
	switch (type) {
	case TFITS_BIN_TYPE_A:
	case TFITS_BIN_TYPE_X:
	case TFITS_BIN_TYPE_L:
	case TFITS_BIN_TYPE_B:
		atomsize = 1;
		break;
	case TFITS_BIN_TYPE_I:
		atomsize = 2;
		break;
	case TFITS_BIN_TYPE_J:
	case TFITS_BIN_TYPE_E:
		atomsize = 4;
		break;
	case TFITS_BIN_TYPE_K:
	case TFITS_BIN_TYPE_D:
		atomsize = 8;
		break;
	default:
		break;
	}
	return atomsize;
}

//////////////////// copied from fitsioutils.c !!! ///////////////////
static 
int fits_convert_data_2(void* vdest, int deststride, tfits_type desttype,
                        const void* vsrc, int srcstride, tfits_type srctype,
                        int arraysize, size_t N,
                        double bzero, double bscale) {
	size_t i;
    int j;
    char* dest = vdest;
    const char* src = vsrc;
    int destatomsize = fits_get_atom_size(desttype);
    int srcatomsize = fits_get_atom_size(srctype);
    int scaling = (bzero != 0.0) || (bscale != 1.0);

    // this loop is over rows of data
    for (i=0; i<N; i++) {
        // store local pointers so we can stride over the array, without
        // affecting the row stride.
        char* adest = dest;
        const char* asrc = src;
        int64_t ival = 0;
        double  dval = 0;

        // this loop is over elements of the array, if the column
        // contains an array.  (ie, for scalar columns, arraysize is
        // 1.)
        for (j=0; j<arraysize; j++) {
            int src_is_int = 1;
            switch (srctype) {
            case TFITS_BIN_TYPE_A:
            case TFITS_BIN_TYPE_X:
            case TFITS_BIN_TYPE_B:
                ival = *((uint8_t*)asrc);
                break;
            case TFITS_BIN_TYPE_L:
                // these are actually the characters 'T' and 'F'.
                ival = *((uint8_t*)asrc);
                if (ival == 'T')
                    ival = 1;
                else
                    ival = 0;
                break;
            case TFITS_BIN_TYPE_I:
                ival = *((int16_t*)asrc);
                break;
            case TFITS_BIN_TYPE_J:
                ival = *((int32_t*)asrc);
                break;
            case TFITS_BIN_TYPE_K:
                ival = *((int64_t*)asrc);
                break;
            case TFITS_BIN_TYPE_E:
                dval = *((float*)asrc);
                src_is_int = 0;
                break;
            case TFITS_BIN_TYPE_D:
                dval = *((double*)asrc);
                src_is_int = 0;
                break;
            default:
                fprintf(stderr, "fits_convert_data: unknown source type %i\n", srctype);
                assert(0);
                return -1;
            }

            if (scaling) {
                if (src_is_int) {
                    src_is_int = 0;
                    dval = ival;
                }
                dval = (bzero + dval * bscale);
            }

            switch (desttype) {
            case TFITS_BIN_TYPE_A:
            case TFITS_BIN_TYPE_X:
            case TFITS_BIN_TYPE_B:
                *((uint8_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_L:
                *((char*)adest) = (src_is_int ? ival : dval) ? 'T' : 'F';
                break;
            case TFITS_BIN_TYPE_I:
                *((int16_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_J:
                *((int32_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_K:
                *((int64_t*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_E:
                *((float*)adest) = (src_is_int ? ival : dval);
                break;
            case TFITS_BIN_TYPE_D:
                *((double*)adest) = (src_is_int ? ival : dval);
                break;
            default:
                fprintf(stderr, "fits_convert_data: unknown destination type %i\n", desttype);
                assert(0);
                return -1;
            }

            asrc  += srcatomsize;
            adest += destatomsize;
        }

        dest += deststride;
        src  +=  srcstride;
    }
    return 0;
}
 */

int qfits_is_table(const char* filename, int ext) {
    int rtn;
    anqfits_t* anq = anqfits_open_hdu(filename, ext);
    if (!anq) {
        fprintf(stderr, "qfits_is_table: failed to open \"%s\"", filename);
        return -1;
    }
    rtn = anqfits_is_table(anq, ext);
    anqfits_close(anq);
    return rtn;
}

int anqfits_is_table(const anqfits_t* qf, int ext) {
    const qfits_header* hdr;
    int ttype;
    hdr = anqfits_get_header_const(qf, ext);
    if (!hdr) {
        printf("Failed to read header of ext %i", ext);
        return -1;
    }
    ttype = qfits_is_table_header(hdr);
    if (ttype == QFITS_ASCIITABLE) {
        return 1;
    }
    if (ttype == QFITS_BINTABLE) {
        return 1;
    }
    return 0;
}


int anqfits_n_ext(const anqfits_t* qf) {
    return qf->Nexts;
}

off_t anqfits_header_start(const anqfits_t* qf, int ext) {
    assert(ext >= 0 && ext < qf->Nexts);
    if (ext < 0 || ext >= qf->Nexts) {
        ERROR("Failed to get header start for file \"%s\" ext %i: ext not in range [0, %i)",
              qf->filename, ext, qf->Nexts);
        return -1;
    }
    return (off_t)qf->exts[ext].hdr_start * (off_t)FITS_BLOCK_SIZE;
}

off_t anqfits_header_size(const anqfits_t* qf, int ext) {
    assert(ext >= 0 && ext < qf->Nexts);
    if (ext < 0 || ext >= qf->Nexts) {
        ERROR("Failed to get header size for file \"%s\" ext %i: ext not in range [0, %i)",
              qf->filename, ext, qf->Nexts);
        return -1;
    }
    return (off_t)qf->exts[ext].hdr_size * (off_t)FITS_BLOCK_SIZE;
}

off_t anqfits_data_start(const anqfits_t* qf, int ext) {
    assert(ext >= 0 && ext < qf->Nexts);
    if (ext < 0 || ext >= qf->Nexts) {
        ERROR("Failed to get data start for file \"%s\" ext %i: ext not in range [0, %i)",
              qf->filename, ext, qf->Nexts);
        return -1;
    }
    return (off_t)qf->exts[ext].data_start * (off_t)FITS_BLOCK_SIZE;
}

off_t anqfits_data_size(const anqfits_t* qf, int ext) {
    assert(ext >= 0 && ext < qf->Nexts);
    if (ext < 0 || ext >= qf->Nexts) {
        ERROR("Failed to get data size for file \"%s\" ext %i: ext not in range [0, %i)",
              qf->filename, ext, qf->Nexts);
        return -1;
    }
    return (off_t)qf->exts[ext].data_size * (off_t)FITS_BLOCK_SIZE;
}

int anqfits_get_data_start_and_size(const anqfits_t* qf, int ext,
                                    off_t* pstart, off_t* psize) {
    if (pstart) {
        *pstart = anqfits_data_start(qf, ext);
        if (*pstart == -1)
            return -1;
    }
    if (psize) {
        *psize = anqfits_data_size(qf, ext);
        if (*psize == -1)
            return -1;
    }
    return 0;
}

int anqfits_get_header_start_and_size(const anqfits_t* qf, int ext,
                                    off_t* pstart, off_t* psize) {
    if (pstart) {
        *pstart = anqfits_header_start(qf, ext);
        if (*pstart == -1)
            return -1;
    }
    if (psize) {
        *psize = anqfits_header_size(qf, ext);
        if (*psize == -1)
            return -1;
    }
    return 0;
}

qfits_header* anqfits_get_header(const anqfits_t* qf, int ext) {
    const qfits_header* hdr = anqfits_get_header_const(qf, ext);
    if (!hdr)
        return NULL;
    return qfits_header_copy(hdr);
}

qfits_header* anqfits_get_header2(const char* fn, int ext) {
    qfits_header* hdr;
    anqfits_t* anq = anqfits_open(fn);
    if (!anq) {
        qfits_error("Failed to read FITS file \"%s\"", fn);
        return NULL;
    }
    hdr = anqfits_get_header(anq, ext);
    anqfits_close(anq);
    return hdr;
}

qfits_header* anqfits_get_header_only(const char* fn, int ext) {
    qfits_header* hdr;
    anqfits_t* anq = anqfits_open_hdu(fn, ext);
    if (!anq) {
        qfits_error("Failed to read FITS file \"%s\" to extension %i", fn, ext);
        return NULL;
    }
    hdr = anqfits_get_header(anq, ext);
    anqfits_close(anq);
    return hdr;
}

/*
// copied from util/ioutils.c
static char* file_get_contents_offset(const char* fn, int offset, int size) {
    char* buf;
    FILE* fid;
    fid = fopen(fn, "rb");
    if (!fid) {
        fprintf(stderr, "file_get_contents_offset: failed to open file \"%s\": %s\n", fn, strerror(errno));
        return NULL;
    }
    buf = malloc(size);
    if (!buf) {
        fprintf(stderr, "file_get_contents_offset: couldn't malloc %lu bytes.\n", (long)size);
        return NULL;
    }
	if (offset) {
		if (fseeko(fid, offset, SEEK_SET)) {
			fprintf(stderr, "file_get_contents_offset: failed to fseeko: %s.\n", strerror(errno));
			return NULL;
		}
	}
	if (fread(buf, 1, size, fid) != size) {
        fprintf(stderr, "file_get_contents_offset: failed to read %lu bytes: %s\n", (long)size, strerror(errno));
        free(buf);
        return NULL;
    }
	fclose(fid);
    return buf;
}
 */

const qfits_header* anqfits_get_header_const(const anqfits_t* qf, int ext) {
    assert(ext >= 0 && ext < qf->Nexts);
    if (!qf->exts[ext].header) {
        off_t start, size;
        char* str;
        start = anqfits_header_start(qf, ext);
        size  = anqfits_header_size (qf, ext);
        if ((start == -1) || (size == -1)) {
            ERROR("failed to get header start + size for file \"%s\" extension %i", qf->filename, ext);
            return NULL;
        }
        str = file_get_contents_offset(qf->filename, (int)start, (int)size);
        if (!str) {
            ERROR("failed to read \"%s\" extension %i: offset %i size %i\n", qf->filename, ext, (int)start, (int)size);
            return NULL;
        }
        qf->exts[ext].header = qfits_header_read_hdr_string
            ((unsigned char*)str, (int)size);
    }
    return qf->exts[ext].header;
}

// Returns a newly-allocated array containing the raw header bytes for the
// given extension.
char* anqfits_header_get_data(const anqfits_t* qf, int ext, int* Nbytes) {
    FILE* fid;
    off_t N, nr;
    char* data;
    off_t start;

    start = anqfits_header_start(qf, ext);
    if (start == -1)
        return NULL;
    N = anqfits_header_size(qf, ext);
    if (N == -1)
        return NULL;
    fid = fopen(qf->filename, "rb");
    if (!fid) {
        return NULL;
    }
    data = malloc(N + 1);
    if (start) {
        if (fseeko(fid, start, SEEK_SET)) {
            SYSERROR("Failed to seek to start of FITS header: byte %li in %s",
                     (long int)start, qf->filename);
            return NULL;
        }
    }
    nr = fread(data, 1, N, fid);
    fclose(fid);
    if (nr != N) {
        free(data);
        return NULL;
    }
    data[N] = '\0';
    if (Nbytes)
        *Nbytes = N;
    return data;
}

qfits_table* anqfits_get_table(const anqfits_t* qf, int ext) {
    const qfits_table* t = anqfits_get_table_const(qf, ext);
    if (!t)
        return NULL;
    return qfits_table_copy(t);
}

const qfits_table* anqfits_get_table_const(const anqfits_t* qf, int ext) {
    assert(ext >= 0 && ext < qf->Nexts);
    if (!qf->exts[ext].table) {
        const qfits_header* hdr = anqfits_get_header_const(qf, ext);
        off_t begin, size;
        if (!hdr) {
            qfits_error("Failed to get header for ext %i\n", ext);
            return NULL;
        }
        if (anqfits_get_data_start_and_size(qf, ext, &begin, &size)) {
            ERROR("failed to get data start and size");
            return NULL;
        }

        qf->exts[ext].table = qfits_table_open2(hdr, begin, size, qf->filename, ext);
    }
    return qf->exts[ext].table;
}

anqfits_image_t* anqfits_get_image(const anqfits_t* qf, int ext) {
    const anqfits_image_t* t = anqfits_get_image_const(qf, ext);
    anqfits_image_t* tout;
    if (!t)
        return NULL;
    tout = anqfits_image_new();
    memcpy(tout, t, sizeof(anqfits_image_t));
    return tout;
}

const anqfits_image_t* anqfits_get_image_const(const anqfits_t* qf, int ext) {
    assert(ext >= 0 && ext < qf->Nexts);
    if (!qf->exts[ext].image) {
        anqfits_image_t* img;
        const qfits_header* hdr = anqfits_get_header_const(qf, ext);
        int naxis1, naxis2, naxis3;
        if (!hdr) {
            qfits_error("Failed to get header for ext %i\n", ext);
            return NULL;
        }
        img = anqfits_image_new();

        // from qfits_image.c : qfitsloader_init()
        img->bitpix = qfits_header_getint(hdr, "BITPIX", -1);
        img->naxis  = qfits_header_getint(hdr, "NAXIS",  -1);
        naxis1 = qfits_header_getint(hdr, "NAXIS1", -1);
        naxis2 = qfits_header_getint(hdr, "NAXIS2", -1);
        naxis3 = qfits_header_getint(hdr, "NAXIS3", -1);
        img->bzero  = qfits_header_getdouble(hdr, "BZERO", 0.0);
        img->bscale = qfits_header_getdouble(hdr, "BSCALE", 1.0);

        if (img->bitpix == -1) {
            qfits_error("Missing BITPIX in file %s ext %i", qf->filename, ext);
            goto bailout;
        }
        if (!((img->bitpix == 8) || (img->bitpix == 16) ||
              (img->bitpix == 32) ||
              (img->bitpix == -32) || (img->bitpix == -64))) {
            qfits_error("Invalid BITPIX %i in file %s ext %i",
                        img->bitpix, qf->filename, ext);
            goto bailout;
        }
        img->bpp = BYTESPERPIXEL(img->bitpix);

        if (img->naxis < 0) {
            qfits_error("No NAXIS in file %s ext %i", qf->filename, ext);
            goto bailout;
        }
        if (img->naxis==0) {
            qfits_error("NAXIS = 0 in file %s ext %i", qf->filename, ext);
            goto bailout;
        }
        if (img->naxis > 3) {
            qfits_error("NAXIS = %i > 3 unsupported in file %s ext %i",
                        img->naxis, qf->filename, ext);
            goto bailout;
        }
        /* NAXIS1 must always be present */
        if (naxis1 < 0) {
            qfits_error("No NAXIS1 in file %s ext %i", qf->filename, ext);
            goto bailout;
        }
        img->width = 1;
        img->height = 1;
        img->planes = 1;
        switch (img->naxis) {
        case 1:
            img->width = naxis1;
            break;
        case 3:
            if (naxis3 < 0) {
                qfits_error("No NAXIS3 in file %s ext %i", qf->filename, ext);
                goto bailout;
            }
            img->planes = naxis3;
            // no break: fall through to...
        case 2:
            if (naxis2 < 0) {
                qfits_error("No NAXIS2 in file %s ext %i", qf->filename, ext);
                goto bailout;
            }
            img->height = naxis2;
            img->width = naxis1;
            break;
        }
        qf->exts[ext].image = img;
        return img;

    bailout:
        anqfits_image_free(img);
        return NULL;
    }
    return qf->exts[ext].image;
}





/*
static int starts_with(const char* str, const char* start) {
    int len = strlen(start);
    return strncmp(str, start, len) == 0;
}
 */

static const char* blankline = "                                                                                ";

static int parse_header_block(const char* buf, qfits_header* hdr, int* found_it) {
    char getval_buf[FITS_LINESZ+1];
    char getkey_buf[FITS_LINESZ+1];
    char getcom_buf[FITS_LINESZ+1];
    char line_buf[FITS_LINESZ+1];
    // Browse through current block
    int i;
    const char* line = buf;
    for (i=0; i<FITS_NCARDS; i++) {
        char *key, *val, *comment;
        debug("Looking at line %i:\n  %.80s\n", i, line);
        // Skip blank lines.
        if (!strcmp(line, blankline))
            continue;
        key = qfits_getkey_r(line, getkey_buf);
        if (!key) {
            fprintf(stderr, "Skipping un-parseable header line: \"%.80s\"\n", line);
            continue;
        }
        val = qfits_getvalue_r(line, getval_buf);
        comment = qfits_getcomment_r(line, getcom_buf);
        debug("Got key/value/comment \"%s\" / \"%s\" / \"%s\"\n", key, val, comment);
        memcpy(line_buf, line, FITS_LINESZ);
        line_buf[FITS_LINESZ] = '\0';
        qfits_header_append(hdr, key, val, comment, line_buf);
        line += 80;
        if (!strcmp(key, "END")) {
            debug("Found END!\n");
            *found_it = 1;
            break;
        }
    }
    return 0;
}

static size_t get_data_bytes(const qfits_header* hdr) {
    int naxis;
    size_t data_bytes;
    size_t npix;
    int i;
    data_bytes = abs(qfits_header_getint(hdr, "BITPIX", 0) / 8);
    naxis = qfits_header_getint(hdr, "NAXIS", 0);
    data_bytes *= qfits_header_getint(hdr, "GCOUNT", 1);
    npix = 1;
    if (!naxis)
        npix = 0;
    for (i=0; i<naxis; i++) {
        char key[32];
        int nax;
        sprintf(key, "NAXIS%i", i+1);
        nax = qfits_header_getint(hdr, key, 0);
        if (i == 0 && nax == 0) {
            // random groups signature; skip naxis1
        } else {
            npix *= (size_t)nax;
        }
    }
    npix += qfits_header_getint(hdr, "PCOUNT", 0);
    data_bytes *= npix;
    return data_bytes;
}

anqfits_t* anqfits_open(const char* filename) {
    return anqfits_open_hdu(filename, -1);
}

anqfits_t* anqfits_open_hdu(const char* filename, int hdu) {
    anqfits_t* qf = NULL;
    // copied from qfits_cache.c: qfits_cache_add()
    FILE* fin = NULL;
    struct stat sta;
    size_t n_blocks;
    int found_it;
    int xtend;
    size_t data_bytes;
    int end_of_file;
    size_t skip_blocks;
    char buf[FITS_BLOCK_SIZE];
    int seeked;
    int firsttime;
    int i;

    // initial maximum number of extensions: we grow automatically
    int ext_capacity = 1024;

    qfits_header* hdr = NULL;

    /* Stat file to get its size */
    if (stat(filename, &sta)!=0) {
        qdebug(printf("anqfits: cannot stat file %s: %s\n",
                      filename, strerror(errno)););
        goto bailout;
    }

    /* Open input file */
    fin=fopen(filename, "r");
    if (!fin) {
        qdebug(printf("anqfits: cannot open file %s: %s\n",
                      filename, strerror(errno)););
        goto bailout;
    }

    /* Read first block in */
    if (fread(buf, 1, FITS_BLOCK_SIZE, fin) != FITS_BLOCK_SIZE) {
        qdebug(printf("anqfits: error reading first block from %s: %s\n",
                      filename, strerror(errno)););
        goto bailout;
    }
    /* Identify FITS magic number */
    if (!starts_with(buf, "SIMPLE  =")) {
        qdebug(printf("anqfits: file %s is not FITS\n", filename););
        goto bailout;
    }

    /*
     * Browse through file to identify primary HDU size and see if there
     * might be some extensions. The size of the primary data zone will
     * also be estimated from the gathering of the NAXIS?? values and
     * BITPIX.
     */

    n_blocks = 0;
    found_it = 0;
    firsttime = 1;

    assert(strlen(blankline) == 80);

    // Parse this header
    hdr = qfits_header_new();
    while (!found_it) {
        debug("Firsttime = %i\n", firsttime);
        if (!firsttime) {
            // Read next FITS block
            debug("Reading next FITS block\n");
            if (fread(buf, 1, FITS_BLOCK_SIZE, fin) != FITS_BLOCK_SIZE) {
                qdebug(printf("anqfits: error reading file %s\n", filename););
                goto bailout;
            }
        }
        firsttime = 0;
        n_blocks++;
        if (parse_header_block(buf, hdr, &found_it))
            goto bailout;
    }
    // otherwise we bail out trying to read blocks past the EOF...
    assert(found_it);

    xtend = qfits_header_getboolean(hdr, "EXTEND", 0);
    data_bytes = get_data_bytes(hdr);

    debug("primary header: data_bytes %zu\n", data_bytes);

    qf = calloc(1, sizeof(anqfits_t));
    qf->filename = strdup(filename);
    qf->exts = calloc(ext_capacity, sizeof(anqfits_ext_t));
    assert(qf->exts);
    if (!qf->exts)
        goto bailout;

    // Set first HDU offsets
    qf->exts[0].hdr_start = 0;
    qf->exts[0].data_start = n_blocks;
    qf->exts[0].header = hdr;
    hdr = NULL;
    qf->Nexts = 1;

    debug("Extensions? %s\n", xtend ? "yes":"no");
    
    if (xtend) {
        /* Look for extensions */
        /*
         * Register all extension offsets
         */
        hdr = qfits_header_new();
        end_of_file = 0;
        while (!end_of_file) {

            if (qf->Nexts-1 == hdu) {
                debug("Stopped reading after finding HDU %i\n", hdu);
                //printf("Stopped reading after HDU %i\n", hdu);
                // Could cache the file offset to continue reading later...
                break;
            }
            /*
             * Skip the previous data section if pixels were declared
             */
            if (data_bytes > 0) {
                /* Skip as many blocks as there are declared pixels */
                size_t off;
                skip_blocks = qfits_blocks_needed(data_bytes);
                off = skip_blocks;
                off *= (size_t)FITS_BLOCK_SIZE;
                seeked = fseeko(fin, off, SEEK_CUR);
                if (seeked == -1) {
                    qfits_error("anqfits: failed to fseeko in file %s: %s",
                                filename, strerror(errno));
                    goto bailout;
                }

                debug("hdu %i, data_bytes %zu, skip_blocks %zu, off %zu, n_blocks %zu\n",
                      qf->Nexts-1, data_bytes, skip_blocks, off, n_blocks);
                /* Increase counter of current seen blocks. */
                n_blocks += skip_blocks;
                data_bytes = 0;
            }
            
            /* Look for extension start */
            found_it = 0;
            while (!found_it && !end_of_file) {
                if (fread(buf, 1, FITS_BLOCK_SIZE, fin) != FITS_BLOCK_SIZE) {
                    /* Reached end of file */
                    end_of_file = 1;
                    break;
                }
                n_blocks++;

                /* Search for XTENSION at block top */
                if (starts_with(buf, "XTENSION=")) {
                    debug("Found XTENSION\n");
                    /* Got an extension */
                    found_it = 1;
                    qf->exts[qf->Nexts].hdr_start = n_blocks-1;
                } else {
                    qfits_warning("Failed to find XTENSION in the FITS block following the previous data block -- whaddup?  Filename %s, block %zi, hdu %i",
                                  filename, n_blocks, qf->Nexts-1);
                }
                // FIXME -- should we really just skip the block if we don't find the "XTENSION=" header?
            }
            if (end_of_file)
                break;

            // Look for extension END
            n_blocks--;
            found_it = 0;
            firsttime = 1;

            if (!hdr)
                hdr = qfits_header_new();

            while (!found_it && !end_of_file) {
                if (!firsttime) {
                    if (fread(buf, 1, FITS_BLOCK_SIZE, fin) != FITS_BLOCK_SIZE) {
                        qdebug(printf("anqfits: XTENSION without END in %s\n",
                                      filename););
                        end_of_file = 1;
                        break;
                    }
                }
                firsttime = 0;
                n_blocks++;

                if (parse_header_block(buf, hdr, &found_it)) {
                    debug("parse_header_block() failed: bailing\n");
                    goto bailout;
                }
                debug("parse_header_block() succeeded: found END? %s\n",
                      found_it ? "yes":"no");
            }
            if (found_it) {
                data_bytes = get_data_bytes(hdr);
                debug("This data block will have %zu bytes\n", data_bytes);

                qf->exts[qf->Nexts].data_start = n_blocks;
                qf->exts[qf->Nexts].header = hdr;
                hdr = NULL;
                qf->Nexts++;
                if (qf->Nexts >= ext_capacity) {
                    ext_capacity *= 2;
                    qf->exts = realloc(qf->exts,
                                       ext_capacity * sizeof(anqfits_ext_t));
                    assert(qf->exts);
                    if (!qf->exts)
                        goto bailout;
                }
            }
        }
    }
    debug("Found %i extensions\n", qf->Nexts);

    if (hdr)
        qfits_header_destroy(hdr);
    hdr = NULL;

    /* Close file */
    fclose(fin);
    fin = NULL;

    // realloc
    qf->exts = realloc(qf->exts, qf->Nexts * sizeof(anqfits_ext_t));
    assert(qf->exts);
    if (!qf->exts)
        goto bailout;

    for (i=0; i<qf->Nexts; i++) {
        qf->exts[i].hdr_size = qf->exts[i].data_start - qf->exts[i].hdr_start;
        if (i == qf->Nexts-1) {
            debug("st_size %zu, /block_size = %zu\n",
                  (size_t)sta.st_size,
                  (size_t)(sta.st_size / (size_t)FITS_BLOCK_SIZE));
            qf->exts[i].data_size = ((sta.st_size/FITS_BLOCK_SIZE) -
                                     qf->exts[i].data_start);
        } else
            qf->exts[i].data_size = (qf->exts[i+1].hdr_start -
                                     qf->exts[i].data_start);
        debug("  Ext %i: header size %i, data size %i; hdr=%p\n",
              i, qf->exts[i].hdr_size, qf->exts[i].data_size,
              qf->exts[i].header);
        debug("ext %i: hdr_start %i, hdr_size %i, data_start %i, data_size %i, blocks\n",
              i,
              qf->exts[i].hdr_start, qf->exts[i].hdr_size,
              qf->exts[i].data_start, qf->exts[i].data_size);
    }
    qf->filesize = sta.st_size / FITS_BLOCK_SIZE;

    return qf;

 bailout:
    if (hdr)
        qfits_header_destroy(hdr);
    if (fin)
        fclose(fin);
    if (qf) {
        free(qf->filename);
        free(qf->exts);
        free(qf);
    }
    return NULL;
}

void anqfits_close(anqfits_t* qf) {
    int i;
    if (!qf)
        return;
    for (i=0; i<qf->Nexts; i++) {
        if (qf->exts[i].header)
            qfits_header_destroy(qf->exts[i].header);
        if (qf->exts[i].table)
            qfits_table_close(qf->exts[i].table);
        if (qf->exts[i].image)
            anqfits_image_free(qf->exts[i].image);
    }
    free(qf->exts);
    free(qf->filename);
    free(qf);
}

anqfits_image_t* anqfits_image_new() {
    anqfits_image_t* img = calloc(1, sizeof(anqfits_image_t));
    assert(img);
    return img;
}
void anqfits_image_free(anqfits_image_t* img) {
    free(img);
}


tfits_type anqfits_ptype_to_ttype(int ptype) {
    switch (ptype) {
    case PTYPE_UINT8:
        return TFITS_BIN_TYPE_B;
    case PTYPE_INT16:
        return TFITS_BIN_TYPE_I;
    case PTYPE_INT:
        return TFITS_BIN_TYPE_J;
    case PTYPE_FLOAT:
        return TFITS_BIN_TYPE_E;
    case PTYPE_DOUBLE:
        return TFITS_BIN_TYPE_D;
    }
    qfits_error("Unknown ptype %i\n", ptype);
    assert(0);
    return -1;
}

void* anqfits_readpix(const anqfits_t* qf, int ext,
                      int x0, int x1, int y0, int y1,
                      /** The plane you want, from 0 to planes-1 */
                      int            plane,
                      /** Pixel type you want
                       (PTYPE_FLOAT, PTYPE_INT or PTYPE_DOUBLE) */
                      int            ptype,
                      void* output,
                      int* pW, int* pH) {
    const anqfits_image_t* img = anqfits_get_image_const(qf, ext);
    //off_t NX, NY;
    //off_t planesize;
    off_t start;
    off_t size;

    off_t mapstart;
    size_t mapsize;
    int mapoffset;
    char* map = NULL;
    FILE* f = NULL;
    char* datastart;
    char* rowstart;
    char* outrowstart;
    off_t outrowsize;

    int x, y;
    off_t inlinesize;
    char* inlinebuf = NULL;

    int inptype;

    char* alloc_output = NULL;
    int outbpp;

    tfits_type in_ttype, out_ttype;

    if (!img)
        return NULL;

    if (x0) {
        if ((x0 < 0) || (x1 && (x0 >= x1)) || (x0 >= img->width)) {
            qfits_error("Invalid x0=%i not in [0, x1=%i <= W=%zi) reading %s ext %i",
                        x0, x1, img->width, qf->filename, ext);
            return NULL;
        }
    }
    if (y0) {
        if ((y0 < 0) || (y1 && (y0 >= y1)) || (y0 >= img->height)) {
            qfits_error("Invalid y0=%i not in [0, y1=%i <= W=%zi) reading %s ext %i",
                        y0, y1, img->height, qf->filename, ext);
            return NULL;
        }
    }
    if (x1) {
        if ((x1 < 0) || (x1 <= x0) || (x1 > img->width)) {
            qfits_error("Invalid x1=%i not in [0, x0=%i, W=%zi) reading %s ext %i",
                        x1, x0, img->width, qf->filename, ext);
            return NULL;
        }
    } else {
        x1 = img->width;
    }
    if (y1) {
        if ((y1 < 0) || (y1 <= y0) || (y1 > img->height)) {
            qfits_error("Invalid y1=%i not in [0, y0=%i, H=%zi) reading %s ext %i",
                        y1, y0, img->height, qf->filename, ext);
            return NULL;
        }
    } else {
        y1 = img->height;
    }

    if ((plane < 0) || (plane >= img->planes)) {
        qfits_error("Plane %i not in [0, %zi) reading %s ext %i\n",
                    plane, img->planes, qf->filename, ext);
        return NULL;
    }

    //NX = x1 - x0;
    //NY = y1 - y0;
    //planesize = img->width * img->height * (off_t)img->bpp;

    f = fopen(qf->filename, "rb");
    if (!f) {
        qfits_error("Failed to fopen %s: %s\n", qf->filename, strerror(errno));
        return NULL;
    }

    start = ((off_t)qf->exts[ext].data_start * (off_t)FITS_BLOCK_SIZE
             + ((off_t)y0 * img->width + (off_t)x0) * (off_t)img->bpp);
    size = (((off_t)(y1 - y0 - 1) * img->width + (off_t)(x1 - x0)) *
            (off_t)img->bpp);

    get_mmap_size(start, size, &mapstart, &mapsize, &mapoffset);
	int mode, flags;
    mode = PROT_READ;
    flags = MAP_SHARED;
    map = mmap(0, mapsize, mode, flags, fileno(f), mapstart);
    if (map == MAP_FAILED) {
        qfits_error("Failed to mmap file %s: %s",
                    qf->filename, strerror(errno));
        map = NULL;
        goto bailout;
    }
    fclose(f);
    f = NULL;

    datastart = map + mapoffset;

    /*
     if (fseeko(f, start, SEEK_SET)) {
     qfits_error("Failed to fseeko(%zu) in file %s ext %i: %s\n",
     start, qf->filename, ext, strerror(errno));
     goto bailout;
     }
     */

    inlinesize = (off_t)(x1 - x0) * (off_t)img->bpp;
    inlinebuf = malloc(inlinesize);

    switch (img->bitpix) {
    case 8:
        inptype = PTYPE_UINT8;
        break;
    case 16:
        inptype = PTYPE_INT16;
        break;
    case 32:
        inptype = PTYPE_INT;
        break;
    case -32:
        inptype = PTYPE_FLOAT;
        break;
    case -64:
        inptype = PTYPE_DOUBLE;
        break;
    default:
        qfits_error("Unknown bitpix %i\n", img->bitpix);
        goto bailout;
    }

    in_ttype = anqfits_ptype_to_ttype(inptype);
    out_ttype = anqfits_ptype_to_ttype(ptype);

    outbpp = qfits_pixel_ctype_size(ptype);
    if (!output) {
        output = alloc_output = malloc((off_t)(x1-x0) * (off_t)(y1-y0) *
                                       (off_t)outbpp);
    }

    rowstart = datastart;
    outrowstart = output;
    outrowsize = (off_t)outbpp * (off_t)(x1-x0);

    // Rows...
    for (y=y0; y<y1; y++) {
        memcpy(inlinebuf, rowstart, (off_t)img->bpp * (off_t)(x1-x0));
        rowstart += (off_t)img->bpp * img->width;
#ifndef WORDS_BIGENDIAN
        char* ptr = inlinebuf;
        for (x=x0; x<x1; x++) {
            qfits_swap_bytes(ptr, img->bpp);
            ptr += img->bpp;
        }
#endif
        // passthrough?
        if ((img->bzero == 0.0) && (img->bscale == 1.0) &&
            (inptype == ptype)) {
            memcpy(outrowstart, inlinebuf, outrowsize);
            outrowstart += outrowsize;
            continue;
        }

        // we treat these as "arrays" since both the input and output
        // are contiguous; hence no stride needed, and N=1.
        if (fits_convert_data_2(outrowstart, 0, out_ttype,
                                inlinebuf, 0, in_ttype,
                                x1-x0, 1, img->bzero, img->bscale)) {
            qfits_error("Failed to fits_convert_data_2\n");
            goto bailout;
        }

        outrowstart += outrowsize;
    }

    munmap(map, mapsize);
    free(inlinebuf);

    if (pW)
        *pW = (x1 - x0);
    if (pH)
        *pH = (y1 - y0);

    return output;

 bailout:
    free(inlinebuf);
    free(alloc_output);
    fclose(f);
    if (map) {
        munmap(map, mapsize);
    }
    return NULL;
}









