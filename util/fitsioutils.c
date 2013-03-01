/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2009 Dustin Lang.

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
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <sys/param.h>

#include "qfits.h"
#include "qfits_convert.h"

#include "fitsioutils.h"
#include "ioutils.h"
#include "keywords.h"
#include "an-endian.h"
#include "errors.h"
#include "qfits_error.h"
#include "qfits_std.h"
#include "log.h"
#include "errors.h"

Malloc
char* fits_to_string(const qfits_header* hdr, int* size) {
	int N = qfits_header_n(hdr);
	char* str;
	int i;

	str = malloc(N * FITS_LINESZ);
	if (!str) {
		SYSERROR("Failed to allocate string for %i FITS lines\n", N);
		return NULL;
	}
	for (i=0; i<N; i++) {
		if (qfits_header_write_line(hdr, i, str + i*FITS_LINESZ)) {
			ERROR("Failed to write FITS header line %i", i);
			free(str);
			return NULL;
		}
	}
	*size = N * FITS_LINESZ;
	return str;
}

int fits_write_header(const qfits_header* hdr, const char* fn) {
	FILE* fid;
	fid = fopen(fn, "wb");
	if (!fid) {
		SYSERROR("Failed to open file \"%s\" to write FITS header", fn);
		return -1;
	}
	if (qfits_header_dump(hdr, fid)) {
		ERROR("Failed to write FITS header to file \"%s\"", fn);
		return -1;
	}
	if (fits_pad_file(fid)) {
		ERROR("Failed to pad FITS header to file \"%s\"", fn);
		return -1;
	}
	if (fclose(fid)) {
		SYSERROR("Failed to close file \"%s\" after writing FITS header", fn);
		return -1;
	}
	return 0;
}


qfits_table* fits_copy_table(qfits_table* tbl) {
	qfits_table* out;
	out = calloc(1, sizeof(qfits_table));
	memcpy(out, tbl, sizeof(qfits_table));
	out->col = malloc(tbl->nc * sizeof(qfits_col));
	memcpy(out->col, tbl->col, tbl->nc * sizeof(qfits_col));
	return out;
}

int fits_pixdump(const qfitsdumper * qd) {
    FILE* f_out;
	const void* vbuf;
	anbool tostdout;
	int i;
	int isize;
	int osize;

	if (!qd) return -1;
	if (!qd->filename) return -1;
	if (qd->npix < 0) {
		ERROR("Negative number of pixels specified.");
		return -1;
	}

	// accept
	vbuf = qd->vbuf;
	switch (qd->ptype) {
	case PTYPE_FLOAT:
		if (!vbuf) vbuf = qd->fbuf;
		break;
	case PTYPE_INT:
		if (!vbuf) vbuf = qd->ibuf;
		break;
	case PTYPE_DOUBLE:
		if (!vbuf) vbuf = qd->dbuf;
		break;
	case PTYPE_UINT8:
	case PTYPE_INT16:
		// ok
		break;
	default:
		ERROR("Invalid input pixel type %i", qd->ptype);
		return -1;
	}

	if (!vbuf) {
		ERROR("No pixel buffer supplied");
		return -1;
	}

    tostdout = streq(qd->filename, "STDOUT");
	if (tostdout)
        f_out = stdout;
	else
        f_out = fopen(qd->filename, "a");

    if (!f_out) {
		SYSERROR("Failed to open output file \"%s\" for writing", qd->filename);
		return -1;
	}

	isize = qfits_pixel_ctype_size(qd->ptype);
	osize = qfits_pixel_fitstype_size(qd->out_ptype);

	for (i=0; i<qd->npix; i++) {
		char buf[8];
		if (qfits_pixel_ctofits(qd->ptype, qd->out_ptype, vbuf, buf)) {
			ERROR("Failed to convert pixel value to FITS");
			return -1;
		}
		if (fwrite(buf, osize, 1, f_out) != 1) {
			SYSERROR("Failed to write FITS pixel value to file \"%s\"", qd->filename);
			return -1;
		}
		vbuf += isize;
	}

	if (!tostdout)
		if (fclose(f_out)) {
			SYSERROR("Failed to close FITS outptu file \"%s\"", qd->filename);
			return -1;
		}
    return 0;
}





int fits_write_float_image(const float* img, int nx, int ny, const char* fn) {
	int rtn;
    qfitsdumper qoutimg;
    memset(&qoutimg, 0, sizeof(qoutimg));
    qoutimg.filename = fn;
    qoutimg.npix = nx * ny;
    qoutimg.ptype = PTYPE_FLOAT;
    qoutimg.fbuf = img;
    qoutimg.out_ptype = BPP_IEEE_FLOAT;
	rtn = fits_write_header_and_image(NULL, &qoutimg, nx);
	if (rtn)
		ERROR("Failed to write FITS image to file \"%s\"", fn);
	return rtn;
}

int fits_write_u8_image(const uint8_t* img, int nx, int ny, const char* fn) {
	int rtn;
    qfitsdumper qoutimg;
    memset(&qoutimg, 0, sizeof(qoutimg));
    qoutimg.filename = fn;
    qoutimg.npix = nx * ny;
    qoutimg.ptype = PTYPE_UINT8;
    qoutimg.vbuf = img;
    qoutimg.out_ptype = BPP_8_UNSIGNED;
	rtn = fits_write_header_and_image(NULL, &qoutimg, nx);
	if (rtn)
		ERROR("Failed to write FITS image to file \"%s\"", fn);
	return rtn;
}

int fits_write_i16_image(const int16_t* img, int nx, int ny, const char* fn) {
	int rtn;
    qfitsdumper qoutimg;
    memset(&qoutimg, 0, sizeof(qoutimg));
    qoutimg.filename = fn;
    qoutimg.npix = nx * ny;
    qoutimg.ptype = PTYPE_INT16;
    qoutimg.vbuf = img;
    qoutimg.out_ptype = BPP_16_SIGNED;
	rtn = fits_write_header_and_image(NULL, &qoutimg, nx);
	if (rtn)
		ERROR("Failed to write FITS image to file \"%s\"", fn);
	return rtn;
}

static void errfunc(char* errstr) {
    report_error("qfits", -1, __func__, "%s", errstr);
}

int fits_write_header_and_image(const qfits_header* hdr, const qfitsdumper* qd, int W) {
	FILE* fid;
	const char* fn = qd->filename;
	qfits_header* freehdr = NULL;

    fid = fopen(fn, "w");
    if (!fid) {
        SYSERROR("Failed to open file \"%s\" for output", fn);
        return -1;
    }
	if (!hdr) {
		freehdr = fits_get_header_for_image(qd, W, NULL);
		hdr = freehdr;
	}
    if (qfits_header_dump(hdr, fid)) {
        ERROR("Failed to write image header to file \"%s\"", fn);
        return -1;
    }
	if (freehdr)
		qfits_header_destroy(freehdr);
    // the qfits pixel dumper appends to the given filename, so close
    // the file here.
    if (fits_pad_file(fid) ||
        fclose(fid)) {
        SYSERROR("Failed to pad or close file \"%s\"", fn);
        return -1;
    }
    // write data.
    if (fits_pixdump(qd)) {
        ERROR("Failed to write image data to file \"%s\"", fn);
        return -1;
    }
    // FITS pad
    fid = fopen(fn, "a");
    if (!fid) {
        SYSERROR("Failed to open file \"%s\" for padding", fn);
        return -1;
    }
    if (fits_pad_file(fid) ||
        fclose(fid)) {
        SYSERROR("Failed to pad or close file \"%s\"", fn);
        return -1;
    }
	return 0;
}

qfits_header* fits_get_header_for_image2(int W, int H, int bitpix,
										 qfits_header* addtoheader) {
	return fits_get_header_for_image3(W, H, bitpix, 1, addtoheader);
}

qfits_header* fits_get_header_for_image3(int W, int H, int bitpix, int planes,
										 qfits_header* addtoheader) {
    qfits_header* hdr;
    if (addtoheader)
        hdr = addtoheader;
    else
        hdr = qfits_header_default();
    fits_header_add_int(hdr, "BITPIX", bitpix, "bits per pixel");
    fits_header_add_int(hdr, "NAXIS", (planes == 1) ? 2 : 3, "number of axes");
    fits_header_add_int(hdr, "NAXIS1", W, "image width");
    fits_header_add_int(hdr, "NAXIS2", H, "image height");
	if (planes > 1)
		fits_header_add_int(hdr, "NAXIS3", planes, "image planes");
    return hdr;
}

qfits_header* fits_get_header_for_image(const qfitsdumper* qd, int W,
                                        qfits_header* addtoheader) {
	return fits_get_header_for_image2(W, qd->npix / W, qd->out_ptype, addtoheader);
}

void fits_use_error_system() {
    qfits_err_remove_all();
    qfits_err_register(errfunc);
    qfits_err_statset(1);
}

int fits_convert_data(void* vdest, int deststride, tfits_type desttype,
                      const void* vsrc, int srcstride, tfits_type srctype,
                      int arraysize, size_t N) {
	size_t i;
    int j;
    char* dest = vdest;
    const char* src = vsrc;
    int destatomsize = fits_get_atom_size(desttype);
    int srcatomsize = fits_get_atom_size(srctype);

    // this loop is over rows of data
    for (i=0; i<N; i++) {
        // store local pointers so we can stride over the array, without
        // affecting the row stride.
        char* adest = dest;
        const char* asrc = src;
        int64_t ival = 0;
        double  dval = 0;
        anbool src_is_int = TRUE;

        // this loop is over elements of the array, if the column contains an array.
        // (ie, for scalar columns, arraysize is 1.)
        for (j=0; j<arraysize; j++) {
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

double fits_get_double_val(const qfits_table* table, int column,
                           const void* rowdata) {
    const unsigned char* cdata = rowdata;
    double dval;
    float fval;

    // oh, the insanity of qfits...
    cdata += (table->col[column].off_beg - table->col[0].off_beg);
    if (table->col[column].atom_type == TFITS_BIN_TYPE_E) {
        memcpy(&fval, cdata, sizeof(fval));
        v32_ntoh(&fval);
        dval = fval;
        return fval;
    } else if (table->col[column].atom_type == TFITS_BIN_TYPE_D) {
        memcpy(&dval, cdata, sizeof(dval));
        v64_ntoh(&dval);
        return dval;
    } else {
        fprintf(stderr, "Invalid column type %i.\n", table->col[column].atom_type);
    }
    return HUGE_VAL;
}

int fits_is_table_header(const char* key) {
    return (!strcasecmp(key, "XTENSION") ||
            !strcasecmp(key, "BITPIX") ||
            !strncasecmp(key, "NAXIS...", 5) ||
            !strcasecmp(key, "PCOUNT") ||
            !strcasecmp(key, "GCOUNT") ||
            !strcasecmp(key, "TFIELDS") ||
            !strncasecmp(key, "TFORM...", 5) ||
            !strncasecmp(key, "TTYPE...", 5) ||
            !strncasecmp(key, "TUNIT...", 5) ||
            !strncasecmp(key, "TNULL...", 5) ||
            !strncasecmp(key, "TSCAL...", 5) ||
            !strncasecmp(key, "TZERO...", 5) ||
            !strncasecmp(key, "TDISP...", 5) ||
            !strncasecmp(key, "THEAP...", 5) ||
            !strncasecmp(key, "TDIM...", 4) ||
            !strcasecmp(key, "END")) ? 1 : 0;
}

int fits_is_primary_header(const char* key) {
    return (!strcasecmp(key, "SIMPLE") ||
            !strcasecmp(key, "BITPIX") ||
            !strncasecmp(key, "NAXIS...", 5) ||
            !strcasecmp(key, "EXTEND") ||
            !strcasecmp(key, "END")) ? 1 : 0;
}

void fits_copy_non_table_headers(qfits_header* dest, const qfits_header* src) {
    char key[FITS_LINESZ+1];
    char val[FITS_LINESZ+1];
    char com[FITS_LINESZ+1];
    char lin[FITS_LINESZ+1];
    int i;
    for (i=0;; i++) {
        if (qfits_header_getitem(src, i, key, val, com, lin) == -1)
            break;
        if (fits_is_table_header(key))
            continue;
        qfits_header_add(dest, key, val, com, lin);
    }
}

char* fits_get_dupstring(const qfits_header* hdr, const char* key) {
	// qfits_pretty_string() never increases the length of the string
	char pretty[FITS_LINESZ+1];
	char* val;
	val = qfits_header_getstr(hdr, key);
	if (!val)
		return NULL;
	qfits_pretty_string_r(val, pretty);
	return strdup_safe(pretty);
}

void fits_header_addf(qfits_header* hdr, const char* key, const char* comment,
                      const char* format, ...) {
    char buf[FITS_LINESZ + 1];
    va_list lst;
    va_start(lst, format);
    vsnprintf(buf, sizeof(buf), format, lst);
    qfits_header_add(hdr, key, buf, comment, NULL);
    va_end(lst);
}

// the column where the value of a header card begins, 0-indexed.
// KEYWORD = 'VALUE'
// 01234567890
#define FITS_VALUE_START 10

void fits_header_addf_longstring(qfits_header* hdr, const char* key,
                                 const char* comment, const char* format, ...) {
    char* str;
    int nb;
    int linelen;
    va_list lst;
    int i;
    int commentlen;
    
    va_start(lst, format);
    nb = vasprintf(&str, format, lst);
    va_end(lst);
    if (nb == -1) {
        SYSERROR("vasprintf failed.");
        return;
    }
    // +2 for the quotes
    linelen = nb + FITS_VALUE_START + 2;
    // +1 for each character ' which must be escaped
    for (i=0; i<nb; i++)
        if (str[i] == '\'')
            linelen++;

    // +3 for the " / "
    commentlen = (comment ? 3 + strlen(comment) : 0);
    linelen += commentlen;

    if (linelen < FITS_LINESZ)
        qfits_header_add(hdr, key, str, comment, NULL);
    else {
        // Long string - use CONTINUE.
        int len = nb;
        char line[FITS_LINESZ + 1];
        char* linebuf;
        char* buf;
        anbool addquotes = FALSE;
        anbool escapequotes = FALSE;
        buf = str;
        while (len > 0) {
            anbool amp = TRUE;
            int maxlen;

            //printf("String: \"%s\"\n", buf);
            //printf("Linelen: %i\n", len);

            maxlen = FITS_LINESZ - (commentlen + FITS_VALUE_START + 2);
            for (i=0; i<MIN(maxlen, len); i++)
                if (buf[i] == '\'')
                    maxlen--;
            if (len <= maxlen) {
                amp = FALSE;
                maxlen = len;
            } else
                // +1 for the &
                maxlen--;
            /* must escape single quotes also...
             snprintf(line, sizeof(line)-1, "%s%.*s%s%s",
             addquotes ? "  '" : "",
             maxlen, buf, amp ? "&" : "",
             addquotes ? "'" : "");
             */
            linebuf = line;
            if (addquotes) {
                *linebuf = ' ';
                linebuf++;
                *linebuf = ' ';
                linebuf++;
                *linebuf = '\'';
                linebuf++;
            }
            for (i=0; i<maxlen; i++) {
                if (escapequotes && buf[i] == '\'') {
                    *linebuf = '\'';
                    linebuf++;
                }
                *linebuf = buf[i];
                linebuf++;
            }
            if (amp) {
                *linebuf = '&';
                linebuf++;
            }
            if (addquotes) {
                *linebuf = '\'';
                linebuf++;
            }
            *linebuf = '\0';
            
            qfits_header_add(hdr, key, line, comment, NULL);
            comment = "";
            commentlen = 0;
            key = "CONTINUE";
            addquotes = TRUE;
            escapequotes = TRUE;
            buf += maxlen;
            len -= maxlen;
        }
    }
    free(str);
}

// modifies s in-place.
// removes leading spaces.
static void trim_leading_spaces(char* s) {
    char* out = s;
    int i, N;
    N = strlen(s);
    for (i=0; i<N; i++)
        if (s[i] != ' ')
            break;
    N = MAX(0, N - i);
    memmove(out, s + i, N);
    out[N] = '\0';
}

// modifies s in-place.
// removes trailing spaces.
static void trim_trailing_spaces(char* s) {
    int N;
    N = strlen(s) - 1;
    while (N >= 0 && s[N] == ' ') {
        s[N] = '\0';
        N--;
    }
}

// modifies s in-place.
static anbool pretty_continue_string(char* s) {
    char* out = s;
    int i, iout, N;
    N = strlen(s);
    i = 0;
    iout = 0;
    for (; i<N; i++) {
        if (s[i] == '\'')
            i++;
        out[iout] = s[i];
        iout++;
    }
    out[iout] = '\0';
    trim_trailing_spaces(out);
    return TRUE;
}

// modifies s in-place.
// removes leading and trailing spaces.
static anbool trim_valid_string(char* s) {
    int i, N, end;
    trim_leading_spaces(s);
    if (s[0] != '\'')
        return FALSE;
    N = strlen(s);
    end = -1;
    for (i=1; i<N; i++) {
        if (s[i] != '\'')
            continue;
        // if it's followed by another ',  it's ok.
        i++;
        if (i<N && s[i] == '\'')
            continue;
        // we found the end of the string.
        end = i;
        // it can be followed by spaces, / and a comment, but nothing else.
        while (i < N && s[i] == ' ')
            i++;
        if (i == N)
            break;
        if (s[i] == '/')
            break;
        return FALSE;
    }
    if (end == -1)
        return FALSE;
    N = end - 2;
    memmove(s, s+1, N);
    s[N] = '\0';
    return TRUE;
}

char* fits_get_long_string(const qfits_header* hdr, const char* thekey) {
    int i, N;

    N = qfits_header_n(hdr);
    for (i=0; i<N; i++) {
        int j;
        char str[FITS_LINESZ+1];
        int len;
        sl* slist;
		char* cptr;
        char key[FITS_LINESZ+1];
        char val[FITS_LINESZ+1];
        qfits_header_getitem(hdr, i, key, val, NULL, NULL);
        /*
         printf("Looking for initial match:\n");
         printf("  key \"%s\"\n", key);
         printf("  val \"%s\"\n", val);
         */
        if (strcmp(key, thekey))
            continue;
        qfits_pretty_string_r(val, str);
        len = strlen(str);
        if (len < 1 || str[len-1] != '&')
            return strdup(str);
        slist = sl_new(4);
        sl_append(slist, str);
        for (j=i+1; j<N; j++) {
            qfits_header_getitem(hdr, j, key, val, NULL, NULL);
            /*
             printf("Looking for CONTINUE cards:\n");
             printf("  key \"%s\"\n", key);
             printf("  val \"%s\"\n", val);
             */
            if (strcmp(key, "CONTINUE"))
                break;
            // must begin with two spaces.
            if (strncmp(val, "  ", 2))
                break;
            //printf("Raw val = \"%s\"\n", val);
            if (!trim_valid_string(val))
                break;
            //printf("Trimmed val = \"%s\"\n", val);
            if (!pretty_continue_string(val))
                break;
            //printf("Pretty val = \"%s\"\n", val);
            sl_append(slist, val);
            len = strlen(val);
            if (len < 1 || val[len-1] != '&')
                break;
        }
        // On all but the last string, strip the trailing "&".
        for (j=0; j<sl_size(slist)-1; j++) {
            cptr = sl_get(slist, j);
            cptr[strlen(cptr)-1] = '\0';
        }
        cptr = sl_join(slist, "");
        sl_free2(slist);
        return cptr;
    }
    // keyword not found.
    return NULL;
}

void fits_header_add_longstring_boilerplate(qfits_header* hdr) {
    qfits_header_add(hdr, "LONGSTRN", "OGIP 1.0", "The OGIP long string convention may be used", NULL);
    qfits_header_add(hdr, "COMMENT", "This FITS file may contain long string keyword values that are",   NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "continued over multiple keywords.  This convention uses the  '&'", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "character at the end of the string which is then continued",       NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "on subsequent keywords whose name = 'CONTINUE'.",                  NULL, NULL);
}

void fits_header_modf(qfits_header* hdr, const char* key, const char* comment,
                      const char* format, ...) {
    char buf[FITS_LINESZ + 1];
    va_list lst;
    va_start(lst, format);
    vsnprintf(buf, sizeof(buf), format, lst);
    qfits_header_mod(hdr, key, buf, comment);
    va_end(lst);
}

void fits_header_set_double(qfits_header* hdr, const char* key, double val,
                            const char* comment) {
	if (qfits_header_getstr(hdr, key))
		fits_header_mod_double(hdr, key, val, comment);
	else
		fits_header_add_double(hdr, key, val, comment);
}

void fits_header_set_int(qfits_header* hdr, const char* key, int val,
                            const char* comment) {
	if (qfits_header_getstr(hdr, key))
		fits_header_mod_int(hdr, key, val, comment);
	else
		fits_header_add_int(hdr, key, val, comment);
}

void fits_header_add_double(qfits_header* hdr, const char* key, double val,
                            const char* comment) {
    fits_header_addf(hdr, key, comment, "%.12G", val);
}

void fits_header_mod_double(qfits_header* hdr, const char* key, double val,
                            const char* comment) {
    fits_header_modf(hdr, key, comment, "%.12G", val);
}

void fits_header_mod_int(qfits_header* hdr, const char* key, int val,
                         const char* comment) {
    fits_header_modf(hdr, key, comment, "%i", val);
}

void fits_header_add_int(qfits_header* hdr, const char* key, int val,
                         const char* comment) {
    fits_header_addf(hdr, key, comment, "%i", val);
}

int fits_update_value(qfits_header* hdr, const char* key, const char* newvalue) {
  char oldcomment[FITS_LINESZ + 1];
  char* c = qfits_header_getcom(hdr, key);
  if (!c) {
    return -1;
  }
  // hmm, not sure I need to copy this...
  strncpy(oldcomment, c, FITS_LINESZ);
  qfits_header_mod(hdr, key, newvalue, oldcomment);
  return 0;
}

static int add_long_line(qfits_header* hdr, const char* keyword, const char* indent, int append, const char* format, va_list lst) {
	const int charsperline = 60;
	char* origstr = NULL;
	char* str;
	int len;
	int indlen = (indent ? strlen(indent) : 0);
	len = vasprintf(&origstr, format, lst);
	if (len == -1) {
		fprintf(stderr, "vasprintf failed: %s\n", strerror(errno));
		return -1;
	}
	str = origstr;
	do {
		char copy[80];
		int doindent = (indent && (str != origstr));
		int nchars = charsperline - (doindent ? indlen : 0);
		int brk;
		if (nchars > len)
			nchars = len;
		else {
			// look for a space to break the line.
			for (brk=nchars-1; (brk>=0) && (str[brk] != ' '); brk--);
			if (brk > 0) {
				// found a place to break the line.
				nchars = brk + 1;
			}
		}
		sprintf(copy, "%s%.*s", (doindent ? indent : ""), nchars, str);
        if (append)
            qfits_header_append(hdr, keyword, copy, NULL, NULL);
        else
            qfits_header_add(hdr, keyword, copy, NULL, NULL);
		len -= nchars;
		str += nchars;
	} while (len > 0);
	free(origstr);
	return 0;
}

static int 
ATTRIB_FORMAT(printf,4,5)
add_long_line_b(qfits_header* hdr, const char* keyword,
                const char* indent, const char* format, ...) {
	va_list lst;
	int rtn;
	va_start(lst, format);
	rtn = add_long_line(hdr, keyword, indent, 0, format, lst);
	va_end(lst);
	return rtn;
}

int 
fits_add_long_comment(qfits_header* dst, const char* format, ...) {
	va_list lst;
	int rtn;
	va_start(lst, format);
	rtn = add_long_line(dst, "COMMENT", "  ", 0, format, lst);
	va_end(lst);
	return rtn;
}

int 
fits_append_long_comment(qfits_header* dst, const char* format, ...) {
	va_list lst;
	int rtn;
	va_start(lst, format);
	rtn = add_long_line(dst, "COMMENT", "  ", 1, format, lst);
	va_end(lst);
	return rtn;
}

int 
fits_add_long_history(qfits_header* dst, const char* format, ...) {
	va_list lst;
	int rtn;
	va_start(lst, format);
	rtn = add_long_line(dst, "HISTORY", "  ", 0, format, lst);
	va_end(lst);
	return rtn;
}

int fits_add_args(qfits_header* hdr, char** args, int argc) {
    sl* s;
    int i;
    char* ss;

    s = sl_new(4);
	for (i=0; i<argc; i++) {
		const char* str = args[i];
        sl_append_nocopy(s, str);
	}
    ss = sl_join(s, " ");
    sl_free_nonrecursive(s);
    i = add_long_line_b(hdr, "HISTORY", "  ", "%s", ss);
    free(ss);
	return i;
}

int fits_copy_header(const qfits_header* src, qfits_header* dest, char* key) {
	char* str = qfits_header_getstr(src, key);
	if (!str) {
		// header not found, or other problem.
		return -1;
	}
	qfits_header_add(dest, key, str,
						qfits_header_getcom(src, key), NULL);
	return 0;
}

static int copy_all_headers(const qfits_header* src, qfits_header* dest, char* targetkey,
                            anbool append) {
	int i, N;
	char key[FITS_LINESZ+1];
	char val[FITS_LINESZ+1];
	char com[FITS_LINESZ+1];
	char lin[FITS_LINESZ+1];
    N = qfits_header_n(src);

	for (i=0; i<N; i++) {
		if (qfits_header_getitem(src, i, key, val, com, lin) == -1)
            break;
		if (targetkey && strcasecmp(key, targetkey))
			continue;
        if (append)
            qfits_header_append(dest, key, val, com, lin);
        else
            qfits_header_add(dest, key, val, com, lin);
	}
	return 0;
}

int fits_copy_all_headers(const qfits_header* src, qfits_header* dest, char* targetkey) {
    return copy_all_headers(src, dest, targetkey, FALSE);
}

int fits_append_all_headers(const qfits_header* src, qfits_header* dest, char* targetkey) {
    return copy_all_headers(src, dest, targetkey, TRUE);
}

int fits_pad_file_with(FILE* fid, char pad) {
	off_t offset;
	int npad;
	
	// pad with zeros up to a multiple of 2880 bytes.
	offset = ftello(fid);
	npad = (offset % FITS_BLOCK_SIZE);
	if (npad) {
		int i;
		npad = FITS_BLOCK_SIZE - npad;
		for (i=0; i<npad; i++)
			if (fwrite(&pad, 1, 1, fid) != 1) {
                SYSERROR("Failed to pad FITS file");
				return -1;
			}
	}
	return 0;
}

int fits_pad_file(FILE* fid) {
    return fits_pad_file_with(fid, 0);
}

int fits_pad_file_name(char* filename) {
	int rtn;
	FILE* fid = fopen(filename, "ab");
	rtn = fits_pad_file(fid);
	if (!rtn && fclose(fid)) {
		SYSERROR("Failed to close file after padding it.");
		return -1;
	}
	return rtn;
}

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

int fits_add_column(qfits_table* table, int column, tfits_type type,
					int ncopies, const char* units, const char* label) {
	int atomsize;
	int colsize;

	atomsize = fits_get_atom_size(type);
	if (atomsize == -1) {
		fprintf(stderr, "Unknown atom size for type %i.\n", type);
		return -1;
	}
	if (type == TFITS_BIN_TYPE_X)
		// bit field: convert bits to bytes, rounding up.
		ncopies = (ncopies + 7) / 8;
	colsize = atomsize * ncopies;
	qfits_col_fill(table->col + column, ncopies, 0, atomsize, type, label, units,
				   "", "", 0, 0, 0, 0, table->tab_w);
	table->tab_w += colsize;
	return 0;
}

int fits_offset_of_column(qfits_table* table, int colnum) {
	int off = 0;
	int i;
	// from qfits_table.c : qfits_compute_table_width()
	for (i=0; i<colnum; i++) {
        if (table->tab_t == QFITS_ASCIITABLE) {
            off += table->col[i].atom_nb;
        } else if (table->tab_t == QFITS_BINTABLE) {
            off += table->col[i].atom_nb * table->col[i].atom_size;
        }
	}
	return off;
}

int fits_write_data_D(FILE* fid, double value, anbool flip) {
	assert(sizeof(double) == 8);
	if (flip)
		v64_hton(&value);
	if (fwrite(&value, 8, 1, fid) != 1) {
		fprintf(stderr, "Failed to write a double to FITS file: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

int fits_write_data_E(FILE* fid, float value, anbool flip) {
	assert(sizeof(float) == 4);
	if (flip)
		v32_hton(&value);
	if (fwrite(&value, 4, 1, fid) != 1) {
		fprintf(stderr, "Failed to write a float to FITS file: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

int fits_write_data_B(FILE* fid, uint8_t value) {
	if (fwrite(&value, 1, 1, fid) != 1) {
		fprintf(stderr, "Failed to write a bit array to FITS file: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

int fits_write_data_L(FILE* fid, char value) {
    return fits_write_data_A(fid, value);
}

int fits_write_data_A(FILE* fid, char value) {
	return fits_write_data_B(fid, value);
}

int fits_write_data_X(FILE* fid, unsigned char value) {
	return fits_write_data_B(fid, value);
}

int fits_write_data_I(FILE* fid, int16_t value, anbool flip) {
	if (flip)
		v16_hton(&value);
	if (fwrite(&value, 2, 1, fid) != 1) {
		fprintf(stderr, "Failed to write a short to FITS file: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

int fits_write_data_J(FILE* fid, int32_t value, anbool flip) {
	if (flip)
		v32_hton(&value);
	if (fwrite(&value, 4, 1, fid) != 1) {
		fprintf(stderr, "Failed to write an int to FITS file: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

int fits_write_data_K(FILE* fid, int64_t value, anbool flip) {
	if (flip)
		v64_hton(&value);
	if (fwrite(&value, 8, 1, fid) != 1) {
		fprintf(stderr, "Failed to write an int64 to FITS file: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

int fits_write_data_array(FILE* fid, const void* vvalue, tfits_type type,
                          int N, anbool flip) {
    int i;
    int rtn = 0;
    const char* pvalue = (const char*)vvalue;

    if (pvalue == NULL) {
        if (fseeko(fid, fits_get_atom_size(type) * N, SEEK_CUR)) {
            fprintf(stderr, "Failed to skip %i bytes in fits_write_data_array: %s\n",
                    fits_get_atom_size(type) * N, strerror(errno));
            return -1;
        }
        return 0;
    }

    for (i=0; i<N; i++) {
        switch (type) {
        case TFITS_BIN_TYPE_A:
            rtn = fits_write_data_A(fid, *(unsigned char*)pvalue);
            pvalue += sizeof(unsigned char);
            break;
        case TFITS_BIN_TYPE_B:
            rtn = fits_write_data_B(fid, *(unsigned char*)pvalue);
            pvalue += sizeof(unsigned char);
            break;
        case TFITS_BIN_TYPE_L:
            rtn = fits_write_data_L(fid, *(anbool*)pvalue);
            pvalue += sizeof(anbool);
            break;
        case TFITS_BIN_TYPE_D:
            rtn = fits_write_data_D(fid, *(double*)pvalue, flip);
            pvalue += sizeof(double);
            break;
        case TFITS_BIN_TYPE_E:
            rtn = fits_write_data_E(fid, *(float*)pvalue, flip);
            pvalue += sizeof(float);
            break;
        case TFITS_BIN_TYPE_I:
            rtn = fits_write_data_I(fid, *(int16_t*)pvalue, flip);
            pvalue += sizeof(int16_t);
            break;
        case TFITS_BIN_TYPE_J:
            rtn = fits_write_data_J(fid, *(int32_t*)pvalue, flip);
            pvalue += sizeof(int32_t);
            break;
        case TFITS_BIN_TYPE_K:
            rtn = fits_write_data_K(fid, *(int64_t*)pvalue, flip);
            pvalue += sizeof(int64_t);
            break;
        case TFITS_BIN_TYPE_X:
            rtn = fits_write_data_X(fid, *(unsigned char*)pvalue);
            pvalue += sizeof(unsigned char);
            break;
        default:
            fprintf(stderr, "fitsioutils: fits_write_data: unknown data type %i.\n", type);
            rtn = -1;
            break;
        }
        if (rtn)
            break;
    }
    return rtn;
}

int fits_write_data(FILE* fid, void* pvalue, tfits_type type, anbool flip) {
    return fits_write_data_array(fid, pvalue, type, 1, flip);
}

int fits_bytes_needed(int size) {
	size += FITS_BLOCK_SIZE - 1;
	return size - (size % FITS_BLOCK_SIZE);
}

int fits_blocks_needed(int size) {
	return (size + FITS_BLOCK_SIZE - 1) / FITS_BLOCK_SIZE;
}

static char fits_endian_string[16];
static int  fits_endian_string_inited = 0;

static void fits_init_endian_string() {
    if (!fits_endian_string_inited) {
        uint32_t endian = ENDIAN_DETECTOR;
        unsigned char* cptr = (unsigned char*)&endian;
        fits_endian_string_inited = 1;
        sprintf(fits_endian_string, "%02x:%02x:%02x:%02x", (uint)cptr[0], (uint)cptr[1], (uint)cptr[2], (uint)cptr[3]);
    }
}

void fits_fill_endian_string(char* str) {
    fits_init_endian_string();
    strcpy(str, fits_endian_string);
}

char* fits_get_endian_string() {
    fits_init_endian_string();
    return fits_endian_string;
}

void fits_add_endian(qfits_header* header) {
	qfits_header_add(header, "ENDIAN", fits_get_endian_string(), "Endianness detector: u32 0x01020304 written ", NULL);
	qfits_header_add(header, "", NULL, " in the order it is stored in memory.", NULL);
	// (don't make this a COMMENT because that makes it get separated from the ENDIAN header line.)
}

void fits_add_reverse_endian(qfits_header* header) {
    uint32_t endian = ENDIAN_DETECTOR;
    unsigned char* cptr = (unsigned char*)&endian;
	fits_header_addf(header, "ENDIAN", "Endianness detector: u32 0x01020304 written ",
                     "%02x:%02x:%02x:%02x", (int)cptr[3], (int)cptr[2], (int)cptr[1], (int)cptr[0]);
	qfits_header_add(header, "", NULL, " in the order it is stored in memory.", NULL);
    qfits_header_add(header, "", NULL, "Note, this was written by a machine of the reverse endianness.", NULL);
}

void fits_mod_reverse_endian(qfits_header* header) {
    uint32_t endian = ENDIAN_DETECTOR;
    unsigned char* cptr = (unsigned char*)&endian;
	fits_header_modf(header, "ENDIAN", "Endianness detector: u32 0x01020304 written ",
                     "%02x:%02x:%02x:%02x", (int)cptr[3], (int)cptr[2], (int)cptr[1], (int)cptr[0]);
}

qfits_table* fits_get_table_column(const char* fn, const char* colname, int* pcol) {
    int i, nextens, start, size;

	nextens = qfits_query_n_ext(fn);
	for (i=0; i<=nextens; i++) {
        qfits_table* table;
        int c;
		if (qfits_get_datinfo(fn, i, &start, &size) == -1) {
			fprintf(stderr, "error getting start/size for ext %i.\n", i);
            return NULL;
        }
		if (!qfits_is_table(fn, i))
            continue;
        table = qfits_table_open(fn, i);
		if (!table) {
			fprintf(stderr, "Couldn't read FITS table from file %s, extension %i.\n",
					fn, i);
			continue;
		}
		c = fits_find_column(table, colname);
		if (c != -1) {
			*pcol = c;
			return table;
		}
		qfits_table_close(table);
    }
	return NULL;
}

int fits_find_table_column(const char* fn, const char* colname, int* pstart, int* psize, int* pext) {
    int i, nextens;
	nextens = qfits_query_n_ext(fn);
	for (i=1; i<=nextens; i++) {
        qfits_table* table;
        int c;
        table = qfits_table_open(fn, i);
		if (!table) {
			ERROR("Couldn't read FITS table from file %s, extension %i.\n", fn, i);
			continue;
		}
		c = fits_find_column(table, colname);
		qfits_table_close(table);
		if (c == -1) {
			continue;
		}
		if (qfits_get_datinfo(fn, i, pstart, psize) == -1) {
			ERROR("error getting start/size for ext %i in file %s.\n", i, fn);
            return -1;
        }
		if (pext) *pext = i;
		return 0;
    }
	debug("searched %i extensions in file %s but didn't find a table with a column \"%s\".\n",
		nextens, fn, colname);
    return -1;
}

int fits_find_column(const qfits_table* table, const char* colname) {
	int c;
	for (c=0; c<table->nc; c++) {
		const qfits_col* col = table->col + c;
		//debug("column: \"%s\"\n", col->tlabel);
		if (strcasecmp(col->tlabel, colname) == 0)
			return c;
	}
	return -1;
}

void fits_add_uint_size(qfits_header* header) {
	fits_header_add_int(header, "UINT_SZ", sizeof(uint), "sizeof(uint)");
}

void fits_add_double_size(qfits_header* header) {
	fits_header_add_int(header, "DUBL_SZ", sizeof(double), "sizeof(double)");
}

int fits_check_uint_size(const qfits_header* header) {
  int uintsz;
  uintsz = qfits_header_getint(header, "UINT_SZ", -1);
  if (sizeof(uint) != uintsz) {
    fprintf(stderr, "File was written with sizeof(uint)=%i, but currently sizeof(uint)=%u.\n",
	    uintsz, (uint)sizeof(uint));
        return -1;
	}
    return 0;
}

int fits_check_double_size(const qfits_header* header) {
    int doublesz;
    doublesz = qfits_header_getint(header, "DUBL_SZ", -1);
    if (sizeof(double) != doublesz) {
      fprintf(stderr, "File was written with sizeof(double)=%i, but currently sizeof(double)=%u.\n",
	      doublesz, (uint)sizeof(double));
      return -1;
	}
    return 0;
}

int fits_check_endian(const qfits_header* header) {
    char* filestr;
    char* localstr;
	char pretty[FITS_LINESZ+1];

	filestr = qfits_header_getstr(header, "ENDIAN");
    if (!filestr) {
        // No ENDIAN header found.
        return 1;
    }
	qfits_pretty_string_r(filestr, pretty);
	filestr = pretty;

    localstr = fits_get_endian_string();
	if (strcmp(filestr, localstr)) {
		fprintf(stderr, "File was written with endianness %s, this machine has endianness %s.\n", filestr, localstr);
		return -1;
	}
    return 0;
}
