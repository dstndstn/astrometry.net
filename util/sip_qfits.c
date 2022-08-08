/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <string.h>

#include "qfits_rw.h"
#include "qfits_tools.h"

#include "os-features.h"
#include "sip_qfits.h"
#include "an-bool.h"
#include "fitsioutils.h"
#include "errors.h"
#include "log.h"
#include "ioutils.h"
#include "anqfits.h"
#include "mathutil.h"

sip_t* sip_from_string(const char* str, int slen, sip_t* dest) {
    qfits_header* hdr;
    sip_t* rtn;
    if (slen == 0) {
        slen = strlen(str);
    }
    hdr = qfits_header_read_hdr_string((const unsigned char*)str, slen);
    if (!hdr) {
        ERROR("Failed to parse a FITS header from the given string");
        return NULL;
    }
    rtn = sip_read_header(hdr, dest);
    qfits_header_destroy(hdr);
    return rtn;
}

sip_t* sip_read_tan_or_sip_header_file_ext(const char* wcsfn, int ext, sip_t* dest, anbool forcetan) {
    sip_t* rtn;
    if (forcetan) {
        sip_t sip;
        memset(&sip, 0, sizeof(sip_t));
        if (!tan_read_header_file_ext(wcsfn, ext, &(sip.wcstan))) {
            ERROR("Failed to parse TAN header from file %s, extension %i", wcsfn, ext);
            return NULL;
        }
        if (!dest)
            dest = malloc(sizeof(sip_t));
        memcpy(dest, &sip, sizeof(sip_t));
        return dest;
    } else {
        rtn = sip_read_header_file_ext(wcsfn, ext, dest);
        if (!rtn)
            ERROR("Failed to parse SIP header from file %s, extension %i", wcsfn, ext);
        return rtn;
    }
}

int sip_write_to(const sip_t* sip, FILE* fid) {
    qfits_header* hdr;
    int res;
    if ((sip->a_order == 0) && (sip->b_order == 0) &&
        (sip->ap_order == 0) && (sip->bp_order == 0))
        return tan_write_to(&(sip->wcstan), fid);
    hdr = sip_create_header(sip);
    if (!hdr) {
        ERROR("Failed to create FITS header from WCS");
        return -1;
    }
    res = qfits_header_dump(hdr, fid);
    qfits_header_destroy(hdr);
    return res;
}

int sip_write_to_file(const sip_t* sip, const char* fn) {
    FILE* fid;
    int res;
    if ((sip->a_order == 0) && (sip->b_order == 0) &&
        (sip->ap_order == 0) && (sip->bp_order == 0))
        return tan_write_to_file(&(sip->wcstan), fn);
		
    fid = fopen(fn, "wb");
    if (!fid) {
        SYSERROR("Failed to open file \"%s\" to write WCS header", fn);
        return -1;
    }
    res = sip_write_to(sip, fid);
    if (res) {
        ERROR("Failed to write FITS header to file \"%s\"", fn);
        return -1;
    }
    if (fclose(fid)) {
        SYSERROR("Failed to close file \"%s\" after writing WCS header", fn);
        return -1;
    }
    return 0;
}

int tan_write_to(const tan_t* tan, FILE* fid) {
    qfits_header* hdr;
    int res;
    hdr = tan_create_header(tan);
    if (!hdr) {
        ERROR("Failed to create FITS header from WCS");
        return -1;
    }
    res = qfits_header_dump(hdr, fid);
    qfits_header_destroy(hdr);
    return res;
}

int tan_write_to_file(const tan_t* tan, const char* fn) {
    FILE* fid;
    int res;
    fid = fopen(fn, "wb");
    if (!fid) {
        SYSERROR("Failed to open file \"%s\" to write WCS header", fn);
        return -1;
    }
    res = tan_write_to(tan, fid);
    if (res) {
        ERROR("Failed to write FITS header to file \"%s\"", fn);
        return -1;
    }
    if (fclose(fid)) {
        SYSERROR("Failed to close file \"%s\" after writing WCS header", fn);
        return -1;
    }
    return 0;
}

static void wcs_hdr_common(qfits_header* hdr, const tan_t* tan) {
    qfits_header_add(hdr, "WCSAXES", "2", NULL, NULL);
    qfits_header_add(hdr, "EQUINOX", "2000.0", "Equatorial coordinates definition (yr)", NULL);
    qfits_header_add(hdr, "LONPOLE", "180.0", NULL, NULL);
    qfits_header_add(hdr, "LATPOLE", "0.0", NULL, NULL);

    fits_header_add_double(hdr, "CRVAL1", tan->crval[0], "RA  of reference point");
    fits_header_add_double(hdr, "CRVAL2", tan->crval[1], "DEC of reference point");
    fits_header_add_double(hdr, "CRPIX1", tan->crpix[0], "X reference pixel");
    fits_header_add_double(hdr, "CRPIX2", tan->crpix[1], "Y reference pixel");
    qfits_header_add(hdr, "CUNIT1", "deg", "X pixel scale units", NULL);
    qfits_header_add(hdr, "CUNIT2", "deg", "Y pixel scale units", NULL);

    fits_header_add_double(hdr, "CD1_1", tan->cd[0][0], "Transformation matrix");
    fits_header_add_double(hdr, "CD1_2", tan->cd[0][1], "");
    fits_header_add_double(hdr, "CD2_1", tan->cd[1][0], "");
    fits_header_add_double(hdr, "CD2_2", tan->cd[1][1], "");

    if (tan->imagew > 0.0)
        fits_header_add_double(hdr, "IMAGEW", tan->imagew, "Image width,  in pixels.");
    if (tan->imageh > 0.0)
        fits_header_add_double(hdr, "IMAGEH", tan->imageh, "Image height, in pixels.");
}

int sip_get_image_size(const qfits_header* hdr, int* pW, int* pH) {
    int W, H;
    W = qfits_header_getint(hdr, "IMAGEW", 0);
    debug("sip_get_image_size: IMAGEW = %i\n", W);
    H = qfits_header_getint(hdr, "IMAGEH", 0);
    debug("sip_get_image_size: IMAGEH = %i\n", H);
    if (W == 0 || H == 0) {
        // no IMAGE[WH].  Check for fpack-compressed image.
        int eq;
        char* str = fits_get_dupstring(hdr, "XTENSION");
        //printf("XTENSION: '%s'\n", str);
        // qfits_header_getstr turns the string double-quotes to single-quotes
        eq = streq(str, "BINTABLE");
        free(str);
        if (eq) {
            // ZNAXIS1 =                 2046 / length of data axis 1
            // ZNAXIS2 =                 4094 / length of data axis 2
            if (!W) {
                W = qfits_header_getint(hdr, "ZNAXIS1", 0);
                debug("sip_get_image_size: ZNAXIS1 = %i\n", W);
            }
            if (!H) {
                H = qfits_header_getint(hdr, "ZNAXIS2", 0);
                debug("sip_get_image_size: ZNAXIS2 = %i\n", H);
            }
        }
        if (!W) {
            W = qfits_header_getint(hdr, "NAXIS1", 0);
            debug("sip_get_image_size: NAXIS1 = %i\n", W);
        }
        if (!H) {
            H = qfits_header_getint(hdr, "NAXIS2", 0);
            debug("sip_get_image_size: NAXIS2 = %i\n", H);
        }
    }
    if (pW) *pW = W;
    if (pH) *pH = H;
    return 0;
}

static void add_polynomial(qfits_header* hdr, const char* format,
                           int order, const double* data, int datastride) {
    int i, j;
    char key[64];
    for (i=0; i<=order; i++)
        for (j=0; (i+j)<=order; j++) {
            //if (i+j < 1)
            //	continue;
            //if (drop_linear && (i+j < 2))
            //	continue;
            sprintf(key, format, i, j);
            fits_header_add_double(hdr, key, data[i*datastride + j], "");
        }
}

void sip_add_to_header(qfits_header* hdr, const sip_t* sip) {
    wcs_hdr_common(hdr, &(sip->wcstan));
    if (sip->wcstan.sin) {
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE2", "DEC--SIN-SIP", "SIN projection + SIP distortions", NULL);
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE1", "RA---SIN-SIP", "SIN projection + SIP distortions", NULL);
    } else {
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE2", "DEC--TAN-SIP", "TAN (gnomic) projection + SIP distortions", NULL);
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE1", "RA---TAN-SIP", "TAN (gnomic) projection + SIP distortions", NULL);
    }

    fits_header_add_int(hdr, "A_ORDER", sip->a_order, "Polynomial order, axis 1");
    add_polynomial(hdr, "A_%i_%i", sip->a_order, (double*)sip->a, SIP_MAXORDER);

    fits_header_add_int(hdr, "B_ORDER", sip->b_order, "Polynomial order, axis 2");
    add_polynomial(hdr, "B_%i_%i", sip->b_order, (double*)sip->b, SIP_MAXORDER);

    fits_header_add_int(hdr, "AP_ORDER", sip->ap_order, "Inv polynomial order, axis 1");
    add_polynomial(hdr, "AP_%i_%i", sip->ap_order, (double*)sip->ap, SIP_MAXORDER);

    fits_header_add_int(hdr, "BP_ORDER", sip->bp_order, "Inv polynomial order, axis 2");
    add_polynomial(hdr, "BP_%i_%i", sip->bp_order, (double*)sip->bp, SIP_MAXORDER);
}

qfits_header* sip_create_header(const sip_t* sip) {
    qfits_header* hdr = qfits_table_prim_header_default();
    sip_add_to_header(hdr, sip);
    return hdr;
}

void tan_add_to_header(qfits_header* hdr, const tan_t* tan) {
    wcs_hdr_common(hdr, tan);
    if (tan->sin) {
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE2", "DEC--SIN", "SIN projection", NULL);
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE1", "RA---SIN", "SIN projection", NULL);
    } else {
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE2", "DEC--TAN", "TAN (gnomic) projection", NULL);
        qfits_header_add_after(hdr, "WCSAXES", "CTYPE1", "RA---TAN", "TAN (gnomic) projection", NULL);
    }
}

qfits_header* tan_create_header(const tan_t* tan) {
    qfits_header* hdr = qfits_table_prim_header_default();
    tan_add_to_header(hdr, tan);
    return hdr;
}

static void* read_header_file(const char* fn, int ext, anbool only, void* dest,
                              void* (*readfunc)(const qfits_header*, void*)) {
    qfits_header* hdr;
    void* result;
    if (only) {
        hdr = anqfits_get_header_only(fn, ext);
    } else {
        hdr = anqfits_get_header2(fn, ext);
    }
    if (!hdr) {
        ERROR("Failed to read FITS header from file \"%s\" extension %i", fn, ext);
        return NULL;
    }
    result = readfunc(hdr, dest);
    if (!result) {
        ERROR("Failed to parse WCS header from file \"%s\" extension %i", fn, ext);
    }
    qfits_header_destroy(hdr);
    return result;
}

// silly little dispatch function to avoid casting - I like a modicum of type safety
static void* call_sip_read_header(const qfits_header* hdr, void* dest) {
    return sip_read_header(hdr, dest);
}
sip_t* sip_read_header_file(const char* fn, sip_t* dest) {
    return read_header_file(fn, 0, FALSE, dest, call_sip_read_header);
}
sip_t* sip_read_header_file_ext(const char* fn, int ext, sip_t* dest) {
    return read_header_file(fn, ext, TRUE, dest, call_sip_read_header);
}
sip_t* sip_read_header_file_ext_only(const char* fn, int ext, sip_t* dest) {
    return read_header_file(fn, ext, TRUE, dest, call_sip_read_header);
}

static void* call_tan_read_header(const qfits_header* hdr, void* dest) {
    return tan_read_header(hdr, dest);
}
tan_t* tan_read_header_file(const char* fn, tan_t* dest) {
    return read_header_file(fn, 0, FALSE, dest, call_tan_read_header);
}
tan_t* tan_read_header_file_ext(const char* fn, int ext, tan_t* dest) {
    return read_header_file(fn, ext, FALSE, dest, call_tan_read_header);
}
tan_t* tan_read_header_file_ext_only(const char* fn, int ext, tan_t* dest) {
    return read_header_file(fn, ext, TRUE, dest, call_tan_read_header);
}


static anbool read_polynomial(const qfits_header* hdr, const char* format,
                              int order, double* data, int datastride,
                              anbool skip_linear, anbool skip_zero) {
    int i, j;
    char key[64];
    double nil = -LARGE_VAL;
    double val;
    for (i=0; i<=order; i++)
        for (j=0; (i+j)<=order; j++) {
            if (skip_zero && (i+j < 1))
                continue;
            if (skip_linear && (i+j < 2))
                continue;
            sprintf(key, format, i, j);
            val = qfits_header_getdouble(hdr, key, nil);
            if (val == nil) {
                // don't warn if linear terms are "missing"
                if (i+j >= 2) {
                    ERROR("SIP: warning: key \"%s\" not found; setting to zero.", key);
                }
                val=0.0;
            }
            data[i*datastride + j] = val;
        }
    return TRUE;
}

sip_t* sip_read_header(const qfits_header* hdr, sip_t* dest) {
    sip_t sip;
    char* str;
    const char* key;
    const char* expect;
    const char* expect2;
    anbool is_sin;
    anbool is_tan;
    anbool skip_linear;
    anbool skip_zero;
    char pretty[FITS_LINESZ];

    memset(&sip, 0, sizeof(sip_t));

    key = "CTYPE1";
    expect  = "RA---TAN-SIP";
    expect2 = "RA---SIN-SIP";
    str = qfits_header_getstr(hdr, key);
    str = qfits_pretty_string_r(str, pretty);
    if (!str) {
        ERROR("SIP header: no key \"%s\"", key);
        return NULL;
    }
    is_tan = (strncmp(str, expect, strlen(expect)) == 0);
    is_sin = (strncmp(str, expect2, strlen(expect2)) == 0);
    if (!(is_tan || is_sin)) {
        if (!tan_read_header(hdr, &(sip.wcstan))) {
            ERROR("SIP: failed to read TAN header");
            return NULL;
        }
        goto gohome;
    }

    key = "CTYPE2";
    if (is_sin) {
        expect = "DEC--SIN-SIP";
    } else {
        expect = "DEC--TAN-SIP";
    }
    str = qfits_header_getstr(hdr, key);
    str = qfits_pretty_string_r(str, pretty);
    if (!str || strncmp(str, expect, strlen(expect))) {
        ERROR("SIP header: incorrect key \"%s\": expected \"%s\", got \"%s\"", key, expect, str);
        return NULL;
    }

    if (!tan_read_header(hdr, &sip.wcstan)) {
        ERROR("SIP: failed to read TAN header");
        return NULL;
    }

    sip.a_order  = qfits_header_getint(hdr, "A_ORDER", -1);
    sip.b_order  = qfits_header_getint(hdr, "B_ORDER", -1);
    sip.ap_order = qfits_header_getint(hdr, "AP_ORDER", 0);
    sip.bp_order = qfits_header_getint(hdr, "BP_ORDER", 0);

    if ((sip.a_order == -1) || 
        (sip.b_order == -1)) {
        ERROR("SIP: failed to read polynomial orders (A_ORDER=%i, B_ORDER=%i, -1 means absent)\n",
              sip.a_order, sip.b_order);
        return NULL;
    }
    if ((sip.ap_order == 0) || 
        (sip.bp_order == 0)) {
        logverb("Warning: SIP: failed to read polynomial orders (A_ORDER=%i, B_ORDER=%i (-1 means absent), AP_ORDER=%i, BP_ORDER=%i, (0 means absent)\n",
                sip.a_order, sip.b_order, sip.ap_order, sip.bp_order);
    }

    if ((sip.a_order > SIP_MAXORDER) || 
        (sip.b_order > SIP_MAXORDER) || 
        (sip.ap_order > SIP_MAXORDER) || 
        (sip.bp_order > SIP_MAXORDER)) {
        ERROR("SIP: polynomial orders (A=%i, B=%i, AP=%i, BP=%i) exceeds maximum of %i",
              sip.a_order, sip.b_order, sip.ap_order, sip.bp_order, SIP_MAXORDER);
        return NULL;
    }

    skip_linear = FALSE;
    skip_zero = FALSE;

    if (!read_polynomial(hdr, "A_%i_%i",  sip.a_order,  (double*)sip.a,  SIP_MAXORDER, skip_linear, skip_zero) ||
        !read_polynomial(hdr, "B_%i_%i",  sip.b_order,  (double*)sip.b,  SIP_MAXORDER, skip_linear, skip_zero) ||
        (sip.ap_order > 0 && !read_polynomial(hdr, "AP_%i_%i", sip.ap_order, (double*)sip.ap, SIP_MAXORDER, FALSE, FALSE)) ||
        (sip.bp_order > 0 && !read_polynomial(hdr, "BP_%i_%i", sip.bp_order, (double*)sip.bp, SIP_MAXORDER, FALSE, FALSE))) {
        ERROR("SIP: failed to read polynomial terms");
        return NULL;
    }

 gohome:
    if (!dest)
        dest = malloc(sizeof(sip_t));

    memcpy(dest, &sip, sizeof(sip_t));
    return dest;
}

static int check_tan_ctypes(char* ct1, char* ct2, anbool* is_sin) {
    const char* ra  = "RA---TAN";
    const char* dec = "DEC--TAN";
    const char* ra2  = "RA---SIN";
    const char* dec2 = "DEC--SIN";
    int NC = 8;
    *is_sin = FALSE;
    if (!ct1 || !ct2)
        return -1;
    if (strlen(ct1) < NC || strlen(ct2) < NC)
        return -1;
    if ((strncmp(ct1, ra, NC) == 0) && (strncmp(ct2, dec, NC) == 0))
        return 0;
    if ((strncmp(ct1, dec, NC) == 0) && (strncmp(ct2, ra, NC) == 0))
        return 1;

    if ((strncmp(ct1, ra2, NC) == 0) && (strncmp(ct2, dec2, NC) == 0)) {
        *is_sin = TRUE;
        return 0;
    }
    if ((strncmp(ct1, dec2, NC) == 0) && (strncmp(ct2, ra2, NC) == 0)) {
        *is_sin = TRUE;
        return 1;
    }
    return -1;
}

tan_t* tan_read_header(const qfits_header* hdr, tan_t* dest) {
    tan_t tan;
    double nil = -1e300;
    char* ct1;
    char* ct2;
    int swap;
    int W, H;
    anbool is_sin;

    memset(&tan, 0, sizeof(tan_t));

    ct1 = fits_get_dupstring(hdr, "CTYPE1");
    ct2 = fits_get_dupstring(hdr, "CTYPE2");
    swap = check_tan_ctypes(ct1, ct2, &is_sin);
    if (swap == -1) {
        ERROR("TAN header: expected CTYPE1 = RA---TAN, CTYPE2 = DEC--TAN "
              "(or vice versa), or RA---SIN, DEC--SIN or vice versa; "
              "got CTYPE1 = \"%s\", CYTPE2 = \"%s\"\n",
              ct1, ct2);
    }
    free(ct1);
    free(ct2);
    if (swap == -1)
        return NULL;

    sip_get_image_size(hdr, &W, &H);
    tan.imagew = W;
    tan.imageh = H;

    {
        const char* keys[] = { "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
                               "CD1_1", "CD1_2", "CD2_1", "CD2_2" };
        double* vals[] = { &(tan.crval[0]), &(tan.crval[1]),
                           &(tan.crpix[0]), &(tan.crpix[1]),
                           &(tan.cd[0][0]), &(tan.cd[0][1]),
                           &(tan.cd[1][0]), &(tan.cd[1][1]) };
        int i;
        for (i=0; i<4; i++) {
            *(vals[i]) = qfits_header_getdouble(hdr, keys[i], nil);
            if (*(vals[i]) == nil) {
                ERROR("TAN header: missing or invalid value for \"%s\"", keys[i]);
                return NULL;
            }
        }
        // Try CD
        int gotcd = 1;
        char* complaint = NULL;
        for (i=4; i<8; i++) {
            *(vals[i]) = qfits_header_getdouble(hdr, keys[i], nil);
            if (*(vals[i]) == nil) {
                asprintf_safe(&complaint, "TAN header: missing or invalid value for key \"%s\"", keys[i]);
                gotcd = 0;
                break;
            }
        }
        if (!gotcd) {
            double cdelt1,cdelt2;
            // Try CDELT
            char* key = "CDELT1";
            cdelt1 = qfits_header_getdouble(hdr, key, nil);
            if (cdelt1 == nil) {
                ERROR("%s; also tried but didn't find \"%s\"", complaint, key);
                free(complaint);
                return NULL;
            }
            key = "CDELT2";
            cdelt2 = qfits_header_getdouble(hdr, key, nil);
            if (cdelt2 == nil) {
                ERROR("%s; also tried but didn't find \"%s\"", complaint, key);
                free(complaint);
                return NULL;
            }
            // Try PCi_j
            double pc11 = qfits_header_getdouble(hdr, "PC1_1", 1.0);
            double pc12 = qfits_header_getdouble(hdr, "PC1_2", 0.0);
            double pc21 = qfits_header_getdouble(hdr, "PC2_1", 0.0);
            double pc22 = qfits_header_getdouble(hdr, "PC2_2", 1.0);

            tan.cd[0][0] = cdelt1 * pc11;
            tan.cd[0][1] = cdelt1 * pc12;
            tan.cd[1][0] = cdelt2 * pc21;
            tan.cd[1][1] = cdelt2 * pc22;
        }
    }

    if (swap == 1) {
        double tmp;
        tmp = tan.crval[0];
        tan.crval[0] = tan.crval[1];
        tan.crval[1] = tmp;
        // swap CD1_1 <-> CD2_1
        tmp = tan.cd[0][0];
        tan.cd[0][0] = tan.cd[1][0];
        tan.cd[1][0] = tmp;
        // swap CD1_2 <-> CD2_2
        tmp = tan.cd[0][1];
        tan.cd[0][1] = tan.cd[1][1];
        tan.cd[1][1] = tmp;
    }

    tan.sin = is_sin;

    if (!dest)
        dest = malloc(sizeof(tan_t));
    memcpy(dest, &tan, sizeof(tan_t));
    return dest;
}
