/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <assert.h>

#ifdef WCSLIB_EXISTS
#include <wcshdr.h>
#include <wcs.h>
#endif

#ifdef WCSTOOLS_EXISTS
#include <libwcs/wcs.h>
#endif

#include "os-features.h"
#include "qfits_std.h"
#include "qfits_header.h"
#include "qfits_rw.h"
#include "anwcs.h"
#include "anqfits.h"
#include "errors.h"
#include "log.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "starutil.h"
#include "mathutil.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "bl.h"

struct anwcslib_t {
    struct wcsprm* wcs;
    // Image width and height, in pixels.
    int imagew;
    int imageh;
};
typedef struct anwcslib_t anwcslib_t;

/*
 This is ugly... this macro gets defined differently depending on
 whether wcslib is available or not... I couldn't figure out how to put
 the #ifdef inside the macro definition to make it cleaner.
 */

#if defined(WCSLIB_EXISTS) && defined(WCSTOOLS_EXISTS)

#define ANWCS_DISPATCH(anwcs, action, defaction, func, ...)	\
    do {                                                        \
        assert(anwcs);                                          \
        switch (anwcs->type) {                                  \
        case ANWCS_TYPE_WCSLIB:                                 \
            {                                                   \
                anwcslib_t* anwcslib = anwcs->data;             \
                action wcslib_##func(anwcslib, ##__VA_ARGS__);  \
                break;                                          \
            }                                                   \
        case ANWCS_TYPE_SIP:                                    \
            {                                                   \
                sip_t* sip = anwcs->data;                       \
                action ansip_##func(sip, ##__VA_ARGS__);        \
                break;                                          \
            }                                                   \
        case ANWCS_TYPE_WCSTOOLS:                               \
            {                                                   \
                struct WorldCoor* wcs = anwcs->data;            \
                action wcstools_##func(wcs, ##__VA_ARGS__);     \
                break;                                          \
            }                                                   \
        default:                                                \
            ERROR("Unknown anwcs type %i", anwcs->type);        \
            defaction;                                          \
        }                                                       \
    } while (0)

#elif defined(WCSLIB_EXISTS)

#define ANWCS_DISPATCH(anwcs, action, defaction, func, ...)	\
    do {                                                        \
        assert(anwcs);                                          \
        switch (anwcs->type) {                                  \
        case ANWCS_TYPE_WCSLIB:                                 \
            {                                                   \
                anwcslib_t* anwcslib = anwcs->data;             \
                action wcslib_##func(anwcslib, ##__VA_ARGS__);  \
                break;                                          \
            }                                                   \
        case ANWCS_TYPE_SIP:                                    \
            {                                                   \
                sip_t* sip = anwcs->data;                       \
                action ansip_##func(sip, ##__VA_ARGS__);        \
                break;                                          \
            }                                                   \
        default:                                                \
            ERROR("Unknown anwcs type %i", anwcs->type);        \
            defaction;                                          \
        }                                                       \
    } while (0)

#elif defined(WCSTOOLS_EXISTS)

#define ANWCS_DISPATCH(anwcs, action, defaction, func, ...)	\
    do {                                                        \
        assert(anwcs);                                          \
        switch (anwcs->type) {                                  \
        case ANWCS_TYPE_SIP:                                    \
            {                                                   \
                sip_t* sip = anwcs->data;                       \
                action ansip_##func(sip, ##__VA_ARGS__);        \
                break;                                          \
            }                                                   \
        case ANWCS_TYPE_WCSTOOLS:                               \
            {                                                   \
                struct WorldCoor* wc = anwcs->data;             \
                action wcstools_##func(wc, ##__VA_ARGS__);      \
                break;                                          \
            }                                                   \
        default:                                                \
            ERROR("Unknown anwcs type %i", anwcs->type);        \
            defaction;                                          \
        }                                                       \
    } while (0)

#else

// No WCSLIB.
#define ANWCS_DISPATCH(anwcs, action, defaction, func, ...)	\
    do {                                                        \
        assert(anwcs);                                          \
        switch (anwcs->type) {                                  \
        case ANWCS_TYPE_WCSLIB:                                 \
            ERROR("WCSlib support was not compiled in");        \
            defaction;                                          \
            break;                                              \
        case ANWCS_TYPE_SIP:                                    \
            {                                                   \
                sip_t* sip = anwcs->data;                       \
                action ansip_##func(sip, ##__VA_ARGS__);        \
                break;                                          \
            }                                                   \
        default:                                                \
            ERROR("Unknown anwcs type %i", anwcs->type);        \
            defaction;                                          \
        }                                                       \
    } while (0)

#endif


/////////////////// wcstools implementations //////////////////////////

#if defined(WCSTOOLS_EXISTS)

static double wcstools_imagew(const struct WorldCoor* wcs) {
    return wcs->nxpix;
}
static double wcstools_imageh(const struct WorldCoor* wcs) {
    return wcs->nypix;
}

static int wcstools_pixelxy2radec(const struct WorldCoor* wcs,
                                  double px, double py,
                                  double* ra, double* dec) {
    pix2wcs(wcs, px, py, ra, dec);
    return 0;
}

static int wcstools_radec2pixelxy(const struct WorldCoor* wcs,
                                  double ra, double dec,
                                  double* px, double* py) {
    int offscl;
    wcs2pix(wcs, ra, dec, px, py, &offscl);
    return offscl;
}

static void wcstools_print(const struct WorldCoor* wcs, FILE* fid) {
    fprintf(fid, "AN WCS type: wcstools\n");
}

static void wcstools_free(struct WorldCoor* wcs) {
    wcsfree(wcs);
}

static int wcstools_add_to_header(struct WorldCoor* wcs, qfits_header* hdr) {
    logerr("UNIMPLEMENTED");
    return -1;
}
static void wcstools_set_size(struct WorldCoor* wcs, int w, int h) {
    logerr("UNIMPLEMENTED");
}
static anbool wcstools_radec_is_inside_image(struct WorldCoor* wcs, double ra, double dec) {
    logerr("UNIMPLEMENTED");
    return FALSE;
}
static double wcstools_pixel_scale(struct WorldCoor* wcs) {
    logerr("UNIMPLEMENTED");
    return 0.0;
}
static int wcstools_write(struct WorldCoor* wcs, const char* filename) {
    logerr("UNIMPLEMENTED");
    return -1;
}
static int wcstools_write_to(const struct WorldCoor* wcs, FILE* fid) {
    logerr("UNIMPLEMENTED");
    return -1;
}
static int wcstools_scale_wcs(struct WorldCoor* anwcs, double scale) {
    logerr("UNIMPLEMENTED");
    return -1;
}
static int wcstools_rotate_wcs(struct WorldCoor* anwcs, double rot) {
    logerr("UNIMPLEMENTED");
    return -1;
}


#endif


/////////////////// wcslib implementations //////////////////////////

#ifdef WCSLIB_EXISTS

static double wcslib_imagew(const anwcslib_t* anwcs) {
    return anwcs->imagew;
}
static double wcslib_imageh(const anwcslib_t* anwcs) {
    return anwcs->imageh;
}

static int wcslib_pixelxy2radec(const anwcslib_t* anwcslib, double px, double py, double* ra, double* dec) {
    double pix[2];
    double world[2];
    double phi;
    double theta;
    double imgcrd[2];
    int status = 0;
    int code;
    struct wcsprm* wcs = anwcslib->wcs;
    pix[0] = px;
    pix[1] = py;
    code = wcsp2s(wcs, 1, 0, pix, imgcrd, &phi, &theta, world, &status);
    /*
     int wcsp2s(struct wcsprm *wcs, int ncoord, int nelem, const double pixcrd[],
     double imgcrd[], double phi[], double theta[], double world[],
     int stat[]);
     */

    if (code) {
        //ERROR("Wcslib's wcsp2s() failed: code=%i, status=%i (%s); (x,y)=(%g,%g)", code, status, wcs_errmsg[status], px, py);
        logverb("Wcslib's wcsp2s() failed: code=%i, status=%i (%s); (x,y)=(%g,%g)", code, status, wcs_errmsg[status], px, py);
        return -1;
    }
    if (ra)  *ra  = world[wcs->lng];
    if (dec) *dec = world[wcs->lat];
    return 0;
}

static int wcslib_radec2pixelxy(const anwcslib_t* anwcslib, double ra, double dec, double* px, double* py) {
    double pix[2];
    double world[2];
    double phi;
    double theta;
    double imgcrd[2];
    int status = 0;
    int code;
    struct wcsprm* wcs = anwcslib->wcs;
    world[wcs->lng] = ra;
    world[wcs->lat] = dec;
    code = wcss2p(wcs, 1, 0, world, &phi, &theta, imgcrd, pix, &status);
    /*
     int wcss2p(struct wcsprm *wcs, int ncoord, int nelem, const double world[],
     double phi[], double theta[], double imgcrd[], double pixcrd[],
     int stat[]);
     */
    if (code) {
        ERROR("Wcslib's wcss2p() failed: code=%i, status=%i", code, status);
        return -1;
    }
    if (px) *px = pix[0];
    if (py) *py = pix[1];
    return 0;
}

static anbool wcslib_radec_is_inside_image(anwcslib_t* wcslib, double ra, double dec) {
    double px, py;
    if (wcslib_radec2pixelxy(wcslib, ra, dec, &px, &py))
        return FALSE;
    return (px >= 1 && px <= wcslib->imagew &&
            py >= 1 && py <= wcslib->imageh);
}

//// This was copied wholesale from sip-utils.c /////

struct radecbounds {
    double rac, decc;
    double ramin, ramax, decmin, decmax;
};

static void radec_bounds_callback(const anwcs_t* wcs, double x, double y, double ra, double dec, void* token) {
    struct radecbounds* b = token;
    b->decmin = MIN(b->decmin, dec);
    b->decmax = MAX(b->decmax, dec);
    if (ra - b->rac > 180)
        // wrap-around: racenter < 180, ra has gone < 0 but been wrapped around to > 180
        ra -= 360;
    if (b->rac - ra > 180)
        // wrap-around: racenter > 180, ra has gone > 360 but wrapped around to > 0.
        ra += 360;

    b->ramin = MIN(b->ramin, ra);
    b->ramax = MAX(b->ramax, ra);
}

static void wcslib_radec_bounds(const anwcs_t* genwcs, const anwcslib_t* wcs, int stepsize,
                                double* pramin, double* pramax,
                                double* pdecmin, double* pdecmax) {
    struct radecbounds b;

    anwcs_get_radec_center_and_radius(genwcs, &(b.rac), &(b.decc), NULL);
    b.ramin  = b.ramax = b.rac;
    b.decmin = b.decmax = b.decc;
    anwcs_walk_image_boundary(genwcs, stepsize, radec_bounds_callback, &b);

    // Check for poles...
    // north pole
    if (anwcs_radec_is_inside_image(genwcs, 0, 90)) {
        b.ramin = 0;
        b.ramax = 360;
        b.decmax = 90;
    }
    if (anwcs_radec_is_inside_image(genwcs, 0, -90)) {
        b.ramin = 0;
        b.ramax = 360;
        b.decmin = -90;
    }

    if (pramin) *pramin = b.ramin;
    if (pramax) *pramax = b.ramax;
    if (pdecmin) *pdecmin = b.decmin;
    if (pdecmax) *pdecmax = b.decmax;
}

static void wcslib_print(const anwcslib_t* anwcslib, FILE* fid) {
    fprintf(fid, "AN WCS type: wcslib\n");
    wcsprt(anwcslib->wcs);
    fprintf(fid, "Image size: %i x %i\n", anwcslib->imagew, anwcslib->imageh);
}

static void wcslib_free(anwcslib_t* anwcslib) {
    wcsfree(anwcslib->wcs);
    free(anwcslib->wcs);
    free(anwcslib);
}

static double wcslib_pixel_scale(const anwcslib_t* anwcslib) {
    struct wcsprm* wcs = anwcslib->wcs;
    //double* cd = wcs->m_cd;
    double* cd = wcs->cd;
    double ps;
    //printf("WCSlib pixel scale: cd %g,%g,%g,%g\n", cd[0], cd[1], cd[2], cd[3]);
    // HACK -- assume "cd" elements are set...
    ps = deg2arcsec(sqrt(fabs(cd[0]*cd[3] - cd[1]*cd[2])));

    if (ps == 0.0) {
        // Try CDELT
        //printf("WCSlib pixel scale: cdelt %g,%g\n", wcs->cdelt[0], wcs->cdelt[1]);
        ps = deg2arcsec(sqrt(fabs(wcs->cdelt[0] * wcs->cdelt[1])));
    }

    assert(ps > 0.0);
    return ps;
}

static int wcslib_write_to(const anwcslib_t* anwcslib, FILE* fid) {
    int res;
    int Ncards;
    char* hdrstr;
    char line[81];
    char spaces[81];
    char val[32];
    const char* hdrformat = "%-8s= %20s /%s";
    sl* lines = NULL;
    int npad;

    res = wcshdo(-1, anwcslib->wcs, &Ncards, &hdrstr);
    if (res) {
        ERROR("wcshdo() failed: %s", wcshdr_errmsg[res]);
        return -1;
    }

    int i;
    printf("wcslib header:\n");
    for (i=0; i<Ncards; i++)
        printf("%.80s\n", hdrstr + i*80);
    printf("\n\n");

    lines = sl_new(16);

    memset(spaces, ' ', sizeof(spaces));
    spaces[sizeof(spaces)-1] = '\0';

    snprintf(line, sizeof(line), hdrformat, "SIMPLE", "T", spaces);
    sl_append(lines, line);
    snprintf(line, sizeof(line), hdrformat, "BITPIX", "8", spaces);
    sl_append(lines, line);
    snprintf(line, sizeof(line), hdrformat, "NAXIS", "0", spaces);
    sl_append(lines, line);
    snprintf(line, sizeof(line), hdrformat, "EXTEND", "T", spaces);
    sl_append(lines, line);

    sprintf(val, "%i", anwcslib->imagew);
    snprintf(line, sizeof(line), hdrformat, "IMAGEW", val, spaces);
    sl_append(lines, line);
    sprintf(val, "%i", anwcslib->imageh);
    snprintf(line, sizeof(line), hdrformat, "IMAGEH", val, spaces);
    sl_append(lines, line);

    for (i=0; i<Ncards; i++) {
        snprintf(line, sizeof(line), "%.80s%s", hdrstr + i*80, spaces);
        sl_append(lines, line);
    }
    snprintf(line, sizeof(line), "END%s", spaces);
    sl_append(lines, line);

    printf("Complete header:\n");
    for (i=0; i<sl_size(lines); i++) {
        printf("|%s|\n", sl_get(lines, i));
    }


    for (i=0; i<sl_size(lines); i++) {
        if (fprintf(fid, "%s", sl_get(lines, i)) < 0) {
            SYSERROR("Failed to write FITS WCS header line");
            return -1;
        }
    }
    npad = 36 - (sl_size(lines) % 36);
    for (i=0; i<npad; i++) {
        if (fprintf(fid, "%s", spaces) < 0) {
            SYSERROR("Failed to write FITS WCS header line");
            return -1;
        }
    }
    return 0;
}

static int wcslib_write(const anwcslib_t* anwcslib, const char* filename) {
    int rtn;
    FILE* fid = fopen(filename, "wb");
    if (!fid) {
        SYSERROR("Failed to open file \"%s\" for FITS WCS output", filename);
        return -1;
    }
    rtn = wcslib_write_to(anwcslib, fid);
    if (fclose(fid)) {
        if (!rtn) {
            SYSERROR("Failed to close output file \"%s\"", filename);
            return -1;
        }
    }
    if (rtn) {
        ERROR("wcslib_write_to file \"%s\" failed", filename);
        return -1;
    }
    return 0;
}

static void wcslib_set_size(anwcslib_t* anwcslib, int W, int H) {
    anwcslib->imagew = W;
    anwcslib->imageh = H;
}

static int wcslib_scale_wcs(anwcslib_t* wcslib, double scale) {
    ERROR("Not implemented!");
    return -1;
}

static int wcslib_rotate_wcs(anwcslib_t* wcslib, double scale) {
    ERROR("Not implemented!");
    return -1;
}

static int wcslib_add_to_header(const anwcslib_t* wcslib, qfits_header* hdr) {
    ERROR("Not implemented!");
    return -1;
}

static void wcslib_get_cd_matrix(const anwcslib_t* wcslib, double* p_cd) {
    ERROR("Not implemented: wcslib_get_cd_matrix!");
    assert(0);
    p_cd[0] = 0;
    p_cd[1] = 0;
    p_cd[2] = 0;
    p_cd[3] = 0;
}

#endif  // end of WCSLIB implementations


/////////////////// sip implementations //////////////////////////

//#define ansip_radec_bounds sip_get_radec_bounds

#define ansip_radec_is_inside_image sip_is_inside_image

#define ansip_imagew sip_imagew
#define ansip_imageh sip_imageh

//#define ansip_pixelxy2radec sip_pixelxy2radec
static int ansip_pixelxy2radec(const sip_t* sip, double px, double py, double* ra, double* dec) {
    sip_pixelxy2radec(sip, px, py, ra, dec);
    return 0;
}

#define ansip_print sip_print_to

#define ansip_free sip_free

#define ansip_pixel_scale sip_pixel_scale

#define ansip_write sip_write_to_file

#define ansip_write_to sip_write_to

static void ansip_set_size(sip_t* sip, int W, int H) {
    sip->wcstan.imagew = W;
    sip->wcstan.imageh = H;
}

static int ansip_scale_wcs(sip_t* sip, double scale) {
    if (sip->a_order || sip->b_order || sip->ap_order || sip->bp_order) {
        // FIXME!!!
        logmsg("Warning: ansip_scale_wcs only scales the TAN, not the SIP coefficients!\n");
    }
    tan_scale(&sip->wcstan, &sip->wcstan, scale);
    return 0;
}

static int ansip_rotate_wcs(sip_t* sip, double angle) {
    logmsg("Warning: ansip_rotate_wcs only scales the TAN, not the SIP coefficients!\n");
    tan_rotate(&sip->wcstan, &sip->wcstan, angle);
    return 0;
}

static int ansip_add_to_header(const sip_t* sip, qfits_header* hdr) {
    sip_add_to_header(hdr, sip);
    return 0;
}

static void ansip_get_cd_matrix(const sip_t* sip, double *p_cd) {
    p_cd[0] = sip->wcstan.cd[0][0];
    p_cd[1] = sip->wcstan.cd[0][1];
    p_cd[2] = sip->wcstan.cd[1][0];
    p_cd[3] = sip->wcstan.cd[1][1];
}


/////////////////// dispatched anwcs_t entry points //////////////////////////

void anwcs_set_size(anwcs_t* anwcs, int W, int H) {
    ANWCS_DISPATCH(anwcs, , , set_size, W, H);
}

void anwcs_get_cd_matrix(const anwcs_t* anwcs, double *p_cd) {
    ANWCS_DISPATCH(anwcs, , , get_cd_matrix, p_cd);
}

void anwcs_get_radec_bounds(const anwcs_t* wcs, int stepsize,
                            double* pramin, double* pramax,
                            double* pdecmin, double* pdecmax) {
    assert(wcs);
    switch (wcs->type) {
    case ANWCS_TYPE_WCSLIB:
#ifdef WCSLIB_EXISTS
        {
            anwcslib_t* anwcslib = wcs->data;
            wcslib_radec_bounds(wcs, anwcslib, stepsize, pramin, pramax, pdecmin, pdecmax);
        }
#else
        ERROR("Wcslib support was not compiled in");
#endif
        break;
    case ANWCS_TYPE_SIP:
        {
            sip_t* sip = wcs->data;
            sip_get_radec_bounds(sip, stepsize, pramin, pramax, pdecmin, pdecmax);
            break;
        }
    default:
        ERROR("Unknown anwcs type %i", wcs->type);
        break;
    }
    //ANWCS_DISPATCH(wcs, , , radec_bounds, stepsize, pramin, pramax, pdecmin, pdecmax);
}

void anwcs_print(const anwcs_t* anwcs, FILE* fid) {
    assert(anwcs);
    assert(fid);
    ANWCS_DISPATCH(anwcs, , , print, fid);
}

void anwcs_print_stdout(const anwcs_t* wcs) {
    anwcs_print(wcs, stdout);
}

void anwcs_free(anwcs_t* anwcs) {
    if (!anwcs)
        return;
    ANWCS_DISPATCH(anwcs, , , free);
    free(anwcs);
}

anbool anwcs_radec_is_inside_image(const anwcs_t* wcs, double ra, double dec) {
    ANWCS_DISPATCH(wcs, return, return FALSE, radec_is_inside_image, ra, dec);
}

double anwcs_imagew(const anwcs_t* anwcs) {
    ANWCS_DISPATCH(anwcs, return, return -1.0, imagew);
}
double anwcs_imageh(const anwcs_t* anwcs) {
    ANWCS_DISPATCH(anwcs, return, return -1.0, imageh);
}

int anwcs_pixelxy2radec(const anwcs_t* anwcs, double px, double py, double* ra, double* dec) {
    ANWCS_DISPATCH(anwcs, return, return -1, pixelxy2radec, px, py, ra, dec);
}

// Approximate pixel scale, in arcsec/pixel, at the reference point.
double anwcs_pixel_scale(const anwcs_t* anwcs) {
    ANWCS_DISPATCH(anwcs, return, return -1, pixel_scale);
}

int anwcs_write(const anwcs_t* wcs, const char* filename) {
    ANWCS_DISPATCH(wcs, return, return -1, write, filename);
}

int anwcs_write_to(const anwcs_t* wcs, FILE* fid) {
    ANWCS_DISPATCH(wcs, return, return -1, write_to, fid);
}

int anwcs_scale_wcs(anwcs_t* anwcs, double scale) {
    ANWCS_DISPATCH(anwcs, return, return -1, scale_wcs, scale);
}

int anwcs_rotate_wcs(anwcs_t* anwcs, double rot) {
    ANWCS_DISPATCH(anwcs, return, return -1, rotate_wcs, rot);
}

int anwcs_add_to_header(const anwcs_t* wcs, qfits_header* hdr) {
    ANWCS_DISPATCH(wcs, return, return -1, add_to_header, hdr);
}






///////////////////////// un-dispatched functions ///////////////////

struct overlap_token {
    const anwcs_t* wcs;
    anbool inside;
};
static void overlap_callback(const anwcs_t* wcs, double x, double y, double ra, double dec, void* token) {
    struct overlap_token* t = token;
    if (t->inside)
        return;
    if (anwcs_radec_is_inside_image(t->wcs, ra, dec))
        t->inside = TRUE;
}

anbool anwcs_overlaps(const anwcs_t* wcs1, const anwcs_t* wcs2, int stepsize) {
    // check for definitely do or don't overlap via bounds:
    double ralo1, rahi1, ralo2, rahi2;
    double declo1, dechi1, declo2, dechi2;
    struct overlap_token token;

    anwcs_get_radec_bounds(wcs1, 1000, &ralo1, &rahi1, &declo1, &dechi1);
    anwcs_get_radec_bounds(wcs2, 1000, &ralo2, &rahi2, &declo2, &dechi2);

    if ((declo1 > dechi2) || (declo2 > dechi1))
        return FALSE;

    // anwcs_get_radec_bounds() has the behavior that ralo <= rahi,
    // but ralo may be < 0 or rahi > 360.
    assert(ralo1 < rahi1);
    assert(ralo2 < rahi2);
    // undo wrap over 360
    if (rahi1 >= 360.0) {
        ralo1 -= 360.0;
        rahi1 -= 360.0;
    }
    if (rahi2 >= 360.0) {
        ralo2 -= 360.0;
        rahi2 -= 360.0;
    }
    assert(rahi1 >= 0);
    assert(rahi2 >= 0);

    if ((ralo1 > rahi2) || (ralo2 > rahi1))
        return FALSE;

    // check for #1 completely inside #2.
    if (ralo1 >= ralo2 && rahi1 <= rahi2 && declo1 >= declo2 && dechi1 <= dechi2)
        return TRUE;

    // check for #2 completely inside #1.
    if (ralo2 >= ralo1 && rahi2 <= rahi1 && declo2 >= declo1 && dechi2 <= dechi1)
        return TRUE;

    // walk the edge of #1, checking whether any point is in #2.
    token.wcs = wcs2;
    token.inside = FALSE;
    if (stepsize == 0)
        stepsize = 100;
    anwcs_walk_image_boundary(wcs1, stepsize, overlap_callback, &token);
    return token.inside;
}

void anwcs_walk_image_boundary(const anwcs_t* wcs, double stepsize,
                               void (*callback)(const anwcs_t* wcs, double x, double y, double ra, double dec, void* token),
                               void* token) {
    int i, side;
    // Walk the perimeter of the image in steps of stepsize pixels
    double W = anwcs_imagew(wcs);
    double H = anwcs_imageh(wcs);
    logverb("Walking WCS image boundary: image size is %g x %g\n", W, H);
    {
        double Xmin = 0.5;
        double Xmax = W + 0.5;
        double Ymin = 0.5;
        double Ymax = H + 0.5;
        double offsetx[] = { Xmin, Xmax, Xmax, Xmin };
        double offsety[] = { Ymin, Ymin, Ymax, Ymax };
        double stepx[] = { +stepsize, 0, -stepsize, 0 };
        double stepy[] = { 0, +stepsize, 0, -stepsize };
        int Nsteps[] = { ceil(W/stepsize), ceil(H/stepsize), ceil(W/stepsize), ceil(H/stepsize) };

        for (side=0; side<4; side++) {
            for (i=0; i<Nsteps[side]; i++) {
                double ra, dec;
                double x, y;
                x = MIN(Xmax, MAX(Xmin, offsetx[side] + i * stepx[side]));
                y = MIN(Ymax, MAX(Ymin, offsety[side] + i * stepy[side]));
                anwcs_pixelxy2radec(wcs, x, y, &ra, &dec);
                callback(wcs, x, y, ra, dec, token);
            }
        }
    }
}

// FIXME -- this is probably the bass-ackwards way -- xyz is more natural; this probably requires converting back and forth between ra,dec and xyz.
int anwcs_pixelxy2xyz(const anwcs_t* wcs, double px, double py, double* xyz) {
    int rtn;
    double ra,dec;
    rtn = anwcs_pixelxy2radec(wcs, px, py, &ra, &dec);
    radecdeg2xyzarr(ra, dec, xyz);
    return rtn;
}

int anwcs_xyz2pixelxy(const anwcs_t* wcs, const double* xyz, double *px, double *py) {
    int rtn;
    double ra,dec;
    xyzarr2radecdeg(xyz, &ra, &dec);
    rtn = anwcs_radec2pixelxy(wcs, ra, dec, px, py);
    return rtn;
}

int anwcs_get_radec_center_and_radius(const anwcs_t* anwcs,
                                      double* p_ra, double* p_dec, double* p_radius) {
    assert(anwcs);
    switch (anwcs->type) {
    case ANWCS_TYPE_WCSLIB:
        {
            anwcslib_t* anwcslib = anwcs->data;
            double x,y;
            double ra1, dec1, ra2, dec2;
            x = anwcslib->imagew/2. + 0.5;
            y = anwcslib->imageh/2. + 0.5;
            if (anwcs_pixelxy2radec(anwcs, x, y, &ra1, &dec1))
                return -1;
            if (p_ra) *p_ra = ra1;
            if (p_dec) *p_dec = dec1;
            // FIXME -- this is certainly not right in general....
            /*
             if (p_radius) {
             if (anwcs_pixelxy2radec(anwcs, 1.0, 1.0, &ra2, &dec2))
             return -1;
             *p_radius = deg_between_radecdeg(ra1, dec1, ra2, dec2);
             }
             */

            // try just moving 1 pixel and extrapolating.
            if (p_radius) {
                if (anwcs_pixelxy2radec(anwcs, x+1, y, &ra2, &dec2))
                    return -1;
                *p_radius = deg_between_radecdeg(ra1, dec1, ra2, dec2) *
                    hypot(anwcslib->imagew, anwcslib->imageh)/2.0;;
            }
        }
        break;

    case ANWCS_TYPE_SIP:
        {
            sip_t* sip;
            sip = anwcs->data;
            if (p_ra || p_dec)
                sip_get_radec_center(sip, p_ra, p_dec);
            if (p_radius)
                *p_radius = sip_get_radius_deg(sip);
        }
        break;

    default:
        ERROR("Unknown anwcs type %i", anwcs->type);
        return -1;
    }
    return 0;
}

anwcs_t* anwcs_new_sip(const sip_t* sip) {
    anwcs_t* anwcs;
    anwcs = calloc(1, sizeof(anwcs_t));
    anwcs->type = ANWCS_TYPE_SIP;
    anwcs->data = sip_create();
    memcpy(anwcs->data, sip, sizeof(sip_t));
    return anwcs;
}

anwcs_t* anwcs_new_tan(const tan_t* tan) {
    sip_t sip;
    sip_wrap_tan(tan, &sip);
    return anwcs_new_sip(&sip);
}

anwcs_t* anwcs_open(const char* filename, int ext) {
    char* errmsg;
    anwcs_t* anwcs = NULL;
    errors_start_logging_to_string();

    // try as SIP:
    anwcs = anwcs_open_sip(filename, ext);
    if (anwcs) {
        errors_pop_state();
        return anwcs;
    } else {
        errmsg = errors_stop_logging_to_string("\n  ");
        logverb("Failed to open file %s, ext %i as SIP:\n%s\n", filename, ext, errmsg);
        free(errmsg);
    }

    // try as WCSLIB:
    anwcs = anwcs_open_wcslib(filename, ext);
    if (anwcs) {
        errors_pop_state();
        return anwcs;
    } else {
        errmsg = errors_stop_logging_to_string(": ");
        logverb("Failed to open file %s, ext %i using WCSLIB: %s", filename, ext, errmsg);
        free(errmsg);
    }

    // try as WCStools:
    anwcs = anwcs_open_wcstools(filename, ext);
    if (anwcs) {
        errors_pop_state();
        return anwcs;
    } else {
        errmsg = errors_stop_logging_to_string(": ");
        logverb("Failed to open file %s, ext %i using WCStools: %s", filename, ext, errmsg);
        free(errmsg);
    }

    return NULL;
}

static anwcs_t* open_tansip(const char* filename, int ext, anbool forcetan) {
    anwcs_t* anwcs = NULL;
    sip_t* sip = NULL;
    sip = sip_read_tan_or_sip_header_file_ext(filename, ext, NULL, forcetan);
    if (!sip) {
        ERROR("Failed to parse SIP header");
        return NULL;
    }
    if (sip->a_order >= 2 && sip->b_order >= 2 &&
        (sip->ap_order == 0 || sip->bp_order == 0)) {
        logverb("Computing inverse SIP polynomial terms...\n");
        sip->ap_order = sip->bp_order = MAX(sip->a_order, sip->b_order) + 1;
        sip_compute_inverse_polynomials(sip, 0, 0, 0, 0, 0, 0);
    }

    anwcs = calloc(1, sizeof(anwcs_t));
    anwcs->type = ANWCS_TYPE_SIP;
    anwcs->data = sip;
    return anwcs;
}

anwcs_t* anwcs_open_tan(const char* filename, int ext) {
    return open_tansip(filename, ext, TRUE);
}

anwcs_t* anwcs_open_sip(const char* filename, int ext) {
    return open_tansip(filename, ext, FALSE);
}

static char* getheader(const char* filename, int ext, int* N) {
    anqfits_t* fits;
    char* hdrstr = NULL;
    assert(N);
    assert(filename);
    fits = anqfits_open(filename);
    if (!fits) {
        ERROR("Failed to open file %s", filename);
        return NULL;
    }
    hdrstr = anqfits_header_get_data(fits, ext, N);
    if (!hdrstr) {
        ERROR("Failed to read header data from file %s, ext %i", filename, ext);
        anqfits_close(fits);
        return NULL;
    }
    anqfits_close(fits);
    return hdrstr;
}

char* anwcs_wcslib_to_string(const anwcs_t* wcs, char** s, int* len) {
#ifndef WCSLIB_EXISTS
    ERROR("Wcslib support was not compiled in");
    return NULL;
#else
    const anwcslib_t* anwcslib = NULL;
    int res;
    char* hdrstr = NULL;
    assert(wcs);
    assert(wcs->type == ANWCS_TYPE_WCSLIB);
    anwcslib = (const anwcslib_t*)wcs->data;
    if (!s)
        s = &hdrstr;

    res = wcshdo(-1, anwcslib->wcs, len, s);
    if (res) {
        ERROR("wcshdo() failed: %s", wcshdr_errmsg[res]);
        return NULL;
    }
    // wcshdo() returns the number of 80-char cards.
    (*len) *= 80;
    /// FIXME -- WIDTH, HEIGHT?
    return *s;
#endif
}

anwcs_t* anwcs_wcslib_from_string(const char* str, int len) {
#ifndef WCSLIB_EXISTS
    ERROR("Wcslib support was not compiled in");
    return NULL;
#else
    int code;
    int nkeys;
    int nrej = 0;
    int nwcs = 0;
    struct wcsprm* wcs = NULL;
    struct wcsprm* wcs2 = NULL;
    anwcs_t* anwcs = NULL;
    anwcslib_t* anwcslib;
    qfits_header* qhdr;
    int W, H;

    /*
     printf("Parsing string: length %i\n", len);
     printf("--------------------------\n");
     //printf("%s\n", str);
     {
     int i;
     for (i=0; i<len; i+=80) {
     char buf[81];
     snprintf(buf, 81, "%s", str+i);
     printf("%s\n", buf);
     }
     }
     printf("--------------------------\n");
     */
	
    qhdr = qfits_header_read_hdr_string((const unsigned char*)str, len);
    if (!qhdr) {
        ERROR("Failed to parse string as qfits header");
        return NULL;
    }
    if (sip_get_image_size(qhdr, &W, &H)) {
        logverb("Failed to find image size in FITS WCS header\n");
        //file %s, ext %i\n", filename, ext);
        //logverb("Header:\n");
        //qfits_header_debug_dump(hdr);
        //logverb("\n");
        W = H = 0;
    }
    qfits_header_destroy(qhdr);

    nkeys = len / FITS_LINESZ;
    code = wcspih((char*)str, nkeys,  WCSHDR_all, 2, &nrej, &nwcs, &wcs);
    str = NULL;
    if (code) {
        ERROR("wcslib's wcspih() failed with code %i", code);
        return NULL;
    }

    if (nwcs > 1) {
        // copy the first entry, free the rest.
        wcs2 = calloc(1, sizeof(struct wcsprm));
        wcscopy(1, wcs, wcs2);
        wcsvfree(&nwcs, &wcs);
    } else {
        wcs2 = wcs;
    }
    code = wcsset(wcs2);
    if (code) {
        ERROR("wcslib's wcsset() failed with code %i: %s", code, wcs_errmsg[code]);
        return NULL;
    }

    anwcs = calloc(1, sizeof(anwcs_t));
    anwcs->type = ANWCS_TYPE_WCSLIB;
    anwcs->data = calloc(1, sizeof(anwcslib_t));
    anwcslib = anwcs->data;
    anwcslib->wcs = wcs2;
    anwcslib->imagew = W;
    anwcslib->imageh = H;

    return anwcs;
#endif
}


anwcs_t* anwcs_open_wcslib(const char* filename, int ext) {
#ifndef WCSLIB_EXISTS
    ERROR("Wcslib support was not compiled in");
    return NULL;
#else
    anwcs_t* anwcs = NULL;
    char* hdrstr;
    int Nhdr;
    hdrstr = getheader(filename, ext, &Nhdr);
    if (!hdrstr)
        return NULL;
    anwcs = anwcs_wcslib_from_string(hdrstr, Nhdr);
    free(hdrstr);
    if (!anwcs) {
        ERROR("Failed to parse FITS WCS header from file \"%s\" ext %i using WCSlib",
              filename, ext);
        return NULL;
    }
    return anwcs;
#endif
}

anwcs_t* anwcs_open_wcstools(const char* filename, int ext) {
#ifndef WCSTOOLS_EXISTS
    ERROR("WCStools support was not compiled in");
    return NULL;
#else
    anwcs_t* anwcs = NULL;
    char* hdrstr;
    int Nhdr;
    hdrstr = getheader(filename, ext, &Nhdr);
    if (!hdrstr)
        return NULL;
    anwcs = anwcs_wcstools_from_string(hdrstr, Nhdr);
    free(hdrstr);
    if (!anwcs) {
        ERROR("Failed to parse FITS WCS header from file \"%s\" ext %i using WCSTools",
              filename, ext);
        return NULL;
    }
    return anwcs;
#endif
}

anwcs_t* anwcs_wcstools_from_string(const char* str, int len) {
#ifndef WCSTOOLS_EXISTS
    ERROR("WCStools support was not compiled in");
    return NULL;
#else
    anwcs_t* anwcs = NULL;
    struct WorldCoor* wcs = wcsninit(str, len);
    if (!wcs) {
        ERROR("Failed to parse FITS WCS header using WCStools");
        // print to stderr.
        wcserr();
        return NULL;
    }
    anwcs = calloc(1, sizeof(anwcs_t));
    anwcs->type = ANWCS_TYPE_WCSTOOLS;
    anwcs->data = wcs;
    return anwcs;
#endif
}

int anwcs_radec2pixelxy(const anwcs_t* anwcs, double ra, double dec, double* px, double* py) {
    switch (anwcs->type) {
    case ANWCS_TYPE_WCSLIB:
#ifndef WCSLIB_EXISTS
	ERROR("Wcslib support was not compiled in");
	return -1;
#else
        {
            anwcslib_t* anwcslib = anwcs->data;
            return wcslib_radec2pixelxy(anwcslib, ra, dec, px, py);
        }
#endif
        break;

    case ANWCS_TYPE_SIP:
        {
            sip_t* sip;
            anbool ok;
            sip = anwcs->data;
            ok = sip_radec2pixelxy(sip, ra, dec, px, py);
            if (!ok)
                return -1;
        }
        break;

    default:
        ERROR("Unknown anwcs type %i", anwcs->type);
        return -1;
    }
    return 0;
}

anbool anwcs_find_discontinuity(const anwcs_t* wcs, double ra1, double dec1,
                                double ra2, double dec2,
                                double* pra3, double* pdec3,
                                double* pra4, double* pdec4) {
#ifdef WCSLIB_EXISTS
    if (wcs->type == ANWCS_TYPE_WCSLIB) {
        struct wcsprm* wcslib = ((anwcslib_t*)wcs->data)->wcs;
        if (ends_with(wcslib->ctype[0], "AIT")) {
            // Hammer-Aitoff -- wraps at 180 deg from CRVAL0
            double ra0 = fmod(wcslib->crval[0] + 180.0, 360.0);
            //printf("ra0 %g, ra1 %g, ra2 %g\n", ra0, ra1, ra2);
            double dr1 = fmod(fmod(ra1 - ra0, 360.) + 360., 360.);
            double dr2 = fmod(fmod(ra2 - ra0, 360.) + 360., 360.);
            //printf("dr1,dr2 %g, %g\n", dr1, dr2);
            if (fabs(dr1 - dr2) <
                MIN(fabs(360. + dr1 - dr2), fabs(360. + dr2 - dr1)))
                return FALSE;

            /*
             printf("d1 = %g, d2 = %g, d3 = %g, d4 = %g\n",
             fabs(ra1 - ra2), 360-fabs(ra1-ra2), 
             fabs(ra1-ra0), fabs(ra2-ra0));
             */
            // If ra1 to ra0 to ra2 is less than ra1 to ra2,, RA2 are closer wrapping-around than by crossing RA0.
            /*
             if (MIN(fabs(ra1 - ra2), 360-fabs(ra1-ra2)) <
             (fabs(ra1-ra0) + fabs(ra2-ra0)))
             return FALSE;
             */
            /*
             if (ra1
             // RA1, RA2 are on the same side of RA0
             if ((ra1 - ra0) * (ra2 - ra0) > 0) {
             return FALSE;
             }
             */
            if (pra3)
                *pra3 = ra0 + (ra1 > ra0 ? -360.0 : 0);
            if (pra4)
                *pra4 = ra0 + (ra2 > ra0 ? -360.0 : 0);
            if (pdec3 || pdec4) {
                // split the distance on sphere to find approximate Dec.
                //double fulldist = deg_between_radec(ra1, dec1, ra2, dec2);
                double dr1 = MIN(fabs(ra1 - ra0), fabs(ra1 - ra0 + 360));
                double dr2 = MIN(fabs(ra2 - ra0), fabs(ra2 - ra0 + 360));
                /*
                 logverb("ra0 = %g.  ra1=%g, dr1=%g;   ra2=%g, dr2=%g\n",
                 ra0, ra1, dr1, ra2, dr2);
                 */
                if (pdec3)
                    *pdec3 = dec1 + (dec2 - dec1) * dr1 / (dr1 + dr2);
                if (pdec4)
                    *pdec4 = dec1 + (dec2 - dec1) * dr1 / (dr1 + dr2);
            }
            return TRUE;
        }
    }
#endif
    /*
     if (anwcs_radec2pixelxy(wcs, ra1, dec1, &x1, &y1) ||
     anwcs_radec2pixelxy(wcs, ra2, dec2, &x2, &y2)) {
     return TRUE;
     }
     */
    return FALSE;
}


anbool anwcs_is_discontinuous(const anwcs_t* wcs, double ra1, double dec1,
                              double ra2, double dec2) {
    return anwcs_find_discontinuity(wcs, ra1, dec1, ra2, dec2,
                                    NULL, NULL, NULL, NULL);
}

/*
 static int anwcs_get_discontinuity(const anwcs_t* wcs,
 double ra1, double dec1,
 double ra2, double dec2,
 double* dra, double* ddec) {
 #ifdef WCSLIB_EXISTS
 if (wcs->type == ANWCS_TYPE_WCSLIB) {
 struct wcsprm* wcslib = ((anwcslib_t*)wcs->data)->wcs;
 if (ends_with(wcslib->ctype[0], "AIT")) {
 // Hammer-Aitoff -- wraps at 180 deg from CRVAL0
 double ra0 = fmod(wcslib->crval[0] + 180.0, 360.0);
			
 if ((ra1 - ra0) * (ra2 - ra0) < 0) {
 return TRUE;
 }
 }
 }
 #endif
 }
 */

// Walk from (ra1,dec1) toward (ra2,dec2) until you hit a boundary;
// then along the boundary until the boundary between (ra3,dec3) and
// (ra4,dec4), then to (ra3,dec3).
// 'stepsize' is in degrees.
dl* anwcs_walk_discontinuity(const anwcs_t* wcs,
                             double ra1, double dec1, double ra2, double dec2,
                             double ra3, double dec3, double ra4, double dec4,
                             double stepsize,
                             dl* radecs) {
    double xyz1[3], xyz2[3], xyz3[3], xyz4[4];
    double xyz[3];
    double dxyz[3];
    int i, j;
    double xyzstep;
    double ra=0,dec=0, lastra, lastdec;
    double raA,decA, raB,decB;
    double xyzA[3], xyzB[3];
    double dab;
    int NMAX;

    radecdeg2xyzarr(ra1, dec1, xyz1);
    radecdeg2xyzarr(ra2, dec2, xyz2);
    radecdeg2xyzarr(ra3, dec3, xyz3);
    radecdeg2xyzarr(ra4, dec4, xyz4);

    if (!radecs)
        radecs = dl_new(256);

    // first, from ra1,dec1 toward ra2,dec2.
    for (i=0; i<3; i++)
        dxyz[i] = xyz2[i] - xyz1[i];
    normalize_3(dxyz);
    xyzstep = deg2dist(stepsize);
    NMAX = ceil(2. / xyzstep);
    logverb("stepsize %g; nmax %i\n", xyzstep, NMAX);
    for (i=0; i<3; i++)
        dxyz[i] *= xyzstep;
    for (i=0; i<3; i++)
        xyz[i] = xyz1[i];

    dl_append(radecs, ra1);
    dl_append(radecs, dec1);
    lastra = ra1;
    lastdec = dec1;

    logverb("Walking from 1 to 2: RA,Decs %g,%g to %g,%g\n", ra1, dec1, ra2, dec2);
    for (j=0; j<NMAX; j++) {
        for (i=0; i<3; i++)
            xyz[i] += dxyz[i];
        normalize_3(xyz);
        xyzarr2radecdeg(xyz, &ra, &dec);
        logverb("  ra,dec %g,%g\n", ra, dec);
        if (anwcs_is_discontinuous(wcs, lastra, lastdec, ra, dec))
            break;
        dl_append(radecs, ra);
        dl_append(radecs, dec);
        lastra = ra;
        lastdec = dec;
    }
    if (j == NMAX)
        logverb("EXCEEDED number of steps\n");
		
    logverb("Hit boundary: %g,%g -- %g,%g\n", lastra, lastdec, ra, dec);
    raA = lastra;
    decA = lastdec;

    // Find the boundary between ra3,dec3 and ra4,dec4.
    for (i=0; i<3; i++)
        dxyz[i] = xyz4[i] - xyz3[i];
    normalize_3(dxyz);
    for (i=0; i<3; i++)
        dxyz[i] *= xyzstep;
    lastra = ra3;
    lastdec = dec3;
    for (i=0; i<3; i++)
        xyz[i] = xyz3[i];

    for (j=0; j<NMAX; j++) {
        for (i=0; i<3; i++)
            xyz[i] += dxyz[i];
        normalize_3(xyz);
        xyzarr2radecdeg(xyz, &ra, &dec);
        if (anwcs_is_discontinuous(wcs, lastra, lastdec, ra, dec))
            break;
        lastra = ra;
        lastdec = dec;
    }
    if (j == NMAX)
        logverb("EXCEEDED number of steps\n");
    logverb("Hit boundary at %g,%g\n", lastra, lastdec);
    raB = lastra;
    decB = lastdec;

    // Walk from A to B
    radecdeg2xyzarr(raA, decA, xyzA);
    radecdeg2xyzarr(raB, decB, xyzB);
    dab = distsq(xyzA, xyzB, 3);
    if (dab > 0) {
        for (i=0; i<3; i++)
            dxyz[i] = xyzB[i] - xyzA[i];
        normalize_3(dxyz);
        for (i=0; i<3; i++)
            dxyz[i] *= xyzstep;
        for (i=0; i<3; i++)
            xyz[i] = xyzA[i];
        for (j=0; j<NMAX; j++) {
            double d;
            for (i=0; i<3; i++)
                xyz[i] += dxyz[i];
            normalize_3(xyz);
            // did we walk past?
            d = distsq(xyzA, xyz, 3);
            if (d > dab)
                break;
            xyzarr2radecdeg(xyz, &ra, &dec);
            dl_append(radecs, ra);
            dl_append(radecs, dec);
        }
        if (j == NMAX)
            logverb("EXCEEDED number of steps\n");
        logverb("Walked along boundary A-B (to %g,%g)\n", ra, dec);
        dl_append(radecs, raB);
        dl_append(radecs, decB);
    }

    // Now from B to ra3,dec3.
    dab = distsq(xyzB, xyz3, 3);
    logverb("Walking B->3 (dist %g)\n", sqrt(dab));
    if (dab > 0) {
        for (i=0; i<3; i++)
            dxyz[i] = xyz3[i] - xyzB[i];
        normalize_3(dxyz);
        for (i=0; i<3; i++)
            dxyz[i] *= xyzstep;
        for (i=0; i<3; i++)
            xyz[i] = xyzB[i];
        for (j=0; j<NMAX; j++) {
            double d;
            for (i=0; i<3; i++)
                xyz[i] += dxyz[i];
            normalize_3(xyz);
            // did we walk past?
            d = distsq(xyzB, xyz, 3);
            // DEBUG
            xyzarr2radecdeg(xyz, &ra, &dec);
            logverb("  -> ra,dec %g,%g, dist %g, target %g\n", ra, dec, sqrt(d), sqrt(dab));
            if (d > dab)
                break;
            //xyzarr2radecdeg(xyz, &ra, &dec);
            dl_append(radecs, ra);
            dl_append(radecs, dec);
        }
        if (j == NMAX)
            logverb("EXCEEDED number of steps\n");
        logverb("Walk to %g,%g\n", ra, dec);
    }
    dl_append(radecs, ra3);
    dl_append(radecs, dec3);

    return radecs;
}

// Returns 0 if the whole line was traced without breaks.
// Otherwise, returns the index of the point on the far side of the
// break.
static int trace_line(const anwcs_t* wcs, const dl* rd,
                      int istart, int idir, int iend,
                      anbool firstmove, dl* plotrd) {
    int i;
    double lastra=0, lastdec=0;
    double first = TRUE;
    logverb("trace_line: %i to %i by %i\n", istart, iend, idir);
    for (i = istart; i != iend; i += idir) {
        double x,y,ra,dec;
        ra  = dl_get_const(rd, 2*i+0);
        dec = dl_get_const(rd, 2*i+1);
        logverb("tracing: i=%i, ra,dec = %g,%g\n", i, ra, dec);
        if (anwcs_radec2pixelxy(wcs, ra, dec, &x, &y))
            // ?
            continue;
        if (first) {
            logdebug("plot to (%.2f, %.2f)\n", ra, dec);
            dl_append(plotrd, x);
            dl_append(plotrd, y);
        } else {
            if (anwcs_is_discontinuous(wcs, lastra, lastdec, ra, dec)) {
                logverb("discont: (%.2f, %.2f) -- (%.2f, %.2f)\n",
                        lastra, lastdec, ra, dec);
                logverb("return %i\n", i);
                return i;
            } else {
                logverb("not discontinuous\n");
            }
            logdebug("plot to (%.2f, %.2f)\n", ra, dec);
            dl_append(plotrd, x);
            dl_append(plotrd, y);
        }
        lastra = ra;
        lastdec = dec;
        first = FALSE;
    }
    return 0;
}

pl* anwcs_walk_outline(const anwcs_t* wcs, const dl* rd, int fill) {
    pl* lists = pl_new(2);
    dl* rd2;
    int brk, end;
    double degstep = 0.0;
    int i;
    dl* plotrd = dl_new(256);

    end = dl_size(rd)/2;
    brk = trace_line(wcs, rd, 0, 1, end, TRUE, plotrd);
    logdebug("tracing line 1: brk=%i\n", brk);

    if (brk) {
        int brk2;
        int brk3;
        // back out the path.
        logdebug("Cancel path\n");
        dl_remove_all(plotrd);
        logdebug("trace segment 1 back to 0\n");
        // trace segment 1 backwards to 0
        brk2 = trace_line(wcs, rd, brk-1, -1, -1, TRUE, plotrd);
        logdebug("traced line 1 backwards: brk2=%i\n", brk2);
        assert(brk2 == 0);

        // trace segment 2: from end of list backward, until we
        // hit brk2 (worst case, we [should] hit brk)
        logdebug("trace segment 2: end back to brk2=%i\n", brk2);
        brk2 = trace_line(wcs, rd, end-1, -1, -1, FALSE, plotrd);
        logdebug("traced segment 2: brk2=%i\n", brk2);

        if (fill) {
            // trace segment 3: from brk2 to brk.
            // 1-pixel steps.
            logdebug("trace segment 3: brk2=%i to brk=%i\n", brk2, brk);
            logdebug("walking discontinuity: (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)\n",
                     dl_get_const(rd, 2*(brk2+1)+0), dl_get_const(rd, 2*(brk2+1)+1),
                     dl_get_const(rd, 2*(brk2  )+0), dl_get_const(rd, 2*(brk2  )+1),
                     dl_get_const(rd, 2*(brk -1)+0), dl_get_const(rd, 2*(brk -1)+1),
                     dl_get_const(rd, 2*(brk   )+0), dl_get_const(rd, 2*(brk   )+1));

            degstep = arcsec2deg(anwcs_pixel_scale(wcs));
            rd2 = anwcs_walk_discontinuity
                (wcs,
                 dl_get_const(rd, 2*(brk2+1)+0), dl_get_const(rd, 2*(brk2+1)+1),
                 dl_get_const(rd, 2*(brk2  )+0), dl_get_const(rd, 2*(brk2  )+1),
                 dl_get_const(rd, 2*(brk -1)+0), dl_get_const(rd, 2*(brk -1)+1),
                 dl_get_const(rd, 2*(brk   )+0), dl_get_const(rd, 2*(brk   )+1),
                 degstep, NULL);
            for (i=0; i<dl_size(rd2)/2; i++) {
                double x,y,ra,dec;
                ra  = dl_get(rd2, 2*i+0);
                dec = dl_get(rd2, 2*i+1);
                if (anwcs_radec2pixelxy(wcs, ra, dec, &x, &y))
                    // oops.
                    continue;
                logdebug("plot to (%.2f, %.2f)\n", ra, dec);
                dl_append(plotrd, x);
                dl_append(plotrd, y);
            }
            dl_free(rd2);
            logdebug("close_path\n");
        }

        // stroke
        pl_append(lists, plotrd);
        plotrd = dl_new(256);

        // trace segments 4+5: from brk to brk2.
        if (brk2 > brk) {
            logdebug("trace segments 4+5: from brk=%i to brk2=%i\n",brk,brk2);
            // (tracing the outline on the far side)
            brk3 = trace_line(wcs, rd, brk, 1, brk2, TRUE, plotrd);
            logdebug("traced segment 4/5: brk3=%i\n", brk3);
            assert(brk3 == 0);
            // trace segment 6: from brk2 to brk.
            // (walking the discontinuity on the far side)
            if (fill) {
                logdebug("walking discontinuity: (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)\n",
                         dl_get_const(rd, 2*(brk2  )+0), dl_get_const(rd, 2*(brk2  )+1),
                         dl_get_const(rd, 2*(brk2+1)+0), dl_get_const(rd, 2*(brk2+1)+1),
                         dl_get_const(rd, 2*(brk   )+0), dl_get_const(rd, 2*(brk   )+1),
                         dl_get_const(rd, 2*(brk -1)+0), dl_get_const(rd, 2*(brk -1)+1));
                rd2 = anwcs_walk_discontinuity
                    (wcs,
                     dl_get_const(rd, 2*(brk2  )+0), dl_get_const(rd, 2*(brk2  )+1),
                     dl_get_const(rd, 2*(brk2+1)+0), dl_get_const(rd, 2*(brk2+1)+1),
                     dl_get_const(rd, 2*(brk   )+0), dl_get_const(rd, 2*(brk   )+1),
                     dl_get_const(rd, 2*(brk -1)+0), dl_get_const(rd, 2*(brk -1)+1),
                     degstep, NULL);
                for (i=0; i<dl_size(rd2)/2; i++) {
                    double x,y,ra,dec;
                    ra  = dl_get(rd2, 2*i+0);
                    dec = dl_get(rd2, 2*i+1);
                    if (anwcs_radec2pixelxy(wcs, ra, dec, &x, &y))
                        // oops.
                        continue;
                    logdebug("plot to (%.2f, %.2f)\n", ra, dec);
                    dl_append(plotrd, x);
                    dl_append(plotrd, y);
                }
                dl_free(rd2);
                logdebug("close_path\n");
            }
        }
    }
    // stroke
    pl_append(lists, plotrd);
    return lists;
}



sip_t* anwcs_get_sip(const anwcs_t* wcs) {
    if (wcs->type == ANWCS_TYPE_SIP)
        return wcs->data;
    return NULL;
}

anwcs_t* anwcs_create_allsky_hammer_aitoff(double refra, double refdec,
                                           int W, int H) {
    return anwcs_create_hammer_aitoff(refra, refdec, 1.0, W, H, TRUE);
}

anwcs_t* anwcs_create_allsky_hammer_aitoff2(double refra, double refdec,
                                            int W, int H) {
    return anwcs_create_hammer_aitoff(refra, refdec, 1.0, W, H, FALSE);
}

static anwcs_t* allsky_wcs(double refra, double refdec,
                           double zoomfactor,
                           int W, int H, anbool yflip,
                           char* wcscode, char* wcsname,
                           anbool square_pixels,
                           char* ctype1, char* ctype2, double rotate) {
    qfits_header* hdr;
    double xscale = -360. / (double)W;
    double yscale;
    char* str = NULL;
    int Nstr = 0;
    anwcs_t* anwcs = NULL;
    char code[64];
    char ctype1str[5];
    char ctype2str[5];
    double cd11,cd12,cd21,cd22;

    if (square_pixels)
        yscale = xscale;
    else
        yscale =  180. / (double)H;

    if (yflip)
        yscale *= -1.;
    xscale /= zoomfactor;
    yscale /= zoomfactor;

    if (ctype1 == NULL)
        ctype1 = "RA";
    if (ctype2 == NULL)
        ctype2 = "DEC";

    memset(ctype1str, '\0', 5);
    memset(ctype2str, '\0', 5);
    strncpy(ctype1str, ctype1, 4);
    strncpy(ctype2str, ctype2, 4);
    for (int i=0; i<4; i++) {
        if (ctype1str[i] == '\0')
            ctype1str[i] = '-';
        if (ctype2str[i] == '\0')
            ctype2str[i] = '-';
    }

    if (rotate == 0.0) {
        cd11 = xscale;
        cd12 = cd21 = 0.;
        cd22 = yscale;
    } else {
        double r = deg2rad(rotate);
        double cr = cos(r);
        double sr = sin(r);
        cd11 = xscale *  cr;
        cd12 = xscale *  sr;
        cd21 = yscale * -sr;
        cd22 = yscale *  cr;
    }
    
    hdr = qfits_header_default();
    //sprintf(code, "RA---%s", wcscode);
    sprintf(code, "%s-%s", ctype1str, wcscode);
    qfits_header_add(hdr, "CTYPE1", code, wcsname, NULL);
    //sprintf(code, "DEC--%s", wcscode);
    sprintf(code, "%s-%s", ctype2str, wcscode);
    qfits_header_add(hdr, "CTYPE2", code, wcsname, NULL);
    fits_header_add_double(hdr, "CRPIX1", W/2. + 0.5, NULL);
    fits_header_add_double(hdr, "CRPIX2", H/2. + 0.5, NULL);
    fits_header_add_double(hdr, "CRVAL1", refra,  NULL);
    fits_header_add_double(hdr, "CRVAL2", refdec, NULL);
    fits_header_add_double(hdr, "CD1_1", cd11, NULL);
    fits_header_add_double(hdr, "CD1_2", cd12, NULL);
    fits_header_add_double(hdr, "CD2_1", cd21, NULL);
    fits_header_add_double(hdr, "CD2_2", cd22, NULL);
    fits_header_add_int(hdr, "IMAGEW", W, NULL);
    fits_header_add_int(hdr, "IMAGEH", H, NULL);

    str = fits_to_string(hdr, &Nstr);
    qfits_header_destroy(hdr);
    if (!str) {
        ERROR("Failed to write %s FITS header as string", wcsname);
        return NULL;
    }
    anwcs = anwcs_wcslib_from_string(str, Nstr);
    free(str);
    if (!anwcs) {
        ERROR("Failed to parse %s header string with wcslib", wcsname);
        return NULL;
    }
    return anwcs;
}

anwcs_t* anwcs_create_mollweide(double refra, double refdec,
                                double zoomfactor,
                                int W, int H, anbool yflip) {
    return allsky_wcs(refra, refdec, zoomfactor, W, H, yflip,
                      "MOL", "Mollweide", FALSE, NULL, NULL, 0.);
}

anwcs_t* anwcs_create_hammer_aitoff(double refra, double refdec,
                                    double zoomfactor,
                                    int W, int H, anbool yflip) {
    return allsky_wcs(refra, refdec, zoomfactor, W, H, yflip,
                      "AIT", "Hammer-Aitoff", FALSE, NULL, NULL, 0.);
}

anwcs_t* anwcs_create_hammer_aitoff_rectangular(double refra, double refdec,
                                                double zoomfactor, double rotate,
                                                int W, int H, anbool yflip) {
    return allsky_wcs(refra, refdec, zoomfactor, W, H, yflip,
                      "AIT", "Hammer-Aitoff", TRUE, NULL, NULL, rotate);
}

anwcs_t* anwcs_create_hammer_aitoff_galactic(double ref_long, double ref_lat,
                                             double zoomfactor,
                                             int W, int H, anbool yflip) {
    return allsky_wcs(ref_long, ref_lat, zoomfactor, W, H, yflip,
                      "AIT", "Hammer-Aitoff", TRUE, "GLON", "GLAT", 0.);
}

anwcs_t* anwcs_create_cea_wcs(double refra, double refdec,
                              double refx, double refy,
                              double pixscale,
                              int W, int H, anbool yflip) {
    qfits_header* hdr;
    char* str = NULL;
    int Nstr = 0;
    anwcs_t* anwcs = NULL;
    char code[64];
    const char* wcsname = "Cylindrical equal-area";
    const char* wcscode = "CEA";
    hdr = qfits_header_default();
    sprintf(code, "RA---%s", wcscode);
    qfits_header_add(hdr, "CTYPE1", code, wcsname, NULL);
    sprintf(code, "DEC--%s", wcscode);
    qfits_header_add(hdr, "CTYPE2", code, wcsname, NULL);
    //fits_header_add_double(hdr, "CRPIX1", W/2. + 0.5, NULL);
    //fits_header_add_double(hdr, "CRPIX2", H/2. + 0.5, NULL);
    fits_header_add_double(hdr, "CRPIX1", refx, NULL);
    fits_header_add_double(hdr, "CRPIX2", refy, NULL);
    fits_header_add_double(hdr, "CRVAL1", refra,  NULL);
    fits_header_add_double(hdr, "CRVAL2", refdec, NULL);
    fits_header_add_double(hdr, "CD1_1", -pixscale, NULL);
    fits_header_add_double(hdr, "CD1_2", 0, NULL);
    fits_header_add_double(hdr, "CD2_1", 0, NULL);
    fits_header_add_double(hdr, "CD2_2", pixscale * (yflip ? -1 : 1), NULL);
    fits_header_add_int(hdr, "IMAGEW", W, NULL);
    fits_header_add_int(hdr, "IMAGEH", H, NULL);
    //fits_header_add_double(hdr, "LATPOLE", 0., NULL);
    //fits_header_add_double(hdr, "LONPOLE", 0., NULL);

    str = fits_to_string(hdr, &Nstr);
    qfits_header_destroy(hdr);
    if (!str) {
        ERROR("Failed to write %s FITS header as string", wcsname);
        return NULL;
    }
    anwcs = anwcs_wcslib_from_string(str, Nstr);
    free(str);
    if (!anwcs) {
        ERROR("Failed to parse %s header string with wcslib", wcsname);
        return NULL;
    }
    return anwcs;
}

#ifndef WCSLIB_EXISTS
#define WCSLIB_HAS_WCSCCS 0
#endif

int anwcs_galactic_to_radec(anwcs_t* wcs) {
    if (!wcs) return -1;
    if (wcs->type != ANWCS_TYPE_WCSLIB) {
        ERROR("anwcs_galactic_to_radec is only implemented for WCSlib.");
        return -1;
    }
#if WCSLIB_HAS_WCSCCS
    // Convert from Galactic to RA,Dec basis
    int rtn;
    anwcslib_t* anwcslib = wcs->data;
    rtn = wcsccs(anwcslib->wcs, 192.8595, 27.1283, 122.9319,
                 "RA", "DEC", "J2000", 2000.0, "");
    if (rtn) {
        ERROR("Failed to convert coordinate system with wcsccs()");
        return rtn;
    }
    return 0;
#else
    ERROR("WCSLib >= v7.5 is required for anwcs_create_galactic_car_wcs");
    return -1;
#endif
}

anwcs_t* anwcs_create_galactic_car_wcs(double refra, double refdec,
                                       double refx, double refy,
                                       double pixscale,
                                       int W, int H, anbool yflip) {
    qfits_header* hdr;
    char* str = NULL;
    int Nstr = 0;
    anwcs_t* anwcs = NULL;
    char code[64];
    const char* wcsname = "Plate Carree";
    const char* wcscode = "CAR";
    hdr = qfits_header_default();
    sprintf(code, "GLON-%s", wcscode);
    qfits_header_add(hdr, "CTYPE1", code, wcsname, NULL);
    sprintf(code, "GLAT-%s", wcscode);
    qfits_header_add(hdr, "CTYPE2", code, wcsname, NULL);
    fits_header_add_double(hdr, "CRPIX1", refx, NULL);
    fits_header_add_double(hdr, "CRPIX2", refy, NULL);
    fits_header_add_double(hdr, "CRVAL1", refra,  NULL);
    fits_header_add_double(hdr, "CRVAL2", refdec, NULL);
    fits_header_add_double(hdr, "CD1_1", -pixscale, NULL);
    fits_header_add_double(hdr, "CD1_2", 0, NULL);
    fits_header_add_double(hdr, "CD2_1", 0, NULL);
    fits_header_add_double(hdr, "CD2_2", pixscale * (yflip ? -1 : 1), NULL);
    fits_header_add_int(hdr, "IMAGEW", W, NULL);
    fits_header_add_int(hdr, "IMAGEH", H, NULL);
    //fits_header_add_double(hdr, "LATPOLE", 0., NULL);
    //fits_header_add_double(hdr, "LONPOLE", 0., NULL);
    str = fits_to_string(hdr, &Nstr);
    qfits_header_destroy(hdr);
    if (!str) {
        ERROR("Failed to write %s FITS header as string", wcsname);
        return NULL;
    }
    anwcs = anwcs_wcslib_from_string(str, Nstr);
    free(str);
    if (!anwcs) {
        ERROR("Failed to parse %s header string with wcslib", wcsname);
        return NULL;
    }
    return anwcs;
}

anwcs_t* anwcs_create_mercator_2(double refra, double refdec,
                                 double crpix1, double crpix2,
                                 double zoomfactor,
                                 int W, int H, anbool yflip) {
    qfits_header* hdr;
    double xscale = -360. / (double)W;
    double yscale = -xscale;
    // Pixel scales are the same...
    char* str = NULL;
    int Nstr = 0;
    anwcs_t* anwcs = NULL;

    if (yflip)
        yscale *= -1.;

    xscale /= zoomfactor;
    yscale /= zoomfactor;

    hdr = qfits_header_default();
    qfits_header_add(hdr, "CTYPE1", "RA---MER", "Mercator", NULL);
    qfits_header_add(hdr, "CTYPE2", "DEC--MER", "Mercator", NULL);
    fits_header_add_double(hdr, "CRPIX1", crpix1, NULL);
    fits_header_add_double(hdr, "CRPIX2", crpix2, NULL);
    fits_header_add_double(hdr, "CRVAL1", refra,  NULL);
    fits_header_add_double(hdr, "CRVAL2", refdec, NULL);
    fits_header_add_double(hdr, "CD1_1", xscale, NULL);
    fits_header_add_double(hdr, "CD1_2", 0, NULL);
    fits_header_add_double(hdr, "CD2_1", 0, NULL);
    fits_header_add_double(hdr, "CD2_2", yscale, NULL);
    fits_header_add_int(hdr, "IMAGEW", W, NULL);
    fits_header_add_int(hdr, "IMAGEH", H, NULL);

    str = fits_to_string(hdr, &Nstr);
    qfits_header_destroy(hdr);
    if (!str) {
        ERROR("Failed to write Mercator FITS header as string");
        return NULL;
    }

    anwcs = anwcs_wcslib_from_string(str, Nstr);
    free(str);
    if (!anwcs) {
        ERROR("Failed to parse Mercator header string with wcslib");
        return NULL;
    }
    return anwcs;
}

anwcs_t* anwcs_create_mercator(double refra, double refdec,
                               double zoomfactor,
                               int W, int H, anbool yflip) {
    return anwcs_create_mercator_2(refra, refdec,
                                   W/2 + 0.5, H/2 + 0.5,
                                   zoomfactor, W, H, yflip);
}





static anwcs_t*
anwcs_create_box_scaled(double ra, double dec, double width, int W, int H,
                        double yscale) {
    tan_t tan;
    double scale;
    tan.crval[0] = ra;
    tan.crval[1] = dec;
    tan.crpix[0] = W / 2.0 + 0.5;
    tan.crpix[1] = H / 2.0 + 0.5;
    scale = width / (double)W;
    tan.cd[0][0] = -scale;
    tan.cd[1][0] = 0;
    tan.cd[0][1] = 0;
    tan.cd[1][1] = scale * yscale;
    tan.imagew = W;
    tan.imageh = H;
    return anwcs_new_tan(&tan);
}

anwcs_t* anwcs_create_box(double ra, double dec, double width, int W, int H) {
    return anwcs_create_box_scaled(ra, dec, width, W, H, 1);
}

anwcs_t* anwcs_create_box_upsidedown(double ra, double dec, double width, int W, int H) {
    return anwcs_create_box_scaled(ra, dec, width, W, H, -1);
}


