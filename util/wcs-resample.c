/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "wcs-resample.h"
#include "sip_qfits.h"
#include "an-bool.h"
#include "anqfits.h"
#include "starutil.h"
#include "bl.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"
#include "anwcs.h"
#include "resample.h"
#include "mathutil.h"

int resample_wcs_files(const char* infitsfn, int infitsext,
                       const char* inwcsfn, int inwcsext,
                       const char* outwcsfn, int outwcsext,
                       const char* outfitsfn, int lorder,
                       int zero_inf) {

    anwcs_t* inwcs;
    anwcs_t* outwcs;
    anqfits_t* anqin;
    qfitsdumper qoutimg;
    float* inimg;
    float* outimg;
    qfits_header* hdr;

    int outW, outH;
    int inW, inH;

    double outpixmin, outpixmax;

    // read input WCS.
    inwcs = anwcs_open(inwcsfn, inwcsext);
    if (!inwcs) {
        ERROR("Failed to parse WCS header from %s extension %i", inwcsfn, inwcsext);
        return -1;
    }

    logmsg("Read input WCS from file \"%s\" ext %i\n", inwcsfn, inwcsext);
    anwcs_print(inwcs, stdout);

    // read output WCS.
    outwcs = anwcs_open(outwcsfn, outwcsext);
    if (!outwcs) {
        ERROR("Failed to parse WCS header from %s extension %i", outwcsfn, outwcsext);
        return -1;
    }

    logmsg("Read output (target) WCS from file \"%s\" ext %i\n", outwcsfn, outwcsext);
    anwcs_print(outwcs, stdout);

    outW = anwcs_imagew(outwcs);
    outH = anwcs_imageh(outwcs);

    // read input image.
    anqin = anqfits_open(infitsfn);
    if (!anqin) {
        ERROR("Failed to open \"%s\"", infitsfn);
        return -1;
    }
    inimg = (float*)anqfits_readpix(anqin, infitsext, 0, 0, 0, 0, 0,
                                    PTYPE_FLOAT, NULL, &inW, &inH);
    anqfits_close(anqin);
    anqin = NULL;
    if (!inimg) {
        ERROR("Failed to read pixels from \"%s\"", infitsfn);
        return -1;
    }

    if (zero_inf) {
        int i;
        for (i=0; i<(inW*inH); i++) {
            if (!isfinite(inimg[i])) {
                inimg[i] = 0.0;
            }
        }
    }

    logmsg("Input  image is %i x %i pixels.\n", inW, inH);
    logmsg("Output image is %i x %i pixels.\n", outW, outH);

    outimg = calloc((size_t)outW * (size_t)outH, sizeof(float));

    if (resample_wcs(inwcs, inimg, inW, inH,
                     outwcs, outimg, outW, outH, 1, lorder)) {
        ERROR("Failed to resample");
        return -1;
    }

    {
        double pmin, pmax;
        int i;
        /*
         pmin =  LARGE_VAL;
         pmax = -LARGE_VAL;
         for (i=0; i<(inW*inH); i++) {
         pmin = MIN(pmin, inimg[i]);
         pmax = MAX(pmax, inimg[i]);
         }
         logmsg("Input image bounds: %g to %g\n", pmin, pmax);
         */
        pmin =  LARGE_VAL;
        pmax = -LARGE_VAL;
        for (i=0; i<(outW*outH); i++) {
            pmin = MIN(pmin, outimg[i]);
            pmax = MAX(pmax, outimg[i]);
        }
        logmsg("Output image bounds: %g to %g\n", pmin, pmax);
        outpixmin = pmin;
        outpixmax = pmax;
    }

    // prepare output image.
    memset(&qoutimg, 0, sizeof(qoutimg));
    qoutimg.filename = outfitsfn;
    qoutimg.npix = outW * outH;
    qoutimg.ptype = PTYPE_FLOAT;
    qoutimg.fbuf = outimg;
    qoutimg.out_ptype = BPP_IEEE_FLOAT;

    hdr = fits_get_header_for_image(&qoutimg, outW, NULL);
    anwcs_add_to_header(outwcs, hdr);
    fits_header_add_double(hdr, "DATAMIN", outpixmin, "min pixel value");
    fits_header_add_double(hdr, "DATAMAX", outpixmax, "max pixel value");

    if (fits_write_header_and_image(hdr, &qoutimg, 0)) {
        ERROR("Failed to write image to file \"%s\"", outfitsfn);
        return -1;
    }
    free(outimg);
    qfits_header_destroy(hdr);

    anwcs_free(inwcs);
    anwcs_free(outwcs);

    return 0;
}

// Check whether output pixels overlap with input pixels,
// on a grid of output pixel positions.
static anbool* find_overlap_grid(int B, int outW, int outH,
                                 const anwcs_t* outwcs, const anwcs_t* inwcs,
                                 int* pBW, int* pBH) {
    int BW, BH;
    anbool* bib = NULL;
    anbool* bib2 = NULL;
    int i,j;

    BW = (int)ceil(outW / (float)B);
    BH = (int)ceil(outH / (float)B);
    bib = calloc((size_t)BW*(size_t)BH, sizeof(anbool));
    for (i=0; i<BH; i++) {
        for (j=0; j<BW; j++) {
            int x,y;
            double ra,dec;
            y = MIN(outH-1, B*i);
            x = MIN(outW-1, B*j);
            if (anwcs_pixelxy2radec(outwcs, x+1, y+1, &ra, &dec))
                continue;
            bib[i*BW+j] = anwcs_radec_is_inside_image(inwcs, ra, dec);
        }
    }
    if (log_get_level() >= LOG_VERB) {
        logverb("Input image overlaps output image:\n");
        for (i=0; i<BH; i++) {
            for (j=0; j<BW; j++)
                logverb((bib[i*BW+j]) ? "*" : ".");
            logverb("\n");
        }
    }
    // Grow the in-bounds area:
    bib2 = calloc((size_t)BW*(size_t)BH, sizeof(anbool));
    for (i=0; i<BH; i++)
        for (j=0; j<BW; j++) {
            int di,dj;
            if (!bib[i*BW+j])
                continue;
            for (di=-1; di<=1; di++)
                for (dj=-1; dj<=1; dj++)
                    bib2[(MIN(MAX(i+di, 0), BH-1))*BW + (MIN(MAX(j+dj, 0), BW-1))] = TRUE;
        }
    // swap!
    free(bib);
    bib = bib2;
    bib2 = NULL;

    if (log_get_level() >= LOG_VERB) {
        logverb("After growing:\n");
        for (i=0; i<BH; i++) {
            for (j=0; j<BW; j++)
                logverb((bib[i*BW+j]) ? "*" : ".");
            logverb("\n");
        }
    }

    *pBW = BW;
    *pBH = BH;
    return bib;
}



int resample_wcs(const anwcs_t* inwcs, const float* inimg, int inW, int inH,
                 const anwcs_t* outwcs, float* outimg, int outW, int outH,
                 int weighted, int lorder) {
    int i,j;
    int jlo,jhi,ilo,ihi;
    lanczos_args_t largs;
    double xyz[3];
    memset(&largs, 0, sizeof(largs));
    largs.order = lorder;
    largs.weighted = weighted;

    jlo = ilo = 0;
    ihi = outW;
    jhi = outH;

    // find the center of "outwcs", and describe the boundary as a
    // polygon in ~IWC coordinates.  Then find the extent of "inwcs".

    {
        int ok = 1;
        double xmin, xmax, ymin, ymax;
        int x, y, W, H;
        double xx,yy;
        xmin = ymin = LARGE_VAL;
        xmax = ymax = -LARGE_VAL;
        // HACK -- just look at the corners.  Could anwcs_walk_boundary.
        W = anwcs_imagew(inwcs);
        H = anwcs_imageh(inwcs);
        for (x=0; x<2; x++) {
            for (y=0; y<2; y++) {
                if (anwcs_pixelxy2xyz(inwcs, 1+x * (W-1), 1+y * (H-1), xyz) ||
                    anwcs_xyz2pixelxy(outwcs, xyz, &xx, &yy)) {
                    ok = 0;
                    break;
                }
                xmin = MIN(xmin, xx);
                xmax = MAX(xmax, xx);
                ymin = MIN(ymin, yy);
                ymax = MAX(ymax, yy);
            }
            if (!ok)
                break;
        }
        if (ok) {
            W = anwcs_imagew(outwcs);
            H = anwcs_imageh(outwcs);
            if ((xmin >= W) || (xmax < 0) ||
                (ymin >= H) || (ymax < 0)) {
                logverb("No overlap between input and output images\n");
                return 0;
            }
            ilo = MAX(0, xmin);
            ihi = MIN(W, xmax);
            jlo = MAX(0, ymin);
            jhi = MIN(H, ymax);
            //logverb("Clipped resampling output box to [%i,%i], [%i,%i]\n",
            //ilo,ihi,jlo,jhi);
        }
    }

    for (j=jlo; j<jhi; j++) {
        for (i=ilo; i<ihi; i++) {
            double inx, iny;
            float pix;
            // +1 for FITS pixel coordinates.
            if (anwcs_pixelxy2xyz(outwcs, i+1, j+1, xyz) ||
                anwcs_xyz2pixelxy(inwcs, xyz, &inx, &iny))
                continue;

            inx -= 1.0;
            iny -= 1.0;

            if (lorder == 0) {
                int x,y;
                // Nearest-neighbour resampling
                x = round(inx);
                y = round(iny);
                if (x < 0 || x >= inW || y < 0 || y >= inH)
                    continue;
                pix = inimg[y * inW + x];
            } else {
                if (inx < (-lorder) || inx >= (inW+lorder) ||
                    iny < (-lorder) || iny >= (inH+lorder))
                    continue;
                pix = lanczos_resample_unw_sep_f(inx, iny, inimg, inW, inH, &largs);
            }
            outimg[j * outW + i] = pix;
        }
    }
    return 0;
}


int resample_wcs_rgba(const anwcs_t* inwcs, const unsigned char* inimg,
                      int inW, int inH,
                      const anwcs_t* outwcs, unsigned char* outimg,
                      int outW, int outH) {
    int i,j;
    int B = 20;
    int BW, BH;
    anbool* bib;
    int bi,bj;

    bib = find_overlap_grid(B, outW, outH, outwcs, inwcs, &BW, &BH);

    // We've expanded the in-bounds boxes by 1 in each direction,
    // so this (using the lower-left corner) should be ok.
    for (bj=0; bj<BH; bj++) {
        for (bi=0; bi<BW; bi++) {
            int jlo,jhi,ilo,ihi;
            if (!bib[bj*BW + bi])
                continue;
            jlo = MIN(outH,  bj   *B);
            jhi = MIN(outH, (bj+1)*B);
            ilo = MIN(outW,  bi   *B);
            ihi = MIN(outW, (bi+1)*B);
            for (j=jlo; j<jhi; j++) {
                for (i=ilo; i<ihi; i++) {
                    double xyz[3];
                    double inx, iny;
                    int x,y;
                    // +1 for FITS pixel coordinates.
                    if (anwcs_pixelxy2xyz(outwcs, i+1, j+1, xyz) ||
                        anwcs_xyz2pixelxy(inwcs, xyz, &inx, &iny))
                        continue;
                    // FIXME - Nearest-neighbour resampling!!
                    // -1 for FITS pixel coordinates.
                    x = round(inx - 1.0);
                    y = round(iny - 1.0);
                    if (x < 0 || x >= inW || y < 0 || y >= inH)
                        continue;
                    // HACK -- straight copy
                    outimg[4 * (j * outW + i) + 0] = inimg[4 * (y * inW + x) + 0];
                    outimg[4 * (j * outW + i) + 1] = inimg[4 * (y * inW + x) + 1];
                    outimg[4 * (j * outW + i) + 2] = inimg[4 * (y * inW + x) + 2];
                    outimg[4 * (j * outW + i) + 3] = inimg[4 * (y * inW + x) + 3];
                }
            }
        }
    }

    free(bib);

    return 0;

}

