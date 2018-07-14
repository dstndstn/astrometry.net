/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include <math.h>

#include "coadd.h"

#include "anwcs.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "mathutil.h"
#include "anqfits.h"
#include "keywords.h"
#include "tic.h"
#include "convolve-image.h"
#include "ioutils.h"
#include "resample.h"

static const char* OPTIONS = "hvw:o:e:O:Ns:p:D";

void printHelp(char* progname) {
    fprintf(stderr, "%s [options] <input-FITS-image> <image-ext> <input-weight (filename or constant)> <weight-ext> <input-WCS> <wcs-ext> \n          [<image> <ext> <weight> <ext> <wcs> <ext>...]\n"
            "  use \"none\" as weight filename for no weighting\n"
            "     -w <output-wcs-file>  (default: input file)\n"
            "    [-e <output-wcs-ext>]: FITS extension to read WCS from (default: primary extension, 0)\n"
            "     -o <output-image-file>\n"
            "    [-O <order>]: Lanczos order (default 3)\n"
            "    [-p <plane>]: image plane to read (default 0)\n"
            "    [-N]: use nearest-neighbour resampling (default: Lanczos)\n"
            "    [-s <sigma>]: smooth before resampling\n"
            "    [-D]: divide each image by its weight image before starting\n"
            "    [-v]: more verbose\n"
            "\n", progname);
}



int main(int argc, char** args) {
    int argchar;
    char* progname = args[0];

    char* outfn = NULL;
    char* outwcsfn = NULL;
    int outwcsext = 0;

    anwcs_t* outwcs;

    sl* inimgfns = sl_new(16);
    sl* inwcsfns = sl_new(16);
    sl* inwtfns = sl_new(16);
    il* inimgexts = il_new(16);
    il* inwcsexts = il_new(16);
    il* inwtexts = il_new(16);

    int i;
    int loglvl = LOG_MSG;
    int order = 3;

    coadd_t* coadd;
    lanczos_args_t largs;

    double sigma = 0.0;
    anbool nearest = FALSE;
    anbool divweight = FALSE;

    int plane = 0;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case '?':
        case 'h':
            printHelp(progname);
            exit(0);
        case 'D':
            divweight = TRUE;
            break;
        case 'p':
            plane = atoi(optarg);
            break;
        case 'N':
            nearest = TRUE;
            break;
        case 's':
            sigma = atof(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case 'e':
            outwcsext = atoi(optarg);
            break;
        case 'w':
            outwcsfn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case 'O':
            order = atoi(optarg);
            break;
        }

    log_init(loglvl);
    fits_use_error_system();

    args += optind;
    argc -= optind;
    if (argc == 0 || argc % 6) {
        printHelp(progname);
        exit(-1);
    }

    for (i=0; i<argc/6; i++) {
        sl_append(inimgfns, args[6*i+0]);
        il_append(inimgexts, atoi(args[6*i+1]));
        sl_append(inwtfns, args[6*i+2]);
        il_append(inwtexts, atoi(args[6*i+3]));
        sl_append(inwcsfns, args[6*i+4]);
        il_append(inwcsexts, atoi(args[6*i+5]));
    }

    logmsg("Reading output WCS file %s\n", outwcsfn);
    outwcs = anwcs_open(outwcsfn, outwcsext);
    if (!outwcs) {
        ERROR("Failed to read WCS from file: %s ext %i\n", outwcsfn, outwcsext);
        exit(-1);
    }

    logmsg("Output image will be %i x %i\n", (int)anwcs_imagew(outwcs), (int)anwcs_imageh(outwcs));

    coadd = coadd_new(anwcs_imagew(outwcs), anwcs_imageh(outwcs));

    coadd->wcs = outwcs;

    if (nearest) {
        coadd->resample_func = nearest_resample_f;
        coadd->resample_token = NULL;
    } else {
        coadd->resample_func = lanczos_resample_f;
        largs.order = order;
        coadd->resample_token = &largs;
    }

    for (i=0; i<sl_size(inimgfns); i++) {
        anqfits_t* anq;
        anqfits_t* wanq;
        float* img;
        float* wt = NULL;
        anwcs_t* inwcs;
        char* fn;
        int ext;
        float overallwt = 1.0;
        int W, H;

        fn = sl_get(inimgfns, i);
        ext = il_get(inimgexts, i);
        logmsg("Reading input image \"%s\" ext %i\n", fn, ext);

        anq = anqfits_open(fn);
        if (!anq) {
            ERROR("Failed to open file \"%s\"\n", fn);
            exit(-1);
        }

        img = anqfits_readpix(anq, ext, 0, 0, 0, 0, plane,
                              PTYPE_FLOAT, NULL, &W, &H);
        if (!img) {
            ERROR("Failed to read image from ext %i of %s\n", ext, fn);
            exit(-1);
        }
        anqfits_close(anq);
        logmsg("Read image: %i x %i.\n", W, H);

        if (sigma > 0.0) {
            int k0, nk;
            float* kernel;
            logmsg("Smoothing by Gaussian with sigma=%g\n", sigma);
            kernel = convolve_get_gaussian_kernel_f(sigma, 4, &k0, &nk);
            convolve_separable_f(img, W, H, kernel, k0, nk, img, NULL);
            free(kernel);
        }

        fn = sl_get(inwcsfns, i);
        ext = il_get(inwcsexts, i);
        logmsg("Reading input WCS file \"%s\" ext %i\n", fn, ext);

        inwcs = anwcs_open(fn, ext);
        if (!inwcs) {
            ERROR("Failed to read WCS from file \"%s\" ext %i\n", fn, ext);
            exit(-1);
        }
        if (anwcs_pixel_scale(inwcs) == 0) {
            ERROR("Pixel scale from the WCS file is zero.  Usually this means the image has no valid WCS header.\n");
            exit(-1);
        }
        if (anwcs_imagew(inwcs) != W || anwcs_imageh(inwcs) != H) {
            ERROR("Size mismatch between image and WCS!");
            exit(-1);
        }

        fn = sl_get(inwtfns, i);
        ext = il_get(inwtexts, i);
        if (streq(fn, "none")) {
            logmsg("Not using weight image.\n");
            wt = NULL;
        } else if (file_exists(fn)) {
            logmsg("Reading input weight image \"%s\" ext %i\n", fn, ext);
            wanq = anqfits_open(fn);
            if (!wanq) {
                ERROR("Failed to open file \"%s\"\n", fn);
                exit(-1);
            }
            int wtW, wtH;
            wt = anqfits_readpix(anq, ext, 0, 0, 0, 0, 0,
                                 PTYPE_FLOAT, NULL, &wtW, &wtH);
            if (!wt) {
                ERROR("Failed to read image from ext %i of %s\n", ext, fn);
                exit(-1);
            }
            anqfits_close(wanq);
            logmsg("Read image: %i x %i.\n", wtW, wtH);
            if (wtW != W || wtH != H) {
                ERROR("Size mismatch between image and weight!");
                exit(-1);
            }
        } else {
            char* endp;
            overallwt = strtod(fn, &endp);
            if (endp == fn) {
                ERROR("Weight: \"%s\" is neither a file nor a double.\n", fn);
                exit(-1);
            }
            logmsg("Parsed weight value \"%g\"\n", overallwt);
        }

        if (divweight && wt) {
            int j;
            logmsg("Dividing image by weight image...\n");
            for (j=0; j<(W*H); j++)
                img[j] /= wt[j];
        }

        coadd_add_image(coadd, img, wt, overallwt, inwcs);

        anwcs_free(inwcs);
        free(img);
        if (wt)
            free(wt);
    }

    //
    logmsg("Writing output: %s\n", outfn);

    coadd_divide_by_weight(coadd, 0.0);

    /*
     if (fits_write_float_image_hdr(coadd->img, coadd->W, coadd->H, outfn)) {
     ERROR("Failed to write output image %s", outfn);
     exit(-1);
     }
     */
    /*
     if (fits_write_float_image(coadd->img, coadd->W, coadd->H, outfn)) {
     ERROR("Failed to write output image %s", outfn);
     exit(-1);
     }
     */
    {
        qfitsdumper qoutimg;
        qfits_header* hdr;
        hdr = anqfits_get_header2(outwcsfn, outwcsext);
        if (!hdr) {
            ERROR("Failed to read WCS file \"%s\" ext %i\n", outwcsfn, outwcsext);
            exit(-1);
        }
        fits_header_mod_int(hdr, "NAXIS", 2, NULL);
        fits_header_set_int(hdr, "NAXIS1", coadd->W, "image width");
        fits_header_set_int(hdr, "NAXIS2", coadd->H, "image height");
        fits_header_modf(hdr, "BITPIX", "-32", "32-bit floats");
        memset(&qoutimg, 0, sizeof(qoutimg));
        qoutimg.filename = outfn;
        qoutimg.npix = coadd->W * coadd->H;
        qoutimg.fbuf = coadd->img;
        qoutimg.ptype = PTYPE_FLOAT;
        qoutimg.out_ptype = BPP_IEEE_FLOAT;
        if (fits_write_header_and_image(NULL, &qoutimg, coadd->W)) {
            ERROR("Failed to write FITS image to file \"%s\"", outfn);
            exit(-1);
        }
        qfits_header_destroy(hdr);
    }

    coadd_free(coadd);
    sl_free2(inimgfns);
    sl_free2(inwcsfns);
    sl_free2(inwtfns);
    il_free(inimgexts);
    il_free(inwcsexts);
    il_free(inwtexts);
    anwcs_free(outwcs);


    return 0;
}


