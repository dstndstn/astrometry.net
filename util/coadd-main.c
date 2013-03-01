#include <sys/param.h>
#include <stdio.h>
#include <math.h>

#include "coadd.h"

#include "anwcs.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "mathutil.h"
#include "qfits_image.h"
#include "keywords.h"
#include "tic.h"
#include "convolve-image.h"
#include "ioutils.h"

static const char* OPTIONS = "hvw:o:e:O:Ns:p:D";

void printHelp(char* progname) {
    fprintf(stderr, "%s [options] <input-FITS-image> <image-ext> <input-weight (filename or constant)> <weight-ext> <input-WCS> <wcs-ext> [<image> <ext> <weight> <ext> <wcs> <ext>...]\n"
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
extern char *optarg;
extern int optind, opterr, optopt;



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
		coadd->resample_func = nearest_resample;
		coadd->resample_token = NULL;
	} else {
		coadd->resample_func = lanczos_resample;
		largs.order = order;
		coadd->resample_token = &largs;
	}

	for (i=0; i<sl_size(inimgfns); i++) {
		qfitsloader ld;
		qfitsloader wld;
		float* img;
		float* wt = NULL;
		anwcs_t* inwcs;
		char* fn;
		int ext;
		float overallwt = 1.0;

		fn = sl_get(inimgfns, i);
		ext = il_get(inimgexts, i);
		logmsg("Reading input image \"%s\" ext %i\n", fn, ext);
		ld.filename = fn;
		// extension
		ld.xtnum = ext;
		// color plane
		ld.pnum = plane;
		ld.map = 1;
		ld.ptype = PTYPE_FLOAT;
		if (qfitsloader_init(&ld)) {
			ERROR("qfitsloader_init() failed");
			exit(-1);
		}
		if (qfits_loadpix(&ld)) {
			ERROR("qfits_loadpix() failed");
			exit(-1);
		}
		img = ld.fbuf;
		logmsg("Read image: %i x %i.\n", ld.lx, ld.ly);

		if (sigma > 0.0) {
			int k0, nk;
			float* kernel;
			logmsg("Smoothing by Gaussian with sigma=%g\n", sigma);
			kernel = convolve_get_gaussian_kernel_f(sigma, 4, &k0, &nk);
			convolve_1d_f(img, ld.lx, ld.ly, kernel, k0, nk, img, NULL);
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
		if (anwcs_imagew(inwcs) != ld.lx || anwcs_imageh(inwcs) != ld.ly) {
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
			wld.filename = fn;
			// extension
			wld.xtnum = ext;
			// color plane
			wld.pnum = 0;
			wld.map = 1;
			wld.ptype = PTYPE_FLOAT;
			if (qfitsloader_init(&wld)) {
				ERROR("qfitsloader_init() failed");
				exit(-1);
			}
			if (qfits_loadpix(&wld)) {
				ERROR("qfits_loadpix() failed");
				exit(-1);
			}
			wt = wld.fbuf;
			logmsg("Read image: %i x %i.\n", wld.lx, wld.ly);
			if (wld.lx != ld.lx || wld.ly != ld.ly) {
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
			for (j=0; j<(ld.lx*ld.ly); j++)
				img[j] /= wt[j];
		}

		coadd_add_image(coadd, img, wt, overallwt, inwcs);

		anwcs_free(inwcs);
		qfitsloader_free_buffers(&ld);
		if (wt)
			qfitsloader_free_buffers(&wld);
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
		hdr = qfits_header_readext(outwcsfn, outwcsext);
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


