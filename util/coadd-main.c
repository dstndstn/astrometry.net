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

static const char* OPTIONS = "hvw:o:e:O:";

void printHelp(char* progname) {
    fprintf(stderr, "%s [options] <input-FITS-image> <image-ext> <input-weight> <weight-ext> <input-WCS> <wcs-ext> [<image> <ext> <weight> <ext> <wcs> <ext>...]\n"
			"     -w <output-wcs-file>  (default: input file)\n"
			"    [-e <output-wcs-ext>]: FITS extension to read WCS from (default: primary extension, 0)\n"
			"     -o <output-image-file>\n"
			"    [-O <order>]: Lanczos order (default 3)\n"
			"    [-v]: more verbose\n"
            "\n", progname);
}
extern char *optarg;
extern int optind, opterr, optopt;

struct lanczos_args_s {
	int order;
};
typedef struct lanczos_args_s lanczos_args_t;

static double lanczos(double x, int order) {
	if (x == 0)
		return 1.0;
	if (x > order || x < -order)
		return 0.0;
	return order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
}

double lanczos_resample(double px, double py,
						const number* img, const number* weightimg,
						int W, int H,
						double* out_wt,
						void* token) {
	lanczos_args_t* args = token;
	int order = args->order;
	int support = order;

	double weight;
	double sum;
	int x0,x1,y0,y1;
	int ix,iy;

	x0 = MAX(0, (int)floor(px - support));
	y0 = MAX(0, (int)floor(py - support));
	x1 = MIN(W-1, (int) ceil(px + support));
	y1 = MIN(H-1, (int) ceil(py + support));
	weight = 0.0;
	sum = 0.0;

	for (iy=y0; iy<=y1; iy++) {
		for (ix=x0; ix<=x1; ix++) {
			double K;
			number pix;
			number wt;
			double d;
			d = hypot(px - ix, py - iy);
			K = lanczos(d, order);
			if (K == 0)
				continue;
			if (weightimg) {
				wt = weightimg[iy*W + ix];
				if (wt == 0.0)
					continue;
			} else
				wt = 1.0;
			pix = img[iy*W + ix];
			if (isnan(pix))
				// out-of-bounds pixel
				continue;
			/*
			 if (!isfinite(pix)) {
			 logverb("Pixel value: %g\n", pix);
			 continue;
			 }
			 */
			weight += K * wt;
			sum += K * wt * pix;
		}
	}

	if (out_wt)
		*out_wt = weight;
	return sum;
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

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case '?':
        case 'h':
			printHelp(progname);
			exit(0);
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

	coadd->resample_func = lanczos_resample;
	largs.order = order;
	coadd->resample_token = &largs;

	for (i=0; i<sl_size(inimgfns); i++) {
		qfitsloader ld;
		qfitsloader wld;
		float* img;
		float* wt;
		anwcs_t* inwcs;
		char* fn;
		int ext;

		fn = sl_get(inimgfns, i);
		ext = il_get(inimgexts, i);
		logmsg("Reading input image \"%s\" ext %i\n", fn, ext);
		ld.filename = fn;
		// extension
		ld.xtnum = ext;
		// color plane
		ld.pnum = 0;
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

		//W = ld.lx;
		//H = ld.ly;
		img = ld.fbuf;
		logmsg("Read image: %i x %i.\n", ld.lx, ld.ly);

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

		fn = sl_get(inwtfns, i);
		ext = il_get(inwtexts, i);
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

		if (anwcs_imagew(inwcs) != ld.lx || anwcs_imageh(inwcs) != ld.ly) {
			ERROR("Size mismatch between image and WCS!");
			exit(-1);
		}

		coadd_add_image(coadd, img, wt, 1.0, inwcs);

		qfitsloader_free_buffers(&ld);
		qfitsloader_free_buffers(&wld);
	}

	//
	logmsg("Writing output: %s\n", outfn);

	coadd_divide_by_weight(coadd, 0.0);

	if (fits_write_float_image(coadd->img, coadd->W, coadd->H, outfn)) {
		ERROR("Failed to write output image %s", outfn);
		exit(-1);
	}

	coadd_free(coadd);

	return 0;
}


