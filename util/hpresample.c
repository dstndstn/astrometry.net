#include <sys/param.h>
#include <stdio.h>
#include <math.h>

#include "anwcs.h"
#include "fitsioutils.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "healpix.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "mathutil.h"
#include "qfits_image.h"

#include "lsqr.h"

/**

 python util/tstimg.py 
 an-fitstopnm -i tstimg.fits -N 0 -X 255 | pnmtopng > tstimg.png

 hpresample tstimg; hpresample -r tstimg
 open tstimg.png tstimg-hp.png tstimg-unhp.png

 for x in tstimg.png tstimg-hp.png tstimg-unhp-{1,2}.png; do
 pngtopnm $x | pnmscale 10 | pnmtopng > zoom-$x;
 done

 for x in tstimg*.png; do
 pngtopnm $x | pnmscale 10 | pnmtopng > zoom-$x;
 done

 cp tstimg.wcs tstdot.wcs
 hpresample tstdot; hpresample -r tstdot

CFHTLS field:
 D1-25-r exposure, 715809p.fits
 wget "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/getData?archive=CFHT&file_id=715809p&dua=true"

 imcopy 715809p.fits.gz"[1][1:1024,1:1024]" 715809p-01-00.fits
 get-wcs -o 715809p-01-00.wcs 715809p-01-00.fits


 imcopy 715809p.fits.gz"[1][1:512,1:512]" small.fits


 imcopy 715809p.fits.gz"[6][100:900,100:900]" 715809p-a.fits
 get-wcs -o 715809p-a.wcs 715809p-a.fits
 an-fitstopnm -i 715809p-a.fits -N 800 -X 5000 | pnmtopng > 715809p-a.png

 hpresample 715809p-a
 hpresample -r 715809p-a
 cp 715809p-a-unhp.png 715809p-a-o2z1.png

 for z in 1 2 3; do
 for o in 2 3; do
 hpresample -o $o -z $z 715809p-a;
 hpresample -o $o -z $z -r 715809p-a;
 cp 715809p-a-unhp.png 715809p-a-o${o}z${z}.png;
 done
 done

import pyfits
import matplotlib
matplotlib.use('Agg')
from pylab import *
I1 = imread('715809p-a.png')
for z in [1,2,3]:
	for o in [2,3]:
		clf()
		I2 = imread('715809p-a-o%iz%i.png' % (o,z))[:,:,0]
		imshow(I2 - I1, vmin=-0.1, vmax=0.1, origin='lower', interpolation='nearest')
		gray()
		colorbar()
		title('Lanczos order %i; healpix oversampling factor %i' % (o,z))
		savefig('diff-o%iz%i.png' % (o,z))

 */

static const char* OPTIONS = "hrvz:o:e:w:";

void printHelp(char* progname) {
    fprintf(stderr, "%s [options] <input-FITS-filename> <output-FITS-filename>\n"
			"    [-e <ext>]: input FITS extension to read (default: primary extension, 0)\n"
            "    [-r]: reverse direction\n"
			"    [-w <wcs-file>] (default: input file)\n"
			"    [-z <zoom>]: oversample healpix grid by this factor x factor (default 1)\n"
			"    [-o <order>]: Lanczos order (default 2)\n"
            "\n", progname);
}
extern char *optarg;
extern int optind, opterr, optopt;

static void resample_image(const float* img, int W, int H, 
						   float* outimg, int outW, int outH,
						   double minx, double miny, double hpstep,
						   int bighp, const anwcs_t* wcs,
						   double support, int order, bool set_or_add) {
	 int i, j;
	 bool dosinc = TRUE;

	 for (i=0; i<outH; i++) {
		 double hx, hy;
		 hy = miny + i*hpstep;
		 for (j=0; j<outW; j++) {
			 double px, py;
			 double xyz[3];
			 int ix, iy;
			 hx = minx + j*hpstep;
			 debug("healpix (%.3f, %.3f)\n", hx, hy);
			 healpix_to_xyzarr(bighp, 1, hx, hy, xyz);
			 debug("radec (%.3f, %.3f)\n", rad2deg(xy2ra(xyz[0], xyz[1])), rad2deg(z2dec(xyz[2])));
			 if (anwcs_xyz2pixelxy(wcs, xyz, &px, &py)) {
				 ERROR("WCS projects to wrong side of sphere\n");
				 continue;
			 }
			 // MAGIC -1: FITS pixel coords...
			 px -= 1;
			 py -= 1;
			 debug("pixel (%.1f, %.1f)\n", px, py);
			 if (dosinc) {
				 float weight;
				 float sum;
				 int x0,x1,y0,y1;
				 if (px < -support || px >= W+support)
					 continue;
				 if (py < -support || py >= H+support)
					 continue;
				 x0 = MAX(0, (int)floor(px - support));
				 y0 = MAX(0, (int)floor(py - support));
				 x1 = MIN(W-1, (int) ceil(px + support));
				 y1 = MIN(H-1, (int) ceil(py + support));
				 weight = 0.0;
				 sum = 0.0;
				 for (iy=y0; iy<=y1; iy++) {
					 for (ix=x0; ix<=x1; ix++) {
						 double d, L;
						 d = hypot(px - ix, py - iy);
						 L = lanczos(d / scale, order);
						 if (L == 0)
							 continue;
						 weight += L;
						 sum += L * img[iy*W + ix];
					 }
				 }
				 if (weight != 0) {
					 if (set_or_add)
						 outimg[i*outW + j] = sum / weight;
					 else
						 outimg[i*outW + j] += sum / weight;
				 }
			 } else {
				 ix = (int)px;
				 iy = (int)py;
				 if (ix < 0 || ix >= W)
					 continue;
				 if (iy < 0 || iy >= H)
					 continue;
				 if (set_or_add)
					 outimg[i*outW + j] = img[iy*W + ix];
				 else
					 outimg[i*outW + j] += img[iy*W + ix];
			 }
		 }
		 printf("Row %i of %i\n", i+1, outH);
	 }
}

struct mvp {
	int W;
	int H;
	int outW;
	int outH;
	double minx;
	double miny;
	double hpstep;
	int bighp;
	anwcs_t* wcs;
	double support;
	int order;
};
typedef struct mvp mvp_t;


static void mat_vec_prod(long mode, dvec* x, dvec* y, void* token) {
	mvp_t* mvp;
	logmsg("mat_vec_prod: mode=%i\n", (int)mode);
	if (mode == 0) {
		// y = y + A * x
		/*
		 resample_image(img, W, H, outimg, outW, outH,
		 minx, miny, hpstep, bighp, wcs,
		 support, order, FALSE);
		 */
	} else if (mode == 1) {
		// x = x + A^T * y
	} else {
		ERROR("Unknown mode %i", (int)mode);
		exit(-1);
	}
}



static double lanczos(double x, int order) {
	if (x == 0)
		return 1.0;
	if (x > order || x < -order)
		return 0.0;
	return order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
}

int main(int argc, char** args) {
    int argchar;

	char* infn = NULL;
	char* outfn = NULL;
	char* wcsfn = NULL;

	float* img = NULL;
	float* outimg = NULL;

	int W, H;
	int wcsW, wcsH;
	int hpW, hpH;
	anwcs_t* wcs;
	double minx, miny, maxx, maxy;
	double hpstep;
	int nside;
	double pixscale;
	double zoom = 1.0;
	double realzoom;
	int outW,outH;
	double hx, hy;
	int i,j,k;
	int bighp;
	double xyz[3];

	bool dosinc = TRUE;
	double scale;
	int order = 2;
	double support;

	bool reverse = FALSE;

	double maxD, minD;
	double amaxD;

	int loglvl = LOG_MSG;
	qfitsloader ld;

	int fitsext = 0;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case '?':
        case 'h':
			printHelp(args[0]);
			exit(0);
		case 'v':
			loglvl++;
			break;
		case 'r':
			reverse = TRUE;
			break;
		case 'z':
			zoom = atof(optarg);
			break;
		case 'e':
			fitsext = atoi(optarg);
			break;
		case 'w':
			wcsfn = optarg;
			break;
		case 'o':
			order = atoi(optarg);
			break;
		}

	log_init(loglvl);
	fits_use_error_system();

	if (argc - optind != 2) {
		ERROR("Need args: input and output FITS image filenames.\n");
		printHelp(args[0]);
		exit(-1);
	}
		
	infn = args[optind];
	outfn = args[optind+1];
	if (!wcsfn)
		wcsfn = infn;

	ld.filename = infn;
	// extension
	ld.xtnum = fitsext;
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
	W = ld.lx;
	H = ld.ly;
	img = ld.fbuf;

	printf("Read image %s: %i x %i.\n", infn, W, H);

	wcs = anwcs_open(wcsfn, fitsext);
	printf("Reading WCS file %s\n", wcsfn);
	if (!wcs) {
		ERROR("Failed to read WCS from file: %s\n", wcsfn);
		exit(-1);
	}

	pixscale = anwcs_pixel_scale(wcs);
	if (pixscale == 0) {
		ERROR("Pixel scale from the WCS file is zero.  Usually this means the image has no valid WCS header.\n");
		exit(-1);
	}
	printf("Target zoom: %g\n", zoom);
	printf("Pixel scale: %g arcsec/pix\n", pixscale);
	nside = (int)ceil(zoom * healpix_nside_for_side_length_arcmin(pixscale / 60.0));
	printf("Using nside %i\n", nside);
	realzoom = (pixscale/60.0) / healpix_side_length_arcmin(nside);
	printf("Real zoom: %g\n", realzoom);

	wcsW = anwcs_imagew(wcs);
	wcsH = anwcs_imageh(wcs);

	// when going forward, wcsW == W

	miny = minx =  HUGE_VAL;
	maxy = maxx = -HUGE_VAL;
	for (i=0; i<4; i++) {
		double px=0,py=0;
		switch (i) {
		case 1:
			px = wcsW;
			break;
		case 2:
			py = wcsH;
			break;
		case 3:
			px = wcsW;
			py = wcsH;
			break;
		}
		anwcs_pixelxy2xyz(wcs, px, py, xyz);
		bighp = xyzarrtohealpixf(xyz, 1, &hx, &hy);
		minx = MIN(minx, hx);
		miny = MIN(miny, hy);
		maxx = MAX(maxx, hx);
		maxy = MAX(maxy, hy);
	}
	// move minx/y down to the next smallest nside pixel value.
	minx = 1.0/(double)nside * floor(minx * nside);
	miny = 1.0/(double)nside * floor(miny * nside);
	maxx = 1.0/(double)nside *  ceil(maxx * nside);
	maxy = 1.0/(double)nside *  ceil(maxy * nside);
	outW = (int)ceil(nside * (maxx - minx));
	outH = (int)ceil(nside * (maxy - miny));
	logverb("Healpix x range [%.3f, %.3f], [%.3f, %.3f]\n", minx, maxx, miny, maxy);

	if (reverse) {
		outW = wcsW;
		outH = wcsH;
		hpW = W;
		hpH = H;
	} else {
		hpW = outW;
		hpH = outH;
	}

	printf("Rendering output image: %i x %i\n", outW, outH);

	hpstep = 1.0 / (float)nside;
	logverb("hpstep %g\n", hpstep);

	outimg = malloc(outW * outH * sizeof(float));
	for (i=0; i<outW*outH; i++)
		outimg[i] = 1.0 / 0.0;



	// ASIDE - find how distances transform from 'image' to 'healpix image' space.
	{
		double chx, chy;
		double cxyz[3];
		double cra, cdec;
		double dravec[3], ddecvec[3];
		// compute distance distortion matrix, poorly, by probing a circle.
		int steps = 360;
		double astep = 2.0 * M_PI / (double)steps;

		// center of image in healpix coords
		chx = (minx + hpW/2 * hpstep);
		chy = (miny + hpH/2 * hpstep);
		healpix_to_xyzarr(bighp, 1, chx, chy, cxyz);
		// directions of increasing RA,Dec
		xyzarr2radecdeg(cxyz, &cra, &cdec);
		radec_derivatives(cra, cdec, dravec, ddecvec);
		maxD = -HUGE_VAL;
		minD =  HUGE_VAL;
		amaxD = -1;
		for (i=0; i<steps; i++) {
			double angle = astep * i;
			double dra, ddec;
			double d;
			hx = sin(angle) * hpstep + chx;
			hy = cos(angle) * hpstep + chy;
			healpix_to_xyzarr(bighp, 1, hx, hy, xyz);
			dra = ddec = 0.0;
			for (k=0; k<3; k++) {
				dra += dravec[k] * (xyz[k] - cxyz[k]);
				ddec += ddecvec[k] * (xyz[k] - cxyz[k]);
			}
			d = sqrt(dra*dra + ddec*ddec);
			if (d > maxD) {
				maxD = d;
				amaxD = angle;
			}
			minD = MIN(d, minD);
		}
		printf("min,max D: %g, %g\n", minD, maxD);
		printf("max D angle: %g\n", rad2deg(amaxD));
	}



	if (reverse) {
		// for sinc:
		scale = 1.0;
		support = (double)order * scale;

		for (i=0; i<outH; i++) {
			for (j=0; j<outW; j++) {
				double px, py;
				int ix, iy;
				// MAGIC +1: FITS pixel coords.
				anwcs_pixelxy2xyz(wcs, j+1, i+1, xyz);
				xyzarrtohealpixf(xyz, 1, &hx, &hy);
				// convert healpix coord to pixel coords in the healpix img.
				px = (hx - minx) / hpstep;
				py = (hy - miny) / hpstep;

				if (dosinc) {
					float weight;
					float sum;
					int x0,x1,y0,y1;
					if (px < -support || px >= W+support)
						continue;
					if (py < -support || py >= H+support)
						continue;
					x0 = MAX(0, (int)floor(px - support));
					y0 = MAX(0, (int)floor(py - support));
					x1 = MIN(W-1, (int) ceil(px + support));
					y1 = MIN(H-1, (int) ceil(py + support));
					weight = 0.0;
					sum = 0.0;
					for (iy=y0; iy<=y1; iy++) {
						for (ix=x0; ix<=x1; ix++) {
							double d, L;
							float pix = img[iy*W + ix];
							if (isnan(pix))
								// out-of-bounds pixel
								continue;
							d = hypot(px - ix, py - iy);
							L = lanczos(d / scale, order);
							weight += L;
							sum += L * pix;
						}
					}
					if (weight != 0)
						outimg[i*outW + j] = sum / weight;

				} else {
					ix = (int)px;
					iy = (int)py;
					if (ix < 0 || ix >= W)
						continue;
					if (iy < 0 || iy >= H)
						continue;
					outimg[i*outW + j] = img[iy*W + ix];
				}
			}
			printf("Row %i of %i\n", i+1, outH);
		}




		// RHL's inverse-resampling method.
		{
			lsqr_input *lin;
			lsqr_output *lout;
			lsqr_work *lwork;
			lsqr_func *lfunc;
			int R, C;
			void* token = NULL;
			//dvec v;

			// rows, cols.
			R = outW * outH;
			C = W * H;

			alloc_lsqr_mem(&lin, &lout, &lwork, &lfunc, R, C);
			lfunc->mat_vec_prod = mat_vec_prod;
			lin->lsqr_fp_out = stdout;
			lin->num_rows = R;
			lin->num_cols = C;
			//lin->rel_mat_err = 1e-10;
			//lin->rel_rhs_err = 1e-10;
			//lin->cond_lim = 10.0; // * cnum;
			lin->damp_val = 1.0;
			lin->rel_mat_err = 0;
			lin->rel_rhs_err = 0;
			lin->cond_lim = 0;
			lin->max_iter = R + C + 50;

			// input image is RHS.
			/*
			 v.length = W * H;
			 v.elements = img;
			 dvec_copy(&v, lin->rhs_vec);
			 */
			for (i=0; i<(W*H); i++)
				lin->rhs_vec->elements[i] = img[i];

			// output image is initial guess
			/*
			 v.length = outW * outH;
			 v.elements = outimg;
			 dvec_copy(&v, lin->sol_vec);
			 */
			for (i=0; i<(outW*outH); i++)
				lin->sol_vec->elements[i] = outimg[i];

			lsqr(lin, lout, lwork, lfunc, token);

			logmsg("Termination reason: %i\n", (int)lout->term_flag);
			logmsg("Iterations: %i\n", (int)lout->num_iters);
			logmsg("Condition number estimate: %g\n", lout->mat_cond_num);
			logmsg("Normal of residuals: %g\n", lout->resid_norm);
			logmsg("Norm of W*resids: %g\n", lout->mat_resid_norm);

			// Grab output solution...
			/*
			 v.length = outW * outH;
			 v.elements = outimg;
			 dvec_copy(lout->sol_vec, &v);
			 */
			for (i=0; i<(outW*outH); i++)
				outimg[i] = lin->sol_vec->elements[i];

			// lout->std_err_vec

			free_lsqr_mem(lin, lout, lwork, lfunc);
		}


	} else {
		scale = 1.0;
		support = (double)order * scale;

		resample_image(img, W, H, outimg, outW, outH,
					   minx, miny, hpstep, bighp, wcs,
					   support, order, TRUE);
	}

	printf("Writing output: %s\n", outfn);
	if (fits_write_float_image(outimg, outW, outH, outfn)) {
		ERROR("Failed to write output image %s", outfn);
		exit(-1);
	}

	free(img);
	free(outimg);
	return 0;
}

