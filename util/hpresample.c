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
#include "keywords.h"
#include "tic.h"
#include "sparsematrix.h"

#include "lsqr.h"

Unused static void testit();

/**
Test with CFHTLS field (D1-25-r exposure, 715809p.fits)

 wget "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/getData?archive=CFHT&file_id=715809p&dua=true"

 #imcopy 715809p.fits.gz"[1][1:1024,1:1024]" 715809p-01-00.fits
 #get-wcs -o 715809p-01-00.wcs 715809p-01-00.fits

 imcopy 715809p.fits.gz"[1][1:512,1:512]" small.fits

 hpresample small.fits hp.fits
 hpresample -r -w small.fits hp.fits unhp.fits

 python hpresample-plots.py



 imcopy ~/DATA/715809p.fits"[1]" big.fits
 hpresample big.fits bighp.fits
 hpresample -r -w big.fits bighp.fits bigunhp.fits

 */

static const char* OPTIONS = "hrvz:o:e:w:W";

void printHelp(char* progname) {
    fprintf(stderr, "%s [options] <input-FITS-filename> <output-FITS-filename>\n"
			"    [-e <ext>]: input FITS extension to read (default: primary extension, 0)\n"
            "    [-r]: reverse direction\n"
			"    [-w <wcs-file>] (default: input file)\n"
			"    [-z <zoom>]: oversample healpix grid by this factor x factor (default 1)\n"
			"    [-o <order>]: Lanczos order (default 2)\n"
			"    [-W]: write out an image at each step.\n"
            "\n", progname);
}
extern char *optarg;
extern int optind, opterr, optopt;

static double lanczos(double x, int order) {
	if (x == 0)
		return 1.0;
	if (x > order || x < -order)
		return 0.0;
	return order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
}

static void resample_image(const double* img, int W, int H, 
						   double* outimg, int outW, int outH,
						   double minx, double miny, double hpstep,
						   int bighp, const anwcs_t* wcs,
						   double support, int order, double scale,
						   bool set_or_add, double* rowsum) {
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
				 double weight;
				 double sum;
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
						 if (!isfinite(img[iy*W + ix])) {
							 logmsg("Image pixel (%i,%i) = %g\n", ix, iy, img[iy*W + ix]);
						 }
					 }
				 }
				 if (rowsum)
					 rowsum[i*outW + j] = weight;

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
		 logverb("Row %i of %i\n", i+1, outH);
	 }
}

static void mat_vec_prod_2(long mode, dvec* x, dvec* y, void* token) {
	sparsematrix_t* sp = token;
	double t0, dt;
	
	logverb("mat_vec_prod_2: mode=%i\n", (int)mode);
	logverb("before update: norm2(x) = %g\n", dvec_norm2(x));
	logverb("before update: norm2(y) = %g\n", dvec_norm2(y));

	t0 = timenow();
	if (mode == 0) {
		// y = y + A * x
		assert(sp->R == y->length);
		assert(sp->C == x->length);
		sparsematrix_mult_vec(sp, x->elements, y->elements, TRUE);
	} else if (mode == 1) {
		// x = x + A^T * y
		assert(sp->R == y->length);
		assert(sp->C == x->length);
		sparsematrix_transpose_mult_vec(sp, y->elements, x->elements, TRUE);
	} else {
		ERROR("Unknown mode %i", (int)mode);
		exit(-1);
	}
	dt = timenow() - t0;
	logmsg("matrix mult took %g s\n", dt);

	logverb("after update: norm2(x) = %g\n", dvec_norm2(x));
	logverb("after update: norm2(y) = %g\n", dvec_norm2(y));
}


struct write_image_token {
	char* fnpat;
	float* img;
	int W, H;
};

static int write_images(lsqr_input* lin, lsqr_output* lout, void* token) {
	struct write_image_token* wit = token;
	char filename[256];
	int i;
	sprintf(filename, wit->fnpat, lout->num_iters);
	for (i=0; i<wit->W*wit->H; i++)
		wit->img[i] = lout->sol_vec->elements[i];
	logmsg("Writing %s...\n", filename);
	if (fits_write_float_image(wit->img, wit->W, wit->H, filename)) {
		ERROR("Failed to write output image %s", filename);
		exit(-1);
	}
	return 0;
}

int main(int argc, char** args) {
    int argchar;

	char* infn = NULL;
	char* outfn = NULL;
	char* wcsfn = NULL;

	double* img = NULL;
	double* outimg = NULL;

	int W, H;
	int wcsW, wcsH;
	int hpW, hpH;
	int imW, imH;
	anwcs_t* wcs;
	double minx, miny, maxx, maxy;
	double hpstep;
	int nside;
	double pixscale;
	double zoom = 1.0;
	double realzoom;
	int outW,outH;
	double hx, hy;
	int i,j;
	int bighp;
	double xyz[3];

	bool dosinc = TRUE;
	double scale;
	int order = 2;
	double support;
	bool reverse = FALSE;

	int loglvl = LOG_MSG;
	qfitsloader ld;

	int fitsext = 0;
	double t0, dt;

	bool writeimages = FALSE;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case '?':
        case 'h':
			printHelp(args[0]);
			exit(0);
		case 'v':
			loglvl++;
			break;
		case 'W':
			writeimages = TRUE;
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

	//testit();

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
	ld.ptype = PTYPE_DOUBLE;
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
	img = ld.dbuf;

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

	imW = wcsW;
	imH = wcsH;

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

	hpstep = 1.0 / (double)nside;
	logverb("hpstep %g\n", hpstep);

	outimg = malloc(outW * outH * sizeof(double));
	for (i=0; i<outW*outH; i++)
		outimg[i] = 1.0 / 0.0;

	if (reverse) {

		// for sinc:
		scale = 1.0;
		support = (double)order * scale;

		t0 = timenow();
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
					double weight;
					double sum;
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
							double pix = img[iy*W + ix];
							if (isnan(pix))
								// out-of-bounds pixel
								continue;
							if (!isfinite(pix)) {
								logverb("Pixel value: %g\n", pix);
								continue;
							}
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
			logverb("Row %i of %i\n", i+1, outH);
		}
		dt = timenow() - t0;
		logmsg("Initial resampling took %g s\n", dt);

		// RHL's inverse-resampling method, try #2, using a sparse matrix representation
		{
			lsqr_input *lin;
			lsqr_output *lout;
			lsqr_work *lwork;
			lsqr_func *lfunc;
			int R, C;
			int* rowmap;
			int Rused;
			int Rall;
			sparsematrix_t* sp;
			int i, j;

			t0 = timenow();

			// rows, cols.
			R = W * H;
			C = outW * outH;
			sp = sparsematrix_new(R, C);

			// Compute the matrix W that applies the Lanczos convolution
			// kernel to the image to produce the healpix image.
			for (i=0; i<hpH; i++) {
				double hx, hy;
				hy = miny + i*hpstep;
				for (j=0; j<hpW; j++) {
					double px, py;
					double xyz[3];
					int ix, iy;
					int x0,x1,y0,y1;
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
					if (px < -support || px >= imW+support)
							continue;
						if (py < -support || py >= imH+support)
							continue;
						x0 = MAX(0, (int)floor(px - support));
						y0 = MAX(0, (int)floor(py - support));
						x1 = MIN(imW-1, (int) ceil(px + support));
						y1 = MIN(imH-1, (int) ceil(py + support));

						for (iy=y0; iy<=y1; iy++) {
							for (ix=x0; ix<=x1; ix++) {
								double d, L;
								d = hypot(px - ix, py - iy);
								L = lanczos(d / scale, order);
								if (L == 0)
									continue;
								sparsematrix_set(sp, i*hpW + j, iy*imW + ix, L);
							}
						}
				}
			}

			printf("Number of non-zero matrix elements: %i\n", sparsematrix_count_elements(sp));

			// find elements (pixels) in the healpix image that are used;
			// ie, rows in the W matrix that contain elements.
			rowmap = malloc(R * sizeof(int));
			Rused = 0;
			for (i=0; i<R; i++) {
				//if (sparsematrix_count_elements_in_row(sp, i)) {
				// Pixels near the boundary can have negative or small weight sums.
				// Just eliminate them.
				double sum = sparsematrix_sum_row(sp, i);
				if (sum > 0.5) {
					sparsematrix_scale_row(sp, i, 1.0/sum);
					rowmap[Rused] = i;
					Rused++;
				}
			}
			Rall = R;
			R = Rused;
			sparsematrix_subset_rows(sp, rowmap, R);

			for (i=0; i<R; i++) {
				assert(sparsematrix_count_elements_in_row(sp, i) > 0);
				//logverb("Row %i (%i): %i elements set.\n", i, rowmap[i], sparsematrix_count_elements_in_row(sp, i));
			}

			printf("Trimmed to %i rows and %i elements\n", R, sparsematrix_count_elements(sp));

			dt = timenow() - t0;
			logmsg("Computing matrix %g s\n", dt);

			alloc_lsqr_mem(&lin, &lout, &lwork, &lfunc, R, C);
			lfunc->mat_vec_prod = mat_vec_prod_2;
			lin->lsqr_fp_out = stdout;
			lin->num_rows = R;
			lin->num_cols = C;
			lin->damp_val = 1e-3;
			//lin->damp_val = 0;
			lin->rel_mat_err = 0;
			lin->rel_rhs_err = 0;
			lin->cond_lim = 0;
			lin->max_iter = 200;

			// input image is RHS.
			assert(lin->rhs_vec->length == R);
			for (i=0; i<R; i++)
				lin->rhs_vec->elements[i] = (isfinite(img[rowmap[i]]) ? img[rowmap[i]] : 0);
			for (i=0; i<R; i++)
				assert(isfinite(lin->rhs_vec->elements[i]));
			logmsg("lin->rhs_vec norm2 is %g\n", dvec_norm2(lin->rhs_vec));

			// output image is initial guess
			assert(lin->sol_vec->length == outW*outH);
			assert(lin->sol_vec->length == C);
			for (i=0; i<(outW*outH); i++)
				lin->sol_vec->elements[i] = (isfinite(outimg[i]) ? outimg[i] : 0);
			logmsg("lin->sol_vec norm2 is %g\n", dvec_norm2(lin->sol_vec));

			{
				char checkfn[256];
				sprintf(checkfn, "step-00.fits");
				float* fimg = (float*)outimg;
				for (i=0; i<outW*outH; i++)
					fimg[i] = isfinite(outimg[i]) ? outimg[i] : 0;
				if (fits_write_float_image(fimg, outW, outH, checkfn)) {
					ERROR("Failed to write output image %s", checkfn);
					exit(-1);
				}
			}

			{
				// check W * I: should be ~= H.
				float* fimg = calloc(W * H, sizeof(float));
				sparsematrix_mult_vec(sp, lin->sol_vec->elements, outimg, FALSE);
				for (i=0; i<R; i++)
					fimg[rowmap[i]] = outimg[i];
				char* checkfn = "WI.fits";
				if (fits_write_float_image(fimg, hpW, hpH, checkfn)) {
					ERROR("Failed to write output image %s", checkfn);
					exit(-1);
				}
				free(fimg);
			}

			struct write_image_token wit;

			int k;
			for (k=0; k<100; k++) {
				lin->max_iter = 100;
				for (i=0; i<R; i++)
					lin->rhs_vec->elements[i] = (isfinite(img[rowmap[i]]) ? img[rowmap[i]] : 0);

				wit.fnpat = "step-%02i.fits";
				wit.img = (float*)outimg;
				wit.W = outW;
				wit.H = outH;

				t0 = timenow();
				if (writeimages)
					lsqr(lin, lout, lwork, lfunc, sp, write_images, &wit);
				else
					lsqr(lin, lout, lwork, lfunc, sp, NULL, NULL);
				dt = timenow() - t0;
				logmsg("lsqr() took %g s\n", dt);

				logmsg("Termination reason: %i\n", (int)lout->term_flag);
				logmsg("Iterations: %i\n", (int)lout->num_iters);
				logmsg("Condition number estimate: %g\n", lout->mat_cond_num);
				logmsg("Normal of residuals: %g\n", lout->resid_norm);
				logmsg("Norm of W*resids: %g\n", lout->mat_resid_norm);

				logmsg("lin->sol_vec = %p.  lout->sol_vec = %p\n", lin->sol_vec, lout->sol_vec);
				logmsg("lin->sol_vec->elems = %p.  lout->sol_vec->elems = %p\n", lin->sol_vec->elements, lout->sol_vec->elements);

				// Grab output solution...
				for (i=0; i<(outW*outH); i++)
					outimg[i] = lout->sol_vec->elements[i];
				// lout->std_err_vec

				/*{
					char checkfn[256];
					sprintf(checkfn, "step-%i.fits", k+1);
					float* fimg = (float*)outimg;
					for (i=0; i<outW*outH; i++)
						fimg[i] = outimg[i];
					if (fits_write_float_image(fimg, outW, outH, checkfn)) {
						ERROR("Failed to write output image %s", checkfn);
						exit(-1);
					}
				 }*/
				break;
			}

			// (re-)grab output solution.
			for (i=0; i<(outW*outH); i++)
				outimg[i] = lout->sol_vec->elements[i];

			free_lsqr_mem(lin, lout, lwork, lfunc);
			free(rowmap); rowmap = NULL;
			sparsematrix_free(sp);
		}

	} else {
		scale = 1.0;
		support = (double)order * scale;

		t0 = timenow();
		resample_image(img, W, H, outimg, outW, outH,
					   minx, miny, hpstep, bighp, wcs,
					   support, order, scale, TRUE, NULL);
		dt = timenow() - t0;
		logmsg("Resampling took %g s\n", dt);
	}

	printf("Writing output: %s\n", outfn);
	// HACK -- reduce output image to float, in-place.
	{
		float* fimg = (float*)outimg;
		for (i=0; i<outW*outH; i++)
			fimg[i] = outimg[i];
		if (fits_write_float_image(fimg, outW, outH, outfn)) {
			ERROR("Failed to write output image %s", outfn);
			exit(-1);
		}
	}

	free(img);
	free(outimg);
	return 0;
}










Unused static void testit() {
	// 1-D tests.
	int Nin=20;
	double input[Nin];
	double mn = Nin/2.0;
	double sig = 2;
	int i, j;
	int Nout = 20;
	double outx[Nout];
	double output[Nout];
	double recon[Nin];
	int order = 2;
	for (i=0; i<Nin; i++)
		input[i] = exp(-(i-mn)*(i-mn) / (2.0*sig*sig));
	for (i=0; i<Nout; i++)
		outx[i] = 0.99 * i + 0.4;
	int R,C;
	sparsematrix_t* sp;
	R = Nout;
	C = Nin;
	sp = sparsematrix_new(R, C);
	for (i=0; i<Nout; i++) {
		double sum, weight;
		sum = weight = 0.0;
		for (j=0; j<Nin; j++) {
			double dx = outx[i] - j;
			double L = lanczos(dx, order);
			if (L == 0)
				continue;
			sum += L * input[j];
			weight += L;
			sparsematrix_set(sp, i, j, L);
		}
		output[i] = sum / weight;
	}
	sparsematrix_normalize_rows(sp);
	lsqr_input *lin;
	lsqr_output *lout;
	lsqr_work *lwork;
	lsqr_func *lfunc;
	alloc_lsqr_mem(&lin, &lout, &lwork, &lfunc, R, C);
	lfunc->mat_vec_prod = mat_vec_prod_2;
	lin->lsqr_fp_out = stdout;
	lin->num_rows = R;
	lin->num_cols = C;
	//lin->damp_val = 1.0;
	lin->damp_val = 0.0;
	lin->rel_mat_err = 0;
	lin->rel_rhs_err = 0;
	lin->cond_lim = 0;
	lin->max_iter = R + C + 50;
	// target = input
	assert(lin->rhs_vec->length == R);
	for (i=0; i<R; i++)
		lin->rhs_vec->elements[i] = output[i];
	logmsg("lin->rhs_vec norm2 is %g\n", dvec_norm2(lin->rhs_vec));
	// initial guess = 0
	for (i=0; i<C; i++)
		lin->sol_vec->elements[i] = 0.0;
	logmsg("lin->sol_vec norm2 is %g\n", dvec_norm2(lin->sol_vec));
	lsqr(lin, lout, lwork, lfunc, sp, NULL, NULL);
	logmsg("Termination reason: %i\n", (int)lout->term_flag);
	logmsg("Iterations: %i\n", (int)lout->num_iters);
	logmsg("Condition number estimate: %g\n", lout->mat_cond_num);
	logmsg("Normal of residuals: %g\n", lout->resid_norm);
	logmsg("Norm of W*resids: %g\n", lout->mat_resid_norm);
	// Grab output solution...
	for (i=0; i<C; i++)
		recon[i] = lout->sol_vec->elements[i];
	printf("In      ----->    Out    ------>    Reconstruct in\n");
	for (i=0; i<C; i++) {
		printf("%12.5f %12.5f %12.5f\n", input[i], output[i], recon[i]);
	}
	free_lsqr_mem(lin, lout, lwork, lfunc);
	sparsematrix_free(sp);
}
