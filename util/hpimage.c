#include <sys/param.h>
#include <stdio.h>
#include <math.h>

#include "cairoutils.h"
#include "sip.h"
#include "sip_qfits.h"
#include "healpix.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "mathutil.h"

/**

 python util/tstimg.py 
 an-fitstopnm -i tstimg.fits -N 0 -X 255 | pnmtopng > tstimg.png


 */

static const char* OPTIONS = "hr";

void printHelp(char* progname) {
    fprintf(stderr, "%s [options] <base-filename>\n"
            "    [-r]: reverse direction\n"
            "\n", progname);
}
extern char *optarg;
extern int optind, opterr, optopt;


double lanczos(double x, int order) {
	if (x == 0)
		return 1.0;
	if (x > order)
		return 0.0;
	return order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
}

int main(int argc, char** args) {
    int argchar;

	char* base = NULL;
	char* fn;
	unsigned char* img = NULL;
	unsigned char* Himg = NULL;
	int W, H;
	int wcsW, wcsH;
	sip_t* wcs;
	double minx, miny, maxx, maxy;
	double hpstep;
	int nside;
	double pixscale;
	double zoom = 1.2;
	int HW,HH;
	double hx, hy;
	int i,j,k;
	int bighp;
	double xyz[3];

	bool dosinc = TRUE;
	double scale;
	int order = 2;
	double support;

	bool reverse = FALSE;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case '?':
        case 'h':
			printHelp(args[0]);
			exit(0);
		case 'r':
			reverse = TRUE;
			break;
		}

	if (argc - optind != 1) {
		ERROR("Need one arg: base filename.\n");
		exit(-1);
	}
		
	base = args[optind];

	/*
	 asprintf(&fn, "%s.jpg", base);
	 img = cairoutils_read_jpeg(fn, &W, &H);
	 if (!img) {
	 ERROR("Failed to read image file as jpeg: %s\n", img);
	 exit(-1);
	 }
	 */
	if (reverse) {
		asprintf(&fn, "%s-hp.png", base);
	} else {
		asprintf(&fn, "%s.png", base);
	}
	img = cairoutils_read_png(fn, &W, &H);
	if (!img) {
		ERROR("Failed to read image file as png: %s\n", img);
		exit(-1);
	}

	printf("Read image %s: %i x %i.\n", fn, W, H);
	free(fn);

	asprintf(&fn, "%s.wcs", base);
	wcs = sip_read_tan_or_sip_header_file_ext(fn, 0, NULL, FALSE);
	if (!wcs) {
		ERROR("Failed to read WCS from file: %s\n", fn);
		exit(-1);
	}
	free(fn);

	pixscale = sip_pixel_scale(wcs);
	nside = (int)ceil(zoom * healpix_nside_for_side_length_arcmin(pixscale / 60.0));
	printf("Using nside %i\n", nside);

	wcsW = wcs->wcstan.imagew;
	wcsH = wcs->wcstan.imageh;

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
		sip_pixelxy2xyzarr(wcs, px, py, xyz);
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
	HW = (int)ceil(nside * (maxx - minx));
	HH = (int)ceil(nside * (maxy - miny));

	if (reverse) {
		HW = wcsW;
		HH = wcsH;
	}

	printf("Rendering output image: %i x %i\n", HW, HH);

	hpstep = 1.0 / (float)nside;

	Himg = malloc(HW * HH * 4);
	for (i=0; i<HW*HH; i++) {
		Himg[4*i + 0] = 128;
		Himg[4*i + 1] = 128;
		Himg[4*i + 2] = 128;
		Himg[4*i + 3] = 128;
	}

	if (reverse) {
		// for sinc:
		// FIXME -- inverse?
		scale = 1.0 / zoom;
		support = (double)order / scale;

		for (i=0; i<HH; i++) {
			for (j=0; j<HW; j++) {
				double px, py;
				int ix, iy;

				sip_pixelxy2xyzarr(wcs, j, i, xyz);
				xyzarrtohealpixf(xyz, 1, &hx, &hy);
				// convert healpix coord to pixel coords in the healpix img.
				px = (hx - minx) / hpstep;
				py = (hy - miny) / hpstep;

				if (dosinc) {
					double weight;
					double sum[3];
					int x0,x1,y0,y1;
					// FIXME -- sloppy edge-handling.
					if (px < -support || px >= W+support)
						continue;
					if (py < -support || py >= H+support)
						continue;
					x0 = MAX(0, (int)floor(px - support));
					y0 = MAX(0, (int)floor(py - support));
					x1 = MIN(W-1, (int) ceil(px + support));
					y1 = MIN(H-1, (int) ceil(py + support));
					weight = 0.0;
					for (k=0; k<3; k++)
						sum[k] = 0.0;
					for (iy=y0; iy<=y1; iy++) {
						for (ix=x0; ix<=x1; ix++) {
							double d, L;
							if (img[4*(iy*W + ix) + 3] == 128) {
								// out-of-bounds pixel
								continue;
							}
							d = hypot(px - ix, py - iy);
							L = lanczos(d * scale, order);
							weight += L;
							for (k=0; k<3; k++)
								sum[k] += L * (double)img[4*(iy*W + ix) + k];
						}
					}
					if (weight > 0) {
						for (k=0; k<3; k++)
							Himg[4*(i*HW + j) + k] = MIN(255, MAX(0, sum[k] / weight));
						Himg[4*(i*HW + j) + 3] = 255;
					}
				} else {
					ix = (int)px;
					iy = (int)py;
					if (ix < 0 || ix >= W)
						continue;
					if (iy < 0 || iy >= H)
						continue;
					memcpy(Himg + 4*(i*HW + j), img + 4*(iy*W + ix), 3);
				}
			}
			printf("Row %i of %i\n", i+1, HH);
		}
	} else {
		// for sinc:
		// FIXME -- inverse?
		scale = zoom;
		support = (double)order / scale;

		for (i=0; i<HH; i++) {
			hy = miny + i*hpstep;
			for (j=0; j<HW; j++) {
				double px, py;
				int ix, iy;
				hx = minx + j*hpstep;
				healpix_to_xyzarr(bighp, 1, hx, hy, xyz);
				if (!sip_xyzarr2pixelxy(wcs, xyz, &px, &py)) {
					ERROR("SIP projects to wrong side of sphere\n");
					exit(-1);
				}
				if (dosinc) {
					double weight;
					double sum[3];
					int x0,x1,y0,y1;
					// FIXME -- sloppy edge-handling.
					/*
					 if (px < 0 || px >= W)
					 continue;
					 if (py < 0 || py >= H)
					 continue;
					 */
					if (px < -support || px >= W+support)
						continue;
					if (py < -support || py >= H+support)
						continue;
					x0 = MAX(0, (int)floor(px - support));
					y0 = MAX(0, (int)floor(py - support));
					x1 = MIN(W-1, (int) ceil(px + support));
					y1 = MIN(H-1, (int) ceil(py + support));
					weight = 0.0;
					for (k=0; k<3; k++)
						sum[k] = 0.0;
					for (iy=y0; iy<=y1; iy++) {
						for (ix=x0; ix<=x1; ix++) {
							double d, L;
							d = hypot(px - ix, py - iy);
							L = lanczos(d * scale, order);
							weight += L;
							for (k=0; k<3; k++)
								sum[k] += L * (double)img[4*(iy*W + ix) + k];
						}
					}
					if (weight > 0) {
						for (k=0; k<3; k++)
							Himg[4*(i*HW + j) + k] = MIN(255, MAX(0, sum[k] / weight));
						Himg[4*(i*HW + j) + 3] = 255;
					}

				} else {
					ix = (int)px;
					iy = (int)py;
					if (ix < 0 || ix >= W)
						continue;
					if (iy < 0 || iy >= H)
						continue;
					memcpy(Himg + 4*(i*HW + j), img + 4*(iy*W + ix), 3);
					Himg[4*(i*HW + j) + 3] = 255;
				}
			}
			printf("Row %i of %i\n", i+1, HH);
		}
	}

	/*
	 asprintf(&fn, "%s-hp.jpg", base);
	 printf("Writing output: %s\n", fn);
	 cairoutils_write_jpeg(fn, Himg, HW, HH);
	 */
	if (reverse) {
		asprintf(&fn, "%s-unhp.png", base);
	} else {
		asprintf(&fn, "%s-hp.png", base);
	}
	printf("Writing output: %s\n", fn);
	cairoutils_write_png(fn, Himg, HW, HH);

	free(fn);
	free(img);
	free(Himg);

	return 0;
}


