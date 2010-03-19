#include <sys/param.h>
#include <stdio.h>
#include <math.h>

#include "cairoutils.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "healpix.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "mathutil.h"

/**

 python util/tstimg.py 
 an-fitstopnm -i tstimg.fits -N 0 -X 255 | pnmtopng > tstimg.png

 hpimage tstimg; hpimage -r tstimg
 open tstimg.png tstimg-hp.png tstimg-unhp.png

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
	double zoom = 1.0;
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
		double maxD, minD;
		double amaxD, aminD;
		// for sinc:
		// FIXME -- inverse?
		scale = 1.0 / zoom;
		support = (double)order / scale;

		{
			double chx, chy;
			double cxyz[3];
			double cra, cdec;
			double dravec[3], ddecvec[3];
			// compute distance distortion matrix
			int steps = 360;
			double astep = 2.0 * M_PI / (double)steps;
			double rara, decdec, radec;
			double xra,yra,xdec,ydec;

			// center of image in healpix coords
			chx = (minx + W/2 * hpstep);
			chy = (miny + H/2 * hpstep);
			healpix_to_xyzarr(bighp, 1, chx, chy, cxyz);
			// directions of increasing RA,Dec
			//sip_get_radec_center(wcs, &cra, &cdec);
			xyzarr2radecdeg(cxyz, &cra, &cdec);
			radec_derivatives(cra, cdec, dravec, ddecvec);

			printf("dra,ddec vec lengths: %g, %g\n",
				   sqrt(square(dravec[0]) + square(dravec[1]) + square(dravec[2])),
				   sqrt(square(ddecvec[0]) + square(ddecvec[1]) + square(ddecvec[2])));
			printf("dot prods: %g, %g, %g\n",
				   dravec[0]*ddecvec[0] + dravec[1]*ddecvec[1] + dravec[2]*ddecvec[2],
				   dravec[0]*cxyz[0] + dravec[1]*cxyz[1] + dravec[2]*cxyz[2],
				   cxyz[0]*ddecvec[0] + cxyz[1]*ddecvec[1] + cxyz[2]*ddecvec[2]);

			rara = decdec = radec = 0.0;
			xra = yra = xdec = ydec = 0.0;

			maxD = -HUGE_VAL;
			minD =  HUGE_VAL;
			amaxD = aminD = -1;

			printf("dist=array([");
			for (i=0; i<steps; i++) {
				double angle = astep * i;
				double dra, ddec;
				hx = sin(angle) * hpstep + chx;
				hy = cos(angle) * hpstep + chy;
				healpix_to_xyzarr(bighp, 1, hx, hy, xyz);
				dra = ddec = 0.0;
				for (k=0; k<3; k++) {
					dra += dravec[k] * (xyz[k] - cxyz[k]);
					ddec += ddecvec[k] * (xyz[k] - cxyz[k]);
				}
				printf("[%g,%g,%g,%g],", hx-chx, hy-chy, dra, ddec);
				rara += dra*dra;
				decdec += ddec*ddec;
				radec += dra*ddec;
				xra += sin(angle) * dra;
				yra += cos(angle) * dra;
				xdec += sin(angle) * ddec;
				ydec += cos(angle) * ddec;


				// yarr.
				double d = sqrt(dra*dra + ddec*ddec);
				if (d > maxD) {
					maxD = d;
					amaxD = angle;
				}
				if (d < minD) {
					minD = d;
					aminD = angle;
				}
			}
			printf("])\n");
			printf("rara %g, decdec %g, radec %g\n", rara, decdec, radec);
			printf("x,y(ra) %g,%g, x,y(dec) %g,%g\n", xra, yra, xdec, ydec);

			printf("min,max D: %g, %g\n", minD, maxD);
			printf("min,max D angle: %g, %g\n", rad2deg(aminD), rad2deg(amaxD));

			printf("dst=array([");
			for (i=0; i<steps; i++) {
				double angle = astep * i;
				double t0,t1;
				double dst;
				hx = sin(angle);
				hy = cos(angle);
				t0 = hx * xra  + hy * yra;
				t1 = hx * xdec + hy * ydec;
				dst = hx * t0 + hy * t1;
				printf("%g,", dst);
			}
			printf("])\n");


			printf("dst2=array([");
			for (i=0; i<steps; i++) {
				double angle = astep * i;
				double t0,t1;
				double dst;
				hx = sin(angle);
				hy = cos(angle);
				t0 = maxD * (hx *  cos(amaxD) + hy * sin(amaxD));
				t1 = minD * (hx * -sin(amaxD) + hy * cos(amaxD));
				//t1 = minD * (hx *  cos(aminD) + hy * sin(aminD));
				dst = sqrt(t0*t0 + t1*t1);
				printf("%g,", dst);
			}
			printf("])\n");

		}

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

							printf("d=%g\n", d);

							double t0,t1;
							hx = ix-px;
							hy = iy-py;
							t0 = maxD * (hx *  cos(amaxD) + hy * sin(amaxD));
							t1 = minD * (hx * -sin(amaxD) + hy * cos(amaxD));
							d = sqrt(t0*t0 + t1*t1);
							d *= nside;
							L = lanczos(d * scale, order);
							printf("d2 = %g\n", d);

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


