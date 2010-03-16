#include <sys/param.h>
#include <stdio.h>

#include "cairoutils.h"
#include "sip.h"
#include "sip_qfits.h"
#include "healpix.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"

int main(int argc, char** args) {
	char* base = NULL;
	char* fn;
	unsigned char* img = NULL;
	unsigned char* Himg = NULL;
	int W, H;
	sip_t* wcs;
	double minx, miny, maxx, maxy;
	double xscale, yscale;
	int nside;
	double pixscale;
	double zoom = 1.2;
	int HW,HH;
	double hx, hy;
	int i,j;
	int bighp;
	double xyz[3];

	if (argc == 2) {
		base = args[1];
	} else {
		ERROR("Need one arg: base filename.\n");
		exit(-1);
	}

	asprintf(&fn, "%s.jpg", base);
	img = cairoutils_read_jpeg(fn, &W, &H);
	if (!img) {
		ERROR("Failed to read image file as jpeg: %s\n", img);
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

	miny = minx =  HUGE_VAL;
	maxy = maxx = -HUGE_VAL;
	for (i=0; i<4; i++) {
		double px=0,py=0;
		switch (i) {
		case 1:
			px = W;
			break;
		case 2:
			py = H;
			break;
		case 3:
			px = W;
			py = H;
			break;
		}
		sip_pixelxy2xyzarr(wcs, px, py, xyz);
		bighp = xyzarrtohealpixf(xyz, 1, &hx, &hy);
		minx = MIN(minx, hx);
		miny = MIN(miny, hy);
		maxx = MAX(maxx, hx);
		maxy = MAX(maxy, hy);
	}

	HW = (int)ceil(nside * (maxx - minx));
	HH = (int)ceil(nside * (maxy - miny));
	printf("Rendering output image: %i x %i\n", HW, HH);

	xscale = yscale = 1.0 / (float)nside;

	Himg = malloc(HW * HH * 4);
	for (i=0; i<HW*HH; i++) {
		Himg[4*i + 0] = 128;
		Himg[4*i + 1] = 128;
		Himg[4*i + 2] = 128;
		Himg[4*i + 3] = 255;
	}

	for (i=0; i<HH; i++) {
		hy = miny + i*yscale;
		for (j=0; j<HW; j++) {
			double px, py;
			int ix, iy;
			hx = minx + j*xscale;
			healpix_to_xyzarr(bighp, 1, hx, hy, xyz);
			if (!sip_xyzarr2pixelxy(wcs, xyz, &px, &py)) {
				ERROR("SIP projects to wrong side of sphere\n");
				exit(-1);
			}
			ix = (int)px;
			iy = (int)py;
			if (ix < 0 || ix >= W)
				continue;
			if (iy < 0 || iy >= H)
				continue;
			memcpy(Himg + 4*(i*HW + j), img + 4*(iy*W + ix), 3);
		}
		printf("Row %i of %i\n", i+1, HH);
	}

	asprintf(&fn, "%s-hp.jpg", base);
	printf("Writing output: %s\n", fn);

	cairoutils_write_jpeg(fn, Himg, HW, HH);
	free(fn);
	free(img);
	free(Himg);

	return 0;
}


