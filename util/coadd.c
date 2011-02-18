#include <stdlib.h>
#include <math.h>
#include <sys/param.h>

#include "coadd.h"
#include "mathutil.h"
#include "errors.h"
#include "log.h"

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

double nearest_resample(double px, double py,
						const number* img, const number* weightimg,
						int W, int H,
						double* out_wt,
						void* token) {
	int ix = round(px);
	int iy = round(py);
	double wt;

	if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
		if (out_wt)
			*out_wt = 0.0;
		return 0.0;
	}

	if (weightimg)
		wt = weightimg[iy * W + ix];
	else
		wt = 1.0;

	if (out_wt)
		*out_wt = wt;

	return img[iy*W + ix] * wt;
}

coadd_t* coadd_new(int W, int H) {
	coadd_t* ca = calloc(1, sizeof(coadd_t));
	ca->img = calloc(W * H, sizeof(number));
	ca->weight = calloc(W * H, sizeof(number));
	ca->W = W;
	ca->H = H;
	ca->resample_func = nearest_resample;
	return ca;
}

void coadd_debug(coadd_t* co) {
	int i;
	double mn,mx;
	mn = 1e300;
	mx = -1e300;
	for (i=0; i<(co->W*co->H); i++) {
		mn = MIN(mn, co->img[i]);
		mx = MAX(mx, co->img[i]);
	}
	logmsg("Coadd img min,max %g,%g\n", mn,mx);
	mn = 1e300;
	mx = -1e300;
	for (i=0; i<(co->W*co->H); i++) {
		mn = MIN(mn, co->weight[i]);
		mx = MAX(mx, co->weight[i]);
	}
	logmsg("Weight img min,max %g,%g\n", mn,mx);
	mn = 1e300;
	mx = -1e300;
	for (i=0; i<(co->W*co->H); i++) {
		if (co->weight[i] > 0) {
			mn = MIN(mn, co->img[i] / co->weight[i]);
			mx = MAX(mx, co->img[i] / co->weight[i]);
		}
	}
	logmsg("Img/Weight min,max %g,%g\n", mn,mx);
}

int coadd_add_image(coadd_t* ca, const number* img,
					const number* weightimg,
					number weight, const anwcs_t* wcs) {
	int W, H;
	int i, j;

	W = anwcs_imagew(wcs);
	H = anwcs_imageh(wcs);

	for (i=0; i<ca->H; i++) {
		for (j=0; j<ca->W; j++) {
			double ra, dec;
			double px, py;
			double wt;
			double val;

			// +1 for FITS
			if (anwcs_pixelxy2radec(ca->wcs, j+1, i+1, &ra, &dec)) {
				ERROR("Failed to project pixel (%i,%i) through output WCS\n", j, i);
				continue;
			}
			if (anwcs_radec2pixelxy(wcs, ra, dec, &px, &py)) {
				ERROR("Failed to project pixel (%i,%i) through input WCS\n", j, i);
				continue;
			}
			// -1 for FITS
			px -= 1;
			py -= 1;

			if (px < 0 || px >= W)
				continue;
			if (py < 0 || py >= H)
				continue;

			val = ca->resample_func(px, py, img, weightimg, W, H, &wt,
									ca->resample_token);
			ca->img[i*ca->W + j] += val * weight;
			ca->weight[i*ca->W + j] += wt * weight;
		}
		logverb("Row %i of %i\n", i+1, ca->H);
	}
	return 0;
}



// divide "img" by "weight"; set img=badpix where weight=0.
void coadd_divide_by_weight(coadd_t* ca, number badpix) {
	int i;
	for (i=0; i<(ca->W * ca->H); i++) {
		if (ca->weight[i] == 0)
			ca->img[i] = badpix;
		else
			ca->img[i] /= ca->weight[i];
	}
}

void coadd_free(coadd_t* ca) {
	free(ca->img);
	free(ca->weight);
	free(ca);
}

number* coadd_create_weight_image_from_range(const number* img, int W, int H,
											 number lowval, number highval) {
	int i;
	number* weight = malloc(W*H*sizeof(number));
	for (i=0; i<(W*H); i++) {
		if (img[i] <= lowval)
			weight[i] = 0;
		else if (img[i] >= highval)
			weight[i] = 0;
		else
			weight[i] = 1;
	}
	return weight;
}

void coadd_weight_image_mask_value(const number* img, int W, int H,
								   number* weight, number badval) {
	int i;
	for (i=0; i<(W*H); i++) {
		if (img[i] == badval) {
			weight[i] = 0;
		}
	}
}
