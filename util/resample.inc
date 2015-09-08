/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

double MANGLE(lanczos_resample_unw_sep)(double px, double py,
										const number* img,
										int W, int H, void* token) {
	lanczos_args_t* args = token;
	int order = args->order;
	int support = order;
	double weight;
	double sum;
	int x0,x1,y0,y1;
	const number* imgrow;
	int weighted = args->weighted;

	// pre-compute Lanczos kernel weights
	double KY[12];
	double KX[12];

	x0 = MAX(0,   (int)floor(px - support));
	y0 = MAX(0,   (int)floor(py - support));
	x1 = MIN(W-1, (int) ceil(px + support));
	y1 = MIN(H-1, (int) ceil(py + support));

	int nx,ny,dx,dy;
	nx = 1+x1-x0;
	ny = 1+y1-y0;
	assert(nx < 12);
	assert(ny < 12);

	for (dy=0; dy<ny; dy++)
		KY[dy] = lanczos(py - (y0+dy), order);
	for (dx=0; dx<nx; dx++)
		KX[dx] = lanczos(px - (x0+dx), order);

	weight = 0.0;
	sum = 0.0;
	for (dy=0; dy<ny; dy++) {
		double Ky;
		double xweight;
		double xsum;
		Ky = KY[dy];
		if (Ky == 0)
			continue;
		imgrow = img + (dy+y0)*W + x0;
		xweight = 0.0;
		xsum = 0.0;
		for (dx=0; dx<nx; dx++) {
			double K;
			number pix;
			K = KX[dx];
			if (K == 0)
				continue;
			pix = imgrow[dx];
			if (isnan(pix))
				continue;
			if (weighted)
				xweight += K;
			xsum += K * pix;
		}
		if (weighted && xweight == 0.0)
			continue;
		if (weighted) {
			/*
			 weight += Ky;
			 sum += Ky * (xsum / xweight);
			 */
			weight += Ky * xweight;
		}
		sum += Ky * xsum;
	}
	if (weighted)
		return sum / weight;
	else
		return sum;
}

double MANGLE(lanczos_resample)(double px, double py,
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

double MANGLE(nearest_resample)(double px, double py,
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

