
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

