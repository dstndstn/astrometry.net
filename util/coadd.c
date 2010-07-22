#include <stdlib.h>

#include "coadd.h"
#include "errors.h"
#include "log.h"

coadd_t* coadd_new(int W, int H) {
	coadd_t* ca = calloc(1, sizeof(coadd_t));
	ca->img = malloc(W * H * sizeof(number));
	ca->weight = malloc(W * H * sizeof(number));
	ca->W = W;
	ca->H = H;
	return ca;
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

