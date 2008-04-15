/*
   A test suite for simplexy.
   Processes every jpeg or png in test_simplexy_images, and outputs
   the verbose debugging data, as well as a list of the coordinates,
   and annotated images into the same folder (each beginning with "out_").
   Annotated images (those beginning with "out_") are not processed.

   I recommend running this:
   make demo_simplexy ; demo_simplexy > & demo_simplexy_images/output.txt ; diff demo_simplexy_images/ground.txt demo_simplexy_images/output.txt

   Diff returns the differences between the current run and the "ground truth"
   stored in ground.txt.

   Jon Barron, 2007
   */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <regex.h>
#include <cairo.h>
#include <math.h>

#include "cairoutils.h"
#include "dimage.h"

#define PEAKDIST 4

int is_png(const struct dirent *de) {
	regex_t preq;
	regcomp(&preq, "[^ ]*[.](png|PNG)$", REG_EXTENDED);
	return !regexec(&preq, de->d_name, (size_t) 0, NULL, 0);
}

int is_jpeg(const struct dirent *de) {
	regex_t preq;
	regcomp(&preq, "[^ ]*[.](jpg|JPG|jpeg|JPEG)$", REG_EXTENDED);
	return !regexec(&preq, de->d_name, (size_t) 0, NULL, 0);
}

int is_image(const struct dirent *de) {
	return is_jpeg(de) || is_png(de);
}

int is_output(const struct dirent *de) {
	regex_t preq;
	regcomp(&preq, "out_[^ ]*", REG_EXTENDED);
	return !regexec(&preq, de->d_name, (size_t) 0, NULL, 0);
}

int is_input_image(const struct dirent *de) {
	return is_image(de) && !is_output(de);
}

float* to_bw_f(unsigned char *image, int imW, int imH) {
	int w, h, c;
	float *image_bw;
	float v;

	image_bw = malloc(sizeof(float) * imW * imH);
	for (w = 0; w < imW; w++) {
		for (h = 0; h < imH; h++) {
			v = 0.0;
			for (c = 0; c <= 2; c++) {
				v = v + ((float)(image[4*(w*imH+h) + c])) / 3.0;
			}
			image_bw[w*imH + h] = v;
		}
	}

	return image_bw;
}

unsigned char* to_bw_u8(unsigned char *image, int imW, int imH) {
	int i;
	unsigned char *image_bw, *p;
	image_bw = p = malloc(sizeof(unsigned char) * imW * imH);
	for (i = 0; i < imW*imH; i++, image += 4, p++) {
		int total = image[0] + image[1] + image[2];
		*p = total / 3;
	}
	return image_bw;
}

unsigned char* to_cairo_bw(float *image_bw, int imW, int imH) {
	int w, h, c;
	unsigned char* image_cairo;

	image_cairo = malloc(sizeof(unsigned char) * imW * imH * 4);
	for (w = 0; w < imW; w++) {
		for (h = 0; h < imH; h++) {
			for (c = 0; c <= 2; c++) {
				image_cairo[4*(w*imH + h) + c] = (unsigned char)(image_bw[w*imH + h]);
			}
			image_cairo[4*(w*imH + h) + 3] = 255;
		}
	}

	return image_cairo;
}

int main(void) {
	struct dirent **namelist;
	int i, n, N, peak;
	unsigned char *image = NULL;
	float *image_bw_f;
	unsigned char *image_bw_u8;
	unsigned char *image_out;
	char fullpath[255];
	char outpath[255];
	int imW, imH;
	int peakX, peakY, peakXi, peakYi, peakDist;

	float dpsf, plim, dlim, saddle;
	int maxper, maxsize, halfbox, maxnpeaks;

	float *x = NULL;
	float *y = NULL;
	float *flux = NULL;
	float sigma;
	int npeaks;

	dpsf = 1.0;
	plim = 8.0;
	dlim = dpsf;
	saddle = 5.0;
	maxper = 1000;
	maxsize = 1000;
	halfbox = 100;
	maxnpeaks = 10000;

//	N = scandir("demo_simplexy_images", &namelist, is_input_image, alphasort);
	N = scandir("m44_images", &namelist, is_input_image, alphasort);
	if (N < 0) {
		perror("scandir");
		return 1;
	}

	for (n = 0; n < N; n++) {

//		strcpy(fullpath, "demo_simplexy_images/");
		strcpy(fullpath, "m44_images/");
		strcat(fullpath, namelist[n]->d_name);

//		strcpy(outpath, "demo_simplexy_images/out_");
		strcpy(outpath, "m44_images/out_");
		strcat(outpath, namelist[n]->d_name);
		outpath[strlen(outpath)-4] = '\0';
		strcat(outpath, ".png");

		fprintf(stderr, "demo_simplexy: loading %s ", fullpath);

		if (is_png(namelist[n])) {
			fprintf(stderr, "as a PNG\n");
			image = cairoutils_read_png(fullpath, &imW, &imH);
		}

		if (is_jpeg(namelist[n])) {
			fprintf(stderr, "as a JPEG\n");
			image = cairoutils_read_jpeg(fullpath, &imW, &imH);
		}

		//      image_bw_f = to_bw_f(image, imW, imH);
		image_bw_u8 = to_bw_u8(image, imW, imH);
		image_bw_f = malloc(sizeof(float) * imW * imH);
		for (i = 0; i < imW*imH; i++) {
			image_bw_f[i] = (float)image_bw_u8[i];
		}

		x = malloc(maxnpeaks * sizeof(float));
		y = malloc(maxnpeaks * sizeof(float));
		flux = malloc(maxnpeaks * sizeof(float));

		fprintf(stderr, "demo_simplexy: running %s\n", fullpath);

		simplexy_u8(image_bw_u8, imW, imH, dpsf, plim, dlim, saddle, maxper,
		            maxnpeaks, maxsize, halfbox, &sigma, x, y, flux, &npeaks, 1);

		image_out = malloc(sizeof(unsigned char) * imW * imH * 4);
		memcpy(image_out, image, imW*imH*4);
		//      image_out = to_cairo_bw(image_bw, imW, imH);

		/* draw the peaks */
		for (peak = 0; peak < npeaks; peak++) {
//			fprintf(stderr, "%.2f %.2f\n", x[peak], y[peak]);

			for (peakXi = -PEAKDIST; peakXi <= PEAKDIST; peakXi++) {
				for (peakYi = -PEAKDIST; peakYi <= PEAKDIST; peakYi++) {
					peakX = ((int)round(x[peak])) + peakXi;
					peakY = ((int)round(y[peak])) + peakYi;
					peakDist = peakXi * peakXi + peakYi * peakYi;

					if (  abs(peakDist - PEAKDIST*PEAKDIST) > 3 ||
					        peakX <= 0 ||
					        peakY <= 0 ||
					        peakX >= imW ||
					        peakY >= imH) {
						continue;
					}

					image_out[4*(peakY*imW + peakX) + 0] = 0;
					image_out[4*(peakY*imW + peakX) + 1] = 255;
					image_out[4*(peakY*imW + peakX) + 2] = 0;
				}
			}
		}

		cairoutils_write_png(outpath, image_out, imW, imH);

		free(namelist[n]);
		free(image);
		free(image_bw_f);
		free(image_bw_u8);
		free(image_out);
		free(x);
		free(y);
		free(flux);

	}
	free(namelist);

	return 0;
}


