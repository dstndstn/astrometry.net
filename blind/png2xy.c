/*
 * extract sources from jpg and pngs and produces a csv + image
 * not intended to use fits machinery
 *
 * keir mierle 2007
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <regex.h>
#include <cairo.h>
#include <math.h>
#include <getopt.h>

#include "cairoutils.h"
#include "dimage.h"

#define PEAKDIST 4

int is_png(const char *name) {
	regex_t preq;
	regcomp(&preq, "[^ ]*[.](png|PNG)$", REG_EXTENDED);
	return !regexec(&preq, name, (size_t) 0, NULL, 0);
}

int is_jpeg(const char *name) {
	regex_t preq;
	regcomp(&preq, "[^ ]*[.](jpg|JPG|jpeg|JPEG)$", REG_EXTENDED);
	return !regexec(&preq, name, (size_t) 0, NULL, 0);
}

int is_image(const char *name) {
	return is_jpeg(name) || is_png(name);
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

char* OPTIONS = "ht:m:g:n:N:";

void print_help_and_exit(char* progname) {
	printf("print out a track matrix for a reconstruction\n");
	printf("usage: %s -t <track_file.ts> -m <reconstruction.mr> -e <ground_truth.exr>\n", progname);
	printf("  -h           this message\n");
	printf("  -t<track>    track data\n");
	printf("  -m<metric>   metric reconstruction to evaluate\n");
	printf("  -g<truth>    ground truth to evaluate against\n");
	printf("  -n<int>      first frame of ground truth\n");
	printf("  -N<int>      last frame of ground truth\n");
	exit(-1);
}

void exit_error(char* progname, char *msg) {
	fprintf(stderr, msg);
	print_help_and_exit(progname);
}
#define ERROR(msg) exit_error(argv[0], "ERROR: " msg "\n")

// getopt needs constant flag addresses
int with_flux=0;

int main(int argc, char *argv[]) {
	struct dirent **namelist;
	int i, n, N, peak;
	unsigned char *image = NULL;
	float *image_bw_f;
	unsigned char *image_bw_u8;
	unsigned char *image_out;
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

	char *fname = argv[1];
	int c;

	dpsf = 1.0;
	plim = 8.0;
	dlim = dpsf;
	saddle = 5.0;
	maxper = 1000;
	maxsize = 1000;
	halfbox = 100;
	maxnpeaks = 10000;

	enum {
		MAXPER = 1,
		MAXNPEAKS,
		MAXSIZE,
		HALFBOX,
		DLIM,
		DPSF,
		SADDLE,
		PLIM,
	};
	while (1)
	{
		static struct option long_options[] = {
			{"flux",                  no_argument, &with_flux,  1      },
			{"max-peaks-per-object",  required_argument, 0, MAXPER    },
			{"max-peaks",             required_argument, 0, MAXNPEAKS },
			{"max-peaks-size",        required_argument, 0, MAXSIZE   },
			{"median-filter-radius",  required_argument, 0, HALFBOX   },
			{"min-dist",              required_argument, 0, DLIM      },
			{"psf",                   required_argument, 0, DPSF      },
			{"saddle",                required_argument, 0, SADDLE    },
			{"sigmas",                required_argument, 0, PLIM      },
			{0, 0, 0, 0}
		};
		int option_index = 0;
		c = getopt_long (argc, argv, "", long_options, &option_index);

		if (c == -1)
			break;

		switch (c) {
			case 0         : break; /* do nothing on no-arg flags */
			case MAXPER    : maxper     = atof(optarg); break;
			case MAXNPEAKS : maxnpeaks  = atoi(optarg); break;
			case MAXSIZE   : maxsize    = atoi(optarg); break;
			case HALFBOX   : halfbox    = atoi(optarg); break;
			case DLIM      : dlim       = atof(optarg); break;
			case DPSF      : dpsf       = atof(optarg); break;
			case SADDLE    : saddle     = atof(optarg); break;
			case PLIM      : plim       = atof(optarg); break;
			default:
							 abort ();
		}
	}

	if (optind != argc-1) {
		fprintf(stderr, "usage: png2xy image.[png|jpg]\n");
		exit(1);
	}

	fname = argv[optind];

	if (!is_image(fname)) {
		fprintf(stderr, "not an image.\n");
		exit(1);
	}

	if (is_png(fname)) {
		fprintf(stderr, "as a PNG\n");
		image = cairoutils_read_png(fname, &imW, &imH);
	} else if (is_jpeg(fname)) {
		fprintf(stderr, "as a JPEG\n");
		image = cairoutils_read_jpeg(fname, &imW, &imH);
	} 

	image_bw_u8 = to_bw_u8(image, imW, imH);

	x = malloc(maxnpeaks * sizeof(float));
	y = malloc(maxnpeaks * sizeof(float));
	flux = malloc(maxnpeaks * sizeof(float));

	simplexy_u8(image_bw_u8, imW, imH, dpsf, plim, dlim, saddle, maxper,
			maxnpeaks, maxsize, halfbox, &sigma, x, y, flux, &npeaks, 1);

	if (1) {
		image_out = malloc(sizeof(unsigned char) * imW * imH * 4);
		memcpy(image_out, image, imW*imH*4);

		/* draw the peaks */
		for (peak = 0; peak < npeaks; peak++) {
			for (peakXi = -PEAKDIST; peakXi <= PEAKDIST; peakXi++) {
				for (peakYi = -PEAKDIST; peakYi <= PEAKDIST; peakYi++) {
					peakX = ((int)round(x[peak])) + peakXi;
					peakY = ((int)round(y[peak])) + peakYi;
					peakDist = peakXi * peakXi + peakYi * peakYi;

					if (abs(peakDist - PEAKDIST*PEAKDIST) > 3 ||
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
		cairoutils_write_png("out.png", image_out, imW, imH);
		free(image_out);
	}

	/* output to CSV */
	for (i=0; i<npeaks; i++) {
		if (with_flux)
			printf("%g,%g,%g\n", x[i],y[i],flux[i]);
		else
			printf("%g,%g\n", x[i],y[i]);
	}

	free(image);
	free(image_bw_u8);
	free(x);
	free(y);
	free(flux);

	return 0;
}


