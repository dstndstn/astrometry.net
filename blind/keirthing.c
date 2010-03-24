
#include <stdio.h>

#include "bl.h"
#include "blind_wcs.h"
#include "sip.h"

static const char* OPTIONS = "hW:H:X:Y:";

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	dl* xys = dl_new(16);
	dl* radecs = dl_new(16);
	dl* otherradecs = dl_new(16);

	double* xy;
	double* xyz;
	int i, N;
	tan_t tan, tan2, tan3;
	int W=0, H=0;
	double crpix[] = { HUGE_VAL, HUGE_VAL };

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case 'h':
			exit(0);
		case 'W':
			W = atoi(optarg);
			break;
		case 'H':
			H = atoi(optarg);
			break;
		case 'X':
			crpix[0] = atof(optarg);
			break;
		case 'Y':
			crpix[1] = atof(optarg);
			break;
		}
	}
	if (optind != argc) {
		exit(-1);
	}

	if (W == 0 || H == 0) {
		printf("Need -W, -H\n");
		exit(-1);
	}
	if (crpix[0] == HUGE_VAL)
		crpix[0] = W/2.0;
	if (crpix[1] == HUGE_VAL)
		crpix[1] = H/2.0;

	while (1) {
		double x,y,ra,dec;
		if (fscanf(stdin, "%lf %lf %lf %lf\n", &x, &y, &ra, &dec) < 4)
			break;
		if (x == -1 && y == -1) {
			dl_append(otherradecs, ra);
			dl_append(otherradecs, dec);
		} else {
			dl_append(xys, x);
			dl_append(xys, y);
			dl_append(radecs, ra);
			dl_append(radecs, dec);
		}
	}
	printf("Read %i x,y,ra,dec tuples\n", dl_size(xys)/2);

	N = dl_size(xys)/2;
	xy = dl_to_array(xys);
	xyz = malloc(3 * N * sizeof(double));
	for (i=0; i<N; i++)
		radecdeg2xyzarr(dl_get(radecs, 2*i), dl_get(radecs, 2*i+1), xyz + i*3);
	dl_free(xys);
	dl_free(radecs);

	blind_wcs_compute(xyz, xy, N, &tan, NULL);

	printf("Computed TAN WCS:\n");
	tan_print(&tan);

	blind_wcs_move_tangent_point(xyz, xy, N, crpix, &tan, &tan2);
	blind_wcs_move_tangent_point(xyz, xy, N, crpix, &tan2, &tan3);
	printf("Moved tangent point to (%g,%g):\n", crpix[0], crpix[1]);
	tan_print(&tan3);

	for (i=0; i<dl_size(otherradecs)/2; i++) {
		double ra, dec, x,y;
		ra = dl_get(otherradecs, 2*i);
		dec = dl_get(otherradecs, 2*i+1);
		if (!tan_radec2pixelxy(&tan3, ra, dec, &x, &y)) {
			printf("Not in tangent plane: %g,%g\n", ra, dec);
			exit(-1);
			//continue;
		}
		printf("%g %g\n", x, y);
	}
	free(xy);
	free(xyz);
	return 0;
}



