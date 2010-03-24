
#include <stdio.h>

#include "bl.h"
#include "blind_wcs.h"
#include "sip.h"

int main(int argc, char** args) {
	dl* xys = dl_new(16);
	dl* radecs = dl_new(16);

	double* xy;
	double* xyz;
	int i, N;
	tan_t tan;

	while (1) {
		double x,y,ra,dec;
		if (fscanf(stdin, "%lf %lf %lf %lf\n", &x, &y, &ra, &dec) < 4)
			break;
		dl_append(xys, x);
		dl_append(xys, y);
		dl_append(radecs, ra);
		dl_append(radecs, dec);
	}
	printf("Read %i x,y,ra,dec tuples\n", dl_size(xys)/2);

	N = dl_size(xys)/2;
	xy = dl_to_array(xys);
	//radec = dl_to_array(radecs);
	xyz = malloc(3 * N * sizeof(double));
	for (i=0; i<N; i++)
		radecdeg2xyzarr(dl_get(radecs, 2*i), dl_get(radecs, 2*i+1), xyz + i*3);
	dl_free(xys);
	dl_free(radecs);

	blind_wcs_compute(xyz, xy, N, &tan, NULL);

	printf("Computed TAN WCS:\n");
	tan_print(&tan);

	free(xy);
	free(xyz);
	return 0;
}



