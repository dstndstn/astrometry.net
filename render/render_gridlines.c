#include <stdio.h>
#include <math.h>
#include <sys/param.h>

#include "tilerender.h"
#include "render_gridlines.h"

int render_gridlines(unsigned char* img, render_args_t* args) {
	double rastep, decstep;
	int ind;
	double steps[] = { 1, 20.0, 10.0, 6.0, 4.0, 2.5, 1.0, 30./60.0,
					   15.0/60.0, 10.0/60.0, 5.0/60.0, 2./60.0 };
	double ra, dec;
	int i;

	ind = MAX(1, args->zoomlevel);
	ind = MIN(ind, sizeof(steps)/sizeof(double)-1);
	rastep = decstep = steps[ind];

	for (ra = rastep * floor(args->ramin / rastep);
		 ra <= rastep * ceil(args->ramax / rastep);
		 ra += rastep) {
		int x = ra2pixel(ra, args);
		if (!in_image(x, 0, args))
			continue;
		for (i=0; i<args->H; i++) {
			uchar* pix = pixel(x, i, img, args);
			pix[0] = 200;
			pix[1] = 200;
			pix[2] = 255;
			pix[3] = 45;
		}
	}
	for (dec = decstep * floor(args->decmin / decstep);
		 dec <= decstep * ceil(args->decmax / decstep);
		 dec += decstep) {
		int y = dec2pixel(dec, args);
		if (!in_image(0, y, args))
			continue;
		for (i=0; i<args->W; i++) {
			uchar* pix = pixel(i, y, img, args);
			pix[0] = 200;
			pix[1] = 200;
			pix[2] = 255;
			pix[3] = 45;
		}
	}

	return 0;
}
