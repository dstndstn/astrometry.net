#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/param.h>

#include "tilerender.h"
#include "render_tycho.h"
#include "starutil.h"
#include "mathutil.h"
#include "merctree.h"
#include "keywords.h"
#include "mercrender.h"

static void logmsg(char* format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "render_tycho: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

int render_tycho(unsigned char* img, render_args_t* args) {
	float* fluximg = NULL;
	float amp = 0.0;
	int i, j;

	if (!args->tycho_mkdt) {
		logmsg("Required argument '-T <tycho-mkdt-path>' was not specified!");
		return -1;
	}

	fluximg = mercrender_file(args->tycho_mkdt, args, RENDERSYMBOL_psf);
	if (!fluximg) {
		logmsg("failed to read Tycho mkdt file \"%s\".\n", args->tycho_mkdt);
		return -1;
	}

	// brightness amplification factor
	amp = pow(4.0, MIN(5, args->zoomlevel)) * 32.0 * exp(args->gain * log(4.0));

	for (j=0; j<args->H; j++) {
		for (i=0; i<args->W; i++) {
			unsigned char* pix;
			double r, g, b, I, f, R, G, B, maxRGB;

			r = fluximg[3*(j*args->W + i)+0];
			b = fluximg[3*(j*args->W + i)+1];

			if (args->arith)
				g = (r + b) / 2.0;
			else
				g = sqrt(r * b);

			// color correction
			g *= sqrt(args->colorcor);
			b *= args->colorcor;
		
			I = (r + g + b) / 3;
			if (I == 0.0) {
				R = G = B = 0.0;
			} else {
				if (args->arc) {
					f = asinh(I * amp);
				} else {
					f = pow(I * amp, 0.25);
				}
				R = f*r/I;
				G = f*g/I;
				B = f*b/I;
				maxRGB = MAX(R, MAX(G, B));
				if (maxRGB > 1.0) {
					R /= maxRGB;
					G /= maxRGB;
					B /= maxRGB;
				}
			}
			pix = pixel(i, j, img, args);

			pix[0] = MIN(255, 255.0*R);
			pix[1] = MIN(255, 255.0*G);
			pix[2] = MIN(255, 255.0*B);
			pix[3] = 255;
		}
	}
	free(fluximg);

	logmsg("done.\n");

	return 0;
}

