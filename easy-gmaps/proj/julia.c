#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/param.h>

const char* OPTIONS = "x:X:y:Y:W:H:s:S:";

extern char *optarg;
extern int optind, opterr, optopt;

static void heatmap(unsigned char pix, unsigned char* outpix) {
    float inpix = pix;
	if (inpix <= 96.0) {
		outpix[0] = inpix * 255.0 / 96.0;
		outpix[1] = 0;
		outpix[2] = 0;
	} else if (inpix <= 192.0) {
		outpix[0] = 255;
		outpix[1] = (inpix - 96.0) * 255.0 / 96.0;
		outpix[2] = 0;
	} else {
		outpix[0] = 255;
		outpix[1] = 255;
		outpix[2] = (inpix - 192.0) * 255.0 / 63.0;
	}
}

int main(int argc, char** args) {
    int argchar;
    double xmin=0, xmax=0, ymin=0, ymax=0;
    int W = 0, H = 0;
    double xstep, ystep;
    int i,j;
    unsigned char* img;
    double cx, cy;

    cx = -0.726895347709114071439;
    cy =  0.188887129043845954792;

	while ((argchar = getopt (argc, args, OPTIONS)) != -1)
		switch (argchar) {
        case 's':
            cx = atof(optarg);
            break;
        case 'S':
            cy = atof(optarg);
            break;
        case 'x':
            xmin = atof(optarg);
            break;
        case 'X':
            xmax = atof(optarg);
            break;
        case 'y':
            ymin = atof(optarg);
            break;
        case 'Y':
            ymax = atof(optarg);
            break;
        case 'W':
            W = atoi(optarg);
            break;
        case 'H':
            H = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Bad arg '%c'\n", argchar);
            exit(-1);
        }

    if (!W || !H) {
        fprintf(stderr, "Missing W,H\n");
        exit(-1);
    }
    if (xmin==0.0 && xmax==0.0) {
        fprintf(stderr, "Missing x or X\n");
        exit(-1);
    }
    if (ymin==0.0 && ymax==0.0) {
        fprintf(stderr, "Missing y or Y\n");
        exit(-1);
    }

    // rescale to the [-1, 1] box.
    xmin = xmin * 2.0 - 1.0;
    xmax = xmax * 2.0 - 1.0;
    ymin = ymin * 2.0 - 1.0;
    ymax = ymax * 2.0 - 1.0;



    xstep = (xmax - xmin) / (double)W;
    ystep = (ymax - ymin) / (double)H;

    img = malloc(3 * W * H);

    for (j=0; j<H; j++) {
        double x, y;
        double xn, yn;
        for (i=0; i<W; i++) {
            int k;
            x = xmin + xstep * i;
            y = ymin + ystep * j;

            for (k=0; k<254; k++) {
                xn = x*x - y*y + cx;
                yn = 2.0 * x * y + cy;
                x = xn;
                y = yn;
                if (x*x + y*y > 2.0)
                    break;
            }
            heatmap(k, img + 3*(j*W + i));
        }
    }

    printf("P6 %i %i\n%i\n", W, H, 255);
    fwrite(img, 1, 3*W*H, stdout);

    return 0;
}

