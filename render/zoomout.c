/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "starutil.h"
#include "pnmutils.h"

#define OPTIONS "hr:d:W:H:z:s:e:o:gnipwbB:"

/*
  eg:

zoomout -r 0 -d 0 -W 512 -H 512 -z 241 -s 12 -e 1 -o /tmp/frame%03i.ppm -w -b;
for x in /tmp/frame*.ppm; do echo $x; ppmquant 256 $x | ppmtogif > `basename $x .ppm`.gif; done
gifsicle -d 10 -O -o zoom.gif -l=1 frame*.gif

 */

static void printHelp(char* progname) {
	fprintf(stderr, "usage: %s\n"
			"    -r <RA>  in degrees\n"
			"    -d <DEC> in degrees\n"
			"    [-W <width>] in pixels, default 256\n"
			"    [-H <height>] in pixels, default 256\n"
			"    [-z <zoom steps>] number of animation frames, default 10, must be >=2\n"
			"    [-s <start-zoom>] default 15\n"
			"    [-e <end-zoom>] default 1\n"
			"    -o <output-file-template> in printf format, given frame number, eg, \"frame%%03i.ppm\".\n"
			"    [-g]  output GIF format\n"
			"    [-n]  just print the command-lines, don't execute them.\n"
			"    [-i]  force rendering from pre-rendered images\n"
			"    [-p]  force direct plotting from catalog\n"
			"    [-w]  do automatic white-balancing\n"
			"    [-b]  add box\n"
			"    [-b]  (specify -b twice to get power-of-10 boxes)\n"
			"    [-B <color>]  set box color\n"
			"\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *args[]) {
    int argchar;
	char* progname = args[0];

	char* outfn = NULL;
	double ra=-1.0, dec=-1.0;
	int W = 256;
	int H = 256;
	int zoomsteps = 10;
	double startzoom = 15.0;
	double endzoom = 15.0;

	double dzoom;
	int i;
	double ucenter, vcenter;
	bool gif = FALSE;
	bool justprint = FALSE;
	bool forceimg = FALSE;
	bool forcecat = FALSE;
	bool whitebalance = FALSE;
	bool whitebalframe = FALSE;
	bool donewhitebalframe = FALSE;
	char* lastfn = NULL;
	double Rgain, Ggain, Bgain;
	char* boxcolor = "red";
	bool dobox = FALSE;
	bool doboxes = FALSE;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'b':
			if (dobox)
				doboxes = TRUE;
			dobox = TRUE;
			break;
		case 'B':
			boxcolor = optarg;
			break;
		case 'w':
			whitebalance = TRUE;
			break;
		case 'i':
			forceimg = TRUE;
			break;
		case 'p':
			forcecat = TRUE;
			break;
		case 'h':
			printHelp(progname);
			exit(0);
		case 'n':
			justprint = TRUE;
			break;
		case 'g':
			gif = TRUE;
			break;
		case 'r':
			ra = atof(optarg);
			break;
		case 'd':
			dec = atof(optarg);
			break;
		case 'W':
			W = atoi(optarg);
			break;
		case 'H':
			H = atoi(optarg);
			break;
		case 'z':
			zoomsteps = atoi(optarg);
			break;
		case 's':
			startzoom = atof(optarg);
			break;
		case 'e':
			endzoom = atof(optarg);
			break;
		case 'o':
			outfn = optarg;
			break;
		}

	if (!outfn) {
		printHelp(progname);
		exit(-1);
	}

	if (zoomsteps < 1) {
		printHelp(progname);
		exit(-1);
	}

	if (zoomsteps > 1)
		dzoom = (endzoom - startzoom) / (zoomsteps - 1);
	else
		dzoom = 0.0;

	ra = deg2rad(ra);
	dec = deg2rad(dec);

	// mercator center.
	ucenter = ra / (2.0 * M_PI);
	while (ucenter < 0.0) ucenter += 1.0;
	while (ucenter > 1.0) ucenter -= 1.0;
	vcenter = asinh(tan(dec));

	for (i=0; i<zoomsteps; i++) {
		double zoom = startzoom + i * dzoom;
		double zoomscale = pow(2.0, 1.0 - zoom);
		char cmdline[1024];
		// My "scale 1" image is 512x512.
		double uscale = 1.0 / 512.0;
		double vscale = (2.0*M_PI) / 512.0;
		double u1, u2, v1, v2;
		double ra1, ra2, dec1, dec2;
		int res;
		char* tempimg = "/tmp/tmpimg.ppm";
		char outfile[256];

		printf("\nZoom %g\n", zoom);

		uscale *= zoomscale;
		vscale *= zoomscale;

		// CHECK BOUNDS!

		u1 = ucenter - W/2 * uscale;
		u2 = ucenter + W/2 * uscale;
		v1 = vcenter - H/2 * vscale;
		v2 = vcenter + H/2 * vscale;

		//if ((zoom <= 5.0) || forceimg) {
		if ((!forcecat && (zoom <= 4.0)) || forceimg || whitebalframe) {
			int zi;
			char* map_template = "/home/gmaps/usnob-images/maps/usnob-zoom%i.ppm";
			char fn[256];
			int left, right, top, bottom;
			double scaleadj;
			int pixelsize;
			/*
			  zi = (int)ceil(zoom);
			  if (zi < 1) zi = 1;
			  if (zi > 5) zi = 5;
			*/

			if (whitebalance && !whitebalframe && !donewhitebalframe) {
				// re-render the previous zoom step from images rather than from catalog.
				whitebalframe = TRUE;
				i-=2;
				continue;
			}

			zi = 5;
			printf("Zoom %g => %i\n", zoom, zi);
			sprintf(fn, map_template, zi);
			scaleadj = pow(2.0, (double)zi - zoom);
			printf("Scale adjustment %g\n", scaleadj);

			pixelsize = (int)rint(pow(2.0, zi)) * 256;

			printf("Pixelsize %i\n", pixelsize);

			printf("u range [%g, %g], v range [%g, %g].\n", u1, u2, v1, v2);

			left   = (int)rint(u1 * pixelsize);
			right  = (int)rint(u2 * pixelsize);
			top    = (int)rint((v1 + M_PI) / (2.0 * M_PI) * pixelsize);
			bottom = (int)rint((v2 + M_PI) / (2.0 * M_PI) * pixelsize);

			printf("L %i, R %i, T %i, B %i\n", left, right, top, bottom);

			if (whitebalframe) {
				sprintf(outfile, "/tmp/whitebal.ppm");
			} else {
				sprintf(outfile, outfn, i);
			}

			if (left >= 0 && right < pixelsize && top >= 0 && bottom < pixelsize) {
				printf("Cutting and scaling...\n");
				if (whitebalance && !whitebalframe) {
					sprintf(cmdline, "pnmcut -left %i -right %i -top %i -bottom %i %s "
							"| pnmscale -width=%i -height=%i - "
							"| pnmflip -tb > %s; "
							"ppmnormrgb -r %g -g %g -b %g %s > %s",
							left, right, top, bottom, fn,
							W, H, tempimg,
							Rgain, Ggain, Bgain, tempimg, outfile);
				} else {
					sprintf(cmdline, "pnmcut -left %i -right %i -top %i -bottom %i %s "
							"| pnmscale -width=%i -height=%i - "
							"| pnmflip -tb > %s",
							left, right, top, bottom, fn, W, H, outfile);
				}

				printf("cmdline: %s\n", cmdline);
				if (!justprint) {
					if ((res = system(cmdline)) == -1) {
						fprintf(stderr, "system() call failed.\n");
						exit(-1);
					}
					if (res) {
						fprintf(stderr, "command line returned a non-zero value.  Quitting.\n");
						break;
					}
				}

				if (whitebalframe) {
					if (pnmutils_whitebalance(lastfn, outfile, &Rgain, &Ggain, &Bgain)) {
						exit(-1);
					}
					printf("Gains: R %g, G %g, B %g\n", Rgain, Ggain, Bgain);
					donewhitebalframe = TRUE;
					whitebalframe = FALSE;
				}

			} else {
				// we've got to paste together the image...
				int L, R;
				int T, B;
				int SX, SY;

				sprintf(cmdline, "ppmmake black %i %i > %s", W, H, outfile);
				printf("cmdline: %s\n", cmdline);
				if (!justprint)
					system(cmdline);

				//if (left < 0) {
				T = (top >= 0 ? top : 0);
				B = (bottom < pixelsize ? bottom : pixelsize-1);
				//T = top;
				//B = bottom;
				L = left;
				R = right;
				SX = 0;
				while (L < right) {
					int pixL, pixR;
					int SW, SH;

					printf("L=%i, R=%i.\n", L, R);

					pixL = ((L % pixelsize) + pixelsize) % pixelsize;
					/*
					  if (L < 0)
					  pixR = pixelsize-1;
					  else
					*/
					pixR = pixL + (R-L);
					pixR = (pixR >= pixelsize ? pixelsize-1 : pixR);
					printf("grabbing %i to %i\n", pixL, pixR);
					SW = rint((1 + pixR - pixL) / scaleadj);
					SH = (1 + B - T) / scaleadj;
					//SX = (L - left) / scaleadj;
					SY = (T - top) / scaleadj;
					if (SX + SW > W) {
						printf("Adjusting width from %i to %i.\n", SW, W-SX);
						SW = W - SX;
					}
					if (SY + SH > H) {
						printf("Adjusting height from %i to %i.\n", SH, H-SY);
						SH = H - SY;
					}
					sprintf(cmdline, "pnmcut -left %i -right %i -top %i -bottom %i %s "
							"| pnmscale -width=%i -height=%i - "
							"| pnmflip -tb "
							"| pnmpaste - %i %i %s > %s; "
							"mv %s %s",
							pixL, pixR, T, B, fn,
							SW, SH,
							SX, SY, outfile, tempimg,
							tempimg, outfile);
					printf("cmdline: %s\n", cmdline);
					if (!justprint)
						system(cmdline);

					SX += SW;
					L += (pixR - pixL + 1);
				}

				if (whitebalance && !whitebalframe) {
					sprintf(cmdline, "ppmnormrgb -r %g -g %g -b %g %s > %s; "
							"mv %s %s",
							Rgain, Ggain, Bgain, outfile, tempimg,
							tempimg, outfile);
					printf("cmdline: %s\n", cmdline);
					if (!justprint)
						system(cmdline);
				}

				if (whitebalframe) {
					if (pnmutils_whitebalance(lastfn, outfile, &Rgain, &Ggain, &Bgain)) {
						exit(-1);
					}
					printf("Gains: R %g, G %g, B %g\n", Rgain, Ggain, Bgain);
					donewhitebalframe = TRUE;
					whitebalframe = FALSE;
				}
			}

		} else {
			ra1 = u1 * 2.0 * M_PI;
			ra2 = u2 * 2.0 * M_PI;
			dec1 = atan(sinh(v1));
			dec2 = atan(sinh(v2));

			ra1 = rad2deg(ra1);
			ra2 = rad2deg(ra2);
			dec1 = rad2deg(dec1);
			dec2 = rad2deg(dec2);

			printf("Zoom %g, scale %g, u range [%g,%g], v range [%g,%g], RA [%g,%g], DEC [%g,%g] degrees\n",
				   zoom, zoomscale, u1, u2, v1, v2, ra1, ra2, dec1, dec2);

			sprintf(outfile, outfn, i);

			sprintf(cmdline, "usnobtile -f -x %g, -X %g, -y %g, -Y %g -w %i -h %i %s> %s",
					ra1, ra2, dec1, dec2, W, H, (gif ? "| pnmquant 256 | ppmtogif " : ""), outfile);

			printf("cmdline: %s\n", cmdline);

			if (!justprint) {
				if ((res = system(cmdline)) == -1) {
					fprintf(stderr, "system() call failed.\n");
					exit(-1);
				}
				if (res) {
					fprintf(stderr, "command line returned a non-zero value.  Quitting.\n");
					break;
				}
			}

			free(lastfn);
			lastfn = strdup(outfile);
		}



		// outline the initial box's position in this frame.
		if (dobox) {
			double step = 10.0;
			int left, top;
			double dscale = pow(2.0, i * dzoom);
			for (;;) {
				int width, height;
				printf("dscale %g\n", dscale);
				if (dscale > 1.0)
					break;
				left   = (int)rint(W/2 - dscale * W/2);
				top    = (int)rint(H/2 - dscale * H/2);
				width   = (int)rint(dscale * (W-1));
				height  = (int)rint(dscale * (H-1));

				sprintf(cmdline, "ppmdraw -script \"setcolor %s; setpos %i %i; line_here %i 0; line_here 0 %i; line_here %i 0; line_here 0 %i\" %s > %s; mv %s %s",
						boxcolor, left, top, width, height, -width, -height, outfile, tempimg, tempimg, outfile);
				printf("cmdline: %s\n", cmdline);
				if (!justprint) {
					system(cmdline);
				}
				dscale *= step;
				if (!doboxes)
					break;
			}
		}



	}

	free(lastfn);
	return 0;
}
