/*
   This file is part of the Astrometry.net suite.
   Copyright 2007 Keir Mierle and Dustin Lang.

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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include <cairo.h>

#include "tilerender.h"
#include "render_messier.h"

static char* messier_fn = "messier.dat";
static char* mess_dirs[] = {
	".",
	"/home/gmaps/usnob-map/execs"
};

int render_messier(unsigned char* img, render_args_t* args) {
	cairo_t* cairo;
	cairo_surface_t* target;
	FILE* fmess = NULL;
	int i;

	target = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32,
												 args->W, args->H, args->W*4);
	cairo = cairo_create(target);
	cairo_set_source_rgb(cairo, 1.0, 1.0, 1.0);
	cairo_select_font_face(cairo, "helvetica", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
	cairo_set_font_size(cairo, 14.0);

	for (i=0; i<sizeof(mess_dirs)/sizeof(char*); i++) {
		char fn[256];
		snprintf(fn, sizeof(fn), "%s/%s", mess_dirs[i], messier_fn);
		fprintf(stderr, "render_messier: Trying messier file: %s\n", fn);
		fmess = fopen(fn, "rb");
		if (fmess)
			break;
	}
	if (!fmess) {
		fprintf(stderr, "render_messier: couldn't find messier data file.\n");
		return -1;
	}


	for (;;) {
		int res;
		char line[256];
		int mess, ngc;
		char* cptr;
		char* rptr;
		char conlong[32], conshort[16], type[16], subtype[16];
		char name[32];
		int rahrs;
		double ramins, ra;
		int decdegs, decmins;
		double dec;
		double mag;
		char diamstr[32];
		double dist;
		int nchars;
		int k;
		char printname[256];
		double px,py;
		int nchars_dec;

		if (!fgets(line, sizeof(line), fmess)) {
			if (feof(fmess))
				break;
			fprintf(stderr, "render_messier: error reading from messier file: %s\n", strerror(errno));
		}
		if (line[0] == '#')
			continue;

		cptr = line;
		if ((res = sscanf(cptr, "%d%*[ ?] %d %n", &mess, &ngc, &nchars)) != 2) {
			fprintf(stderr, "failed parsing mess, ngc: got %i\n", res);
			return -1;
		}
		cptr += nchars;

		// Long constellation name is enclosed in quotes ""
		if (*cptr != '"') {
			fprintf(stderr, "failed parsing name.\n");
			return -1;
		}
		cptr++;
		rptr = conlong;
		while (*cptr && *cptr != '"') {
			*rptr = *cptr;
			rptr++;
			cptr++;
		}
		if (!*cptr) {
			fprintf(stderr, "failed parsing name.\n");
			return -1;
		}
		*rptr = '\0';
		cptr++;

		if (sscanf(cptr, " %s %s %s %n", conshort, type, subtype, &nchars) != 3) {
			fprintf(stderr, "failed parsing shortname, type, subtype.\n");
			return -1;
		}
		cptr += nchars;

		if ((res = sscanf(cptr, "%d %lg %n%d %d %lg %s %lg %n",
						  &rahrs, &ramins, &nchars_dec, &decdegs, &decmins, &mag, diamstr, &dist, &nchars)) != 7) {
			fprintf(stderr, "failed parsing remainder: got %i.\n", res);
			return -1;
		}

		ra = (rahrs + ramins/60.0) * 15.0;
		if (cptr[nchars_dec] == '+') {
			dec = (decdegs + decmins/60.0);
		} else if (cptr[nchars_dec] == '-') {
			dec = (decdegs - decmins/60.0);
		} else {
			fprintf(stderr, "Failed parsing dec.\n");
			return -1;
		}

		for (k=0;; k++) {
			if ((cptr[nchars + k] == '\n') ||
				(cptr[nchars + k] == '\0')) {
				name[k] = '\0';
				break;
			}
			name[k] = cptr[nchars+k];
		}

		/*
		  fprintf(stderr, "ra: %g\n", ra);
		  fprintf(stderr, "dec: %g\n", dec);
		  fprintf(stderr, "ra hrs: %i\n", rahrs);
		  fprintf(stderr, "ra mins: %g\n", ramins);
		  fprintf(stderr, "dec degrees: %i\n", decdegs);
		  fprintf(stderr, "dec mins: %i\n", decmins);
		  fprintf(stderr, "mag: %g\n", mag);
		  fprintf(stderr, "diamstr: %s\n", diamstr);
		  fprintf(stderr, "dist: %g\n", dist);
		  fprintf(stderr, "name: \"%s\"\n", name);
		  fprintf(stderr, "\n");
		*/

		if (strlen(name)) {
			sprintf(printname, " M%i (%s)", mess, name);
		} else {
			sprintf(printname, " M%i", mess);
		}

		px = ra2pixel(ra, args);
		py = dec2pixel(dec, args);
		cairo_move_to(cairo, px, py);
		cairo_show_text(cairo, printname);
	}
	fclose(fmess);

	// Cairo's uint32 ARGB32 format is a little different than what we need,
	// which is uchar R,G,B,A.
	for (i=0; i<(args->H*args->W); i++) {
		unsigned char r,g,b,a;
		uint32_t ipix = *((uint32_t*)(img + 4*i));
		a = (ipix >> 24) & 0xff;
		r = (ipix >> 16) & 0xff;
		g = (ipix >>  8) & 0xff;
		b = (ipix      ) & 0xff;
		img[4*i + 0] = r;
		img[4*i + 1] = g;
		img[4*i + 2] = b;
		img[4*i + 3] = a;
	}

	cairo_surface_destroy(target);
	cairo_destroy(cairo);

	return 0;
}
