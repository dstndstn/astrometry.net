/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <math.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include "keywords.h"
#include "starutil.h"
#include "catalog.h"
#include "an-catalog.h"
#include "usnob-fits.h"
#include "tycho2-fits.h"
#include "mathutil.h"
#include "rdlist.h"
#include "boilerplate.h"
#include "starkd.h"

#define OPTIONS "bhgN:f:tsS"

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [-b] [-h] [-g] [-N imsize]"
		   " <filename> [<filename> ...] > outfile.pgm\n"
		   "  -h sets Hammer-Aitoff (default is an equal-area, positive-Z projection)\n"
	       "  -S squishes Hammer-Aitoff projection to make an ellipse; height becomes N/2.\n"
		   "  -b sets reverse (negative-Z projection)\n"
		   "  -g adds RA,DEC grid\n"
		   "  -N sets edge size (width) of output image\n"
		   "  [-s]: write two-byte-per-pixel PGM (default is one-byte-per-pixel)\n"
		   "  [-f <field-num>]: for RA,Dec lists (rdls), which field to use (default: all)\n\n"
		   "  [-L <field-range-low>]\n"
		   "  [-H <field-range-high>]\n"
		   "\n"
		   "  [-t]: for USNOB inputs, include Tycho-2 stars, even though their format isn't quite right.\n"
		   "\n"
		   "Can read Tycho2.fits, USNOB.fits, AN.fits, AN.objs.fits, AN.skdt.fits, and rdls.fits files.\n"
		   "\n", progname);
}
		   
extern char *optarg;
extern int optind, opterr, optopt;

#define PI M_PI

Inline void getxy(double px, double py, int W, int H,
				  int* X, int* Y) {
	px = 0.5 + (px - 0.5) * 0.99;
	py = 0.5 + (py - 0.5) * 0.99;
	*X = (int)nearbyint(px * W);
	*Y = (int)nearbyint(py * H);
}

void add_ink(double* xyz, int hammer, int reverse,
	     int* backside, int W, int H,
	     double* projection, int value) {
  double px, py;
  int X,Y;
  if (!hammer) {
    double z = xyz[2];
    if ((z <= 0 && !reverse) || (z >= 0 && reverse)) {
      (*backside)++;
      return;
    }
    if (reverse)
      z = -z;
    project_equal_area(xyz[0], xyz[1], z, &px, &py);
  } else {
    /* Hammer-Aitoff projection */
    project_hammer_aitoff_x(xyz[0], xyz[1], xyz[2], &px, &py);
  }
  getxy(px, py, W, H, &X, &Y);
  if (value)
    projection[X+W*Y] = value;
  else
    projection[X+W*Y]++;
}

int main(int argc, char *argv[])
{
  double *projection;
	char* progname = argv[0];
	uint ii,jj,numstars=0;
	int reverse=0, hammer=0, grid=0;
	int maxval;
	char* fname = NULL;
	int argchar;
	FILE* fid;
	qfits_header* hdr;
	char* valstr;
	int BLOCK = 100000;
	catalog* cat;
	startree_t* skdt;
	an_catalog* ancat;
	usnob_fits* usnob;
	tycho2_fits* tycho;
	rd_t* rd;
	il* fields;
	int backside = 0;
	int W = 3000, H;
	unsigned char* img;
	  double xyz[3];

	int fieldslow = -1;
	int fieldshigh = -1;
	int notycho = 1;
	int imgmax = 255;
	int squish = 0;

	fields = il_new(32);

	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 's':
			imgmax = 65535;
			break;
		case 'S':
		  squish = 1;
		  break;
		case 'b':
			reverse = 1;
			break;
		case 'h':
			hammer = 1;
			break;
		case 'g':
			grid = 1;
			break;
		case 'N':
		  W=(int)strtoul(optarg, NULL, 0);
		  break;
		case 'f':
			il_append(fields, atoi(optarg));
			break;
		case 'L':
			fieldslow = atoi(optarg);
			break;
		case 'H':
			fieldshigh = atoi(optarg);
			break;
		case 't':
			notycho = 0;
			break;
		default:
			return (OPT_ERR);
		}

	if (optind == argc) {
		printHelp(progname);
		exit(-1);
	}

	if (((fieldslow == -1) && (fieldshigh != -1)) ||
		((fieldslow != -1) && (fieldshigh == -1)) ||
		(fieldslow > fieldshigh)) {
		printHelp(progname);
		fprintf(stderr, "If you specify -L you must also specify -H.\n");
		exit(-1);
	}
	if (fieldslow != -1) {
		int f;
		for (f=fieldslow; f<=fieldshigh; f++)
			il_append(fields, f);
	}

	if (squish)
	  H = W/2;
	else
	  H = W;

	projection=calloc(sizeof(double), W*H);

	for (; optind<argc; optind++) {
		int i;
		char* key;
		cat = NULL;
		skdt = NULL;
		ancat = NULL;
		usnob = NULL;
		tycho = NULL;
		rd = NULL;
		numstars = 0;
		fname = argv[optind];
		fprintf(stderr, "Reading file %s...\n", fname);
		fid = fopen(fname, "rb");
		if (!fid) {
			fprintf(stderr, "Couldn't open file %s.  (Specify the complete filename with -f <filename>)\n", fname);
			exit(-1);
		}
		fclose(fid);

		hdr = qfits_header_read(fname);
		if (!hdr) {
			fprintf(stderr, "Couldn't read FITS header from file %s.\n", fname);
			exit(-1);
		}
		// look for AN_FILE (Astrometry.net filetype) in the FITS header.
		valstr = qfits_pretty_string(qfits_header_getstr(hdr, "AN_FILE"));
		if (valstr) {
			fprintf(stderr, "Astrometry.net file type: \"%s\".\n", valstr);

            if (strncasecmp(valstr, AN_FILETYPE_USNOB, strlen(AN_FILETYPE_USNOB)) == 0) {
				fprintf(stderr, "Looks like a USNOB file.\n");
                usnob = usnob_fits_open(fname);
                if (!usnob) {
                    fprintf(stderr, "Couldn't open catalog.\n");
                    exit(-1);
                }
                numstars = usnob_fits_count_entries(usnob);
            } else if (strncasecmp(valstr, AN_FILETYPE_TYCHO2, strlen(AN_FILETYPE_TYCHO2)) == 0) {
                fprintf(stderr, "Looks like a Tycho-2 file.\n");
                tycho = tycho2_fits_open(fname);
                if (!tycho) {
                    fprintf(stderr, "Couldn't open catalog.\n");
                    exit(-1);
                }
                numstars = tycho2_fits_count_entries(tycho);
            } else if (strncasecmp(valstr, AN_FILETYPE_CATALOG, strlen(AN_FILETYPE_CATALOG)) == 0) {
				fprintf(stderr, "Looks like a catalog.\n");
				cat = catalog_open(fname);
				if (!cat) {
					fprintf(stderr, "Couldn't open catalog.\n");
					return 1;
				}
				numstars = cat->numstars;
			} else if (strncasecmp(valstr, AN_FILETYPE_STARTREE, strlen(AN_FILETYPE_STARTREE)) == 0) {
				fprintf(stderr, "Looks like a star kdtree.\n");
				skdt = startree_open(fname);
				if (!skdt) {
					fprintf(stderr, "Couldn't open star kdtree.\n");
					return 1;
				}
				numstars = startree_N(skdt);
			} else if (strncasecmp(valstr, AN_FILETYPE_RDLS, strlen(AN_FILETYPE_RDLS)) == 0) {
                pl* rds;
				rdlist_t* rdls;
				int nfields, f;
                int ntotal;
				fprintf(stderr, "Looks like an rdls (RA,DEC list)\n");
				rdls = rdlist_open(fname);
				if (!rdls) {
					fprintf(stderr, "Couldn't open RDLS file.\n");
					return 1;
				}
				nfields = il_size(fields);
				if (!nfields) {
					nfields = rdlist_n_fields(rdls);
					fprintf(stderr, "Plotting all %i fields.\n", nfields);
				}
                rds = pl_new(nfields);
                ntotal = 0;
				for (f=1; f<=nfields; f++) {
					int fld;
                    rd_t* thisrd;
					if (il_size(fields))
						fld = il_get(fields, f-1);
					else
						fld = f;
                    thisrd = rdlist_read_field_num(rdls, fld, NULL);
                    if (!thisrd) {
						fprintf(stderr, "Failed to open extension %i in RDLS file %s.\n", fld, fname);
                        return 1;
                    }
					fprintf(stderr, "Field %i has %i entries.\n", fld, rd_n(thisrd));
                    ntotal += rd_n(thisrd);
                    pl_append(rds, thisrd);
				}
				rdlist_close(rdls);
				numstars = ntotal;
                // merge all the rd_t data.
                if (pl_size(rds) == 1) {
                    rd = pl_get(rds, 0);
                } else {
                    int j;
                    int nsofar = 0;
                    rd = rd_alloc(ntotal);
                    for (j=0; j<pl_size(rds); j++) {
                        rd_t* thisrd = pl_get(rds, j);
                        rd_copy(rd, nsofar, thisrd, 0, rd_n(thisrd));
                        nsofar += rd_n(thisrd);
                        rd_free(thisrd);
                    }
                }
                pl_free(rds);

			} else {
				fprintf(stderr, "Unknown Astrometry.net file type: \"%s\".\n", valstr);
				exit(-1);
			}
		}
		// "AN_CATALOG" gets truncated...
		key = qfits_header_findmatch(hdr, "AN_CAT");
		if (key) {
			if (qfits_header_getboolean(hdr, key, 0)) {
				fprintf(stderr, "File has AN_CATALOG = T header.\n");
				ancat = an_catalog_open(fname);
				if (!ancat) {
					fprintf(stderr, "Couldn't open catalog.\n");
					exit(-1);
				}
				numstars = an_catalog_count_entries(ancat);
				an_catalog_set_blocksize(ancat, BLOCK);
			}
		}
		qfits_header_destroy(hdr);
		if (!(cat || skdt || ancat || usnob || tycho || rd)) {
			fprintf(stderr, "I can't figure out what kind of file %s is.\n", fname);
			exit(-1);
		}

		fprintf(stderr, "Reading %i stars...\n", numstars);

		for (i=0; i<numstars; i++) {
			if (is_power_of_two(i+1)) {
				if (backside) {
					fprintf(stderr, "%i stars project onto the opposite hemisphere.\n", backside);
				}
				fprintf(stderr,"  done %u/%u stars\r",i+1,numstars);
			}

			if (cat) {
			  double* sxyz;
			  sxyz = catalog_get_star(cat, i);
			  xyz[0] = sxyz[0];
			  xyz[1] = sxyz[1];
			  xyz[2] = sxyz[2];
			} else if (skdt) {
				if (startree_get(skdt, i, xyz)) {
					fprintf(stderr, "Failed to read star %i from star kdtree.\n", i);
					exit(-1);
				}
			} else if (rd) {
				double ra, dec;
				ra  = rd_getra (rd, i);
				dec = rd_getdec(rd, i);
				radecdeg2xyzarr(ra, dec, xyz);
			} else if (ancat) {
			  an_entry* entry = an_catalog_read_entry(ancat);
			  radecdeg2xyzarr(entry->ra, entry->dec, xyz);
			} else if (usnob) {
			  usnob_entry* entry = usnob_fits_read_entry(usnob);
			  if (notycho && (entry->ndetections == 0))
			    continue;
			  radecdeg2xyzarr(entry->ra, entry->dec, xyz);
			} else if (tycho) {
			  tycho2_entry* entry = tycho2_fits_read_entry(tycho);
			  radecdeg2xyzarr(entry->ra, entry->dec, xyz);
			}

			add_ink(xyz, hammer, reverse, &backside, W, H, projection, 0);

		}

		if (cat)
			catalog_close(cat);
		if (skdt)
			startree_close(skdt);
		if (rd)
            rd_free(rd);
		if (ancat)
			an_catalog_close(ancat);
		if (usnob)
			usnob_fits_close(usnob);
		if (tycho)
			tycho2_fits_close(tycho);
	}

	maxval = 0;
	for (ii = 0; ii < (W*H); ii++)
		if (projection[ii] > maxval)
			maxval = projection[ii];

	if (grid) {
		/* Draw a line for ra=-160...+160 in 10 degree sections */
		for (ii=-160; ii <= 160; ii+= 10) {
			int RES = 10000;
			for (jj=-RES; jj<RES; jj++) {
				/* Draw a bunch of points for dec -90...90*/

				double ra = deg2rad(ii);
				double dec = jj/(double)RES * PI/2.0;
				radec2xyzarr(ra, dec, xyz);

				add_ink(xyz, hammer, reverse, &backside, W, H, projection, maxval);
			}
		}
		/* Draw a line for dec=-80...+80 in 10 degree sections */
		for (ii=-80; ii <= 80; ii+= 10) {
			int RES = 10000;
			for (jj=-RES; jj<RES; jj++) {
				/* Draw a bunch of points for dec -90...90*/

				double ra = jj/(double)RES * PI;
				double dec = deg2rad(ii);
				radec2xyzarr(ra, dec, xyz);

				add_ink(xyz, hammer, reverse, &backside, W, H, projection, maxval);
			}
		}
	}

	// Output PGM format
	printf("P5 %d %d %d\n", W, H, imgmax);
	// hack - we reuse the "projection" storage to store the image data:
	if (imgmax == 255) {
		img = (unsigned char*)projection;
		for (ii=0; ii<(W*H); ii++)
			img[ii] = (int)((double)imgmax * projection[ii] / (double)maxval);
		if (fwrite(img, 1, W*H, stdout) != (W*H)) {
			fprintf(stderr, "Failed to write image: %s\n", strerror(errno));
			exit(-1);
		}
	} else {
		uint16_t* img = (uint16_t*)projection;
		for (ii=0; ii<(W*H); ii++) {
			img[ii] = (int)((double)imgmax * projection[ii] / (double)maxval);
			img[ii] = htons(img[ii]);
		}
		if (fwrite(img, 2, W*H, stdout) != (W*H)) {
			fprintf(stderr, "Failed to write image: %s\n", strerror(errno));
			exit(-1);
		}
	}
	free(projection);

	il_free(fields);

	return 0;
}
