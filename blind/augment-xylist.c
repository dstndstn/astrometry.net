/*
 This file is part of the Astrometry.net suite.
 Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

 The Astrometry.net suite is free software; you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA	 02110-1301 USA
 */

/**
 * Accepts an xylist and command-line options, and produces an augmented
 * xylist.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>
#include <getopt.h>

#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "scriptutils.h"
#include "sip_qfits.h"
#include "tabsort.h"
#include "errors.h"
#include "fits-guess-scale.h"

#include "qfits.h"

static const char* OPTIONS = "2AC:E:F:H:I:L:M:P:R:S:TV:W:X:Y:ac:d:e:fg:hi:k:m:o:rs:t:u:vw:x:z::";

static struct option long_options[] = {
	{"help",		   no_argument,	      0, 'h'},
	{"verbose",        no_argument,       0, 'v'},
	{"image",		   required_argument, 0, 'i'},
	{"xylist",		   required_argument, 0, 'x'},
	{"guess-scale",    no_argument,	      0, 'g'},
	{"cancel",		   required_argument, 0, 'C'},
	{"solved",		   required_argument, 0, 'S'},
	{"solved-in",      required_argument, 0, 'I'},
	{"match",		   required_argument, 0, 'M'},
	{"rdls",		   required_argument, 0, 'R'},
	{"wcs",			   required_argument, 0, 'W'},
	{"pnm",			   required_argument, 0, 'P'},
	{"force-ppm",	   no_argument,       0, 'f'},
	{"width",		   required_argument, 0, 'w'},
	{"height",		   required_argument, 0, 'e'},
	{"scale-low",	   required_argument, 0, 'L'},
	{"scale-high",	   required_argument, 0, 'H'},
	{"scale-units",    required_argument, 0, 'u'},
    {"fields",         required_argument, 0, 'F'},
	{"depth",		   required_argument, 0, 'd'},
	{"tweak-order",    required_argument, 0, 't'},
	{"out",			   required_argument, 0, 'o'},
	{"no-tweak",	   no_argument,	      0, 'T'},
	{"no-fits2fits",   no_argument,       0, '2'},
	{"temp-dir",       required_argument, 0, 'm'},
	{"x-column",       required_argument, 0, 'X'},
	{"y-column",       required_argument, 0, 'Y'},
    {"sort-column",    required_argument, 0, 's'},
    {"sort-ascending", no_argument,       0, 'a'},
    {"keep-xylist",    required_argument, 0, 'k'},
    {"verify",         required_argument, 0, 'V'},
    {"code-tolerance", required_argument, 0, 'c'},
    {"pixel-error",    required_argument, 0, 'E'},
    {"resort",         no_argument,       0, 'r'},
    {"downsample",     optional_argument, 0, 'z'},
    {"dont-augment",   no_argument,       0, 'A'},
	{0, 0, 0, 0}
};

static void print_help(const char* progname) {
	printf("Usage:	 %s [options] -o <output augmented xylist filename>\n"
		   "  (    -i <image-input-file>\n"
		   "   OR  -x <xylist-input-file>  )\n"
	       "  [--guess-scale]: try to guess the image scale from the FITS headers  (-g)\n"
           "  [--cancel <filename>]: filename whose creation signals the process to stop  (-C)\n"
           "  [--solved <filename>]: output filename for solved file  (-S)\n"
           "  [--solved-in <filename>]: input filename for solved file  (-I)\n"
           "  [--match  <filename>]: output filename for match file   (-M)\n"
           "  [--rdls   <filename>]: output filename for RDLS file    (-R)\n"
           "  [--wcs    <filename>]: output filename for WCS file     (-W)\n"
           "  [--pnm <filename>]: save the PNM file in <filename>  (-P)\n"
           "  [--force-ppm]: force the PNM file to be a PPM  (-f)\n"
           "  [--width  <int>]: specify the image width  (for xyls inputs)  (-w)\n"
           "  [--height <int>]: specify the image height (for xyls inputs)  (-e)\n"
           "  [--x-column <name>]: for xyls inputs: the name of the FITS column containing the X coordinate of the sources  (-X)\n"
           "  [--y-column <name>]: for xyls inputs: the name of the FITS column containing the Y coordinate of the sources  (-Y)\n"
           "  [--sort-column <name>]: for xyls inputs: the name of the FITS column that should be used to sort the sources  (-s)\n"
           "  [--sort-ascending]: when sorting, sort in ascending (smallest first) order   (-a)\n"
	       "  [--scale-units <units>]: in what units are the lower and upper bound specified?   (-u)\n"
	       "     choices:  \"degwidth\"    : width of the image, in degrees\n"
	       "               \"arcminwidth\" : width of the image, in arcminutes\n"
	       "               \"arcsecperpix\": arcseconds per pixel\n"
	       "  [--scale-low  <number>]: lower bound of image scale estimate   (-L)\n"
	       "  [--scale-high <number>]: upper bound of image scale estimate   (-H)\n"
           "  [--fields <number>]: specify a field (ie, FITS extension) to solve  (-F)\n"
           "  [--fields <min>/<max>]: specify a range of fields (FITS extensions) to solve; inclusive  (-F)\n"
		   "  [--depth <number>]: number of field objects to look at   (-d)\n"
	       "  [--tweak-order <integer>]: polynomial order of SIP WCS corrections  (-t)\n"
           "  [--no-fits2fits]: don't sanitize FITS files; assume they're already sane  (-2)\n"
	       "  [--no-tweak]: don't fine-tune WCS by computing a SIP polynomial  (-T)\n"
           "  [--temp-dir <dir>]: where to put temp files, default /tmp  (-m)\n"
           "  [--verbose]: be more chatty!\n"
           "  [--keep-xylist <filename>]: save the (unaugmented) xylist to <filename>  (-k)\n"
           "  [--verify <wcs-file>]: try to verify an existing WCS file  (-V)\n"
           "  [--code-tolerance <tol>]: matching distance for quads, default 0.01  (-c)\n"
           "  [--pixel-error <pix>]: for verification, size of pixel positional error, default 1  (-E)\n"
           "  [--resort]: sort the star brightnesses using a compromise between background-subtraction and no background-subtraction (-r). \n"
           "  [--downsample <n>]: downsample the image by factor <n> before running source extraction  (-z).\n"
           "  [--dont-augment]: quit after writing the xylist (use with --keep-xylist)  (-A)\n"
		   "\n", progname);
}

static int parse_depth_string(il* depths, const char* str);
static int parse_fields_string(il* fields, const char* str);

static void append_escape(sl* list, const char* fn) {
    sl_append_nocopy(list, shell_escape(fn));
}
static void append_executable(sl* list, const char* fn, const char* me) {
    char* exec = find_executable(fn, me);
    if (!exec) {
        fprintf(stderr, "Error, couldn't find executable \"%s\".\n", fn);
        exit(-1);
    }
    sl_append_nocopy(list, shell_escape(exec));
    free(exec);
}

static sl* backtick(sl* cmd, bool verbose) {
    char* cmdstr = sl_implode(cmd, " ");
    sl* lines;
    if (verbose)
        printf("Running: %s\n", cmdstr);
    if (run_command_get_outputs(cmdstr, &lines, NULL)) {
        free(cmdstr);
        ERROR("Failed to run %s", sl_get(cmd, 0));
        exit(-1);
    }
    free(cmdstr);
    sl_remove_all(cmd);
    return lines;
}

static void run(sl* cmd, bool verbose) {
    sl* lines = backtick(cmd, verbose);
    sl_free2(lines);
}

int main(int argc, char** args) {
	int c;
	int rtn;
	int help_flag = 0;
	char* outfn = NULL;
	char* imagefn = NULL;
	char* xylsfn = NULL;
    sl* cmd;
	int W = 0, H = 0;
	double scalelo = 0.0, scalehi = 0.0;
	char* scaleunits = NULL;
	qfits_header* hdr = NULL;
	bool tweak = TRUE;
	int tweak_order = 0;
	int orig_nheaders;
	FILE* fout = NULL;
	char* savepnmfn = NULL;
    bool force_ppm = FALSE;
	bool guess_scale = FALSE;
	dl* scales;
	int i, I;
	bool guessed_scale = FALSE;
	char* cancelfile = NULL;
	char* solvedfile = NULL;
	char* matchfile = NULL;
	char* rdlsfile = NULL;
	char* wcsfile = NULL;
    // contains ranges of depths as pairs of ints.
    il* depths;
    // contains ranges of fields as pairs of ints.
    il* fields;
    bool nof2f = FALSE;
    char* xcol = NULL;
    char* ycol = NULL;
    char* me;
    char* tempdir = "/tmp";
	// tempfiles to delete when we finish
    sl* tempfiles;
    char* sortcol = NULL;
    bool descending = TRUE;
    bool dosort = FALSE;
    bool verbose = FALSE;
    char* keep_xylist = NULL;
    char* solvedin = NULL;
    bool addwh = TRUE;
    sl* verifywcs;
    double codetol = 0.0;
    double pixerr = 0.0;
    bool resort = FALSE;
    bool doaugment = TRUE;
    int scaledown = 0;

    depths = il_new(4);
    fields = il_new(16);
    cmd = sl_new(16);
    tempfiles = sl_new(4);
    verifywcs = sl_new(4);

    me = find_executable(args[0], NULL);

	while (1) {
		int option_index = 0;
		c = getopt_long(argc, args, OPTIONS, long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
		case 0:
            fprintf(stderr, "Unknown option '-%c'\n", optopt);
            exit(-1);
        case 'A':
            doaugment = FALSE;
            break;
        case 'z':
            if (optarg)
                scaledown = atoi(optarg);
            else
                scaledown = 2;
            break;
		case 'h':
			help_flag = 1;
			break;
        case 'r':
            resort = TRUE;
            break;
        case 'E':
            pixerr = atof(optarg);
            break;
        case 'c':
            codetol = atof(optarg);
            break;
        case 'v':
            verbose = TRUE;
            break;
        case 'V':
            sl_append(verifywcs, optarg);
            break;
        case 'I':
            solvedin = optarg;
            break;
        case 'k':
            keep_xylist = optarg;
            break;
        case 's':
            sortcol = optarg;
            break;
        case 'a':
            descending = FALSE;
            break;
        case 'X':
            xcol = optarg;
            break;
        case 'Y':
            ycol = optarg;
            break;
        case 'm':
            tempdir = optarg;
            break;
        case '2':
            nof2f = TRUE;
            break;
        case 'F':
            if (parse_fields_string(fields, optarg)) {
                fprintf(stderr, "Failed to parse fields specification \"%s\".\n", optarg);
                exit(-1);
            }
            break;
        case 'd':
            if (parse_depth_string(depths, optarg)) {
                fprintf(stderr, "Failed to parse depth specification: \"%s\"\n", optarg);
                exit(-1);
            }
            break;
		case 'o':
			outfn = optarg;
			break;
		case 'i':
			imagefn = optarg;
			break;
		case 'x':
			xylsfn = optarg;
			break;
		case 'L':
			scalelo = atof(optarg);
			break;
		case 'H':
			scalehi = atof(optarg);
			break;
		case 'u':
			scaleunits = optarg;
			break;
		case 'w':
			W = atoi(optarg);
			break;
		case 'e':
			H = atoi(optarg);
			break;
		case 'T':
			tweak = FALSE;
			break;
		case 't':
			tweak_order = atoi(optarg);
			break;
		case 'P':
			savepnmfn = optarg;
			break;
        case 'f':
            force_ppm = TRUE;
            break;
		case 'g':
			guess_scale = TRUE;
			break;
		case 'S':
			solvedfile = optarg;
			break;
		case 'C':
			cancelfile = optarg;
			break;
		case 'M':
			matchfile = optarg;
			break;
		case 'R':
			rdlsfile = optarg;
			break;
		case 'W':
			wcsfile = optarg;
			break;
		case '?':
			break;
		default:
            fprintf(stderr, "Unknown option '-%c'\n", optopt);
			exit( -1);
		}
	}

	rtn = 0;
	if (optind != argc) {
		int i;
		printf("Unknown arguments:\n  ");
		for (i=optind; i<argc; i++) {
			printf("%s ", args[i]);
		}
		printf("\n");
		help_flag = 1;
		rtn = -1;
	}
	if (!outfn) {
		printf("Output filename (-o / --out) is required.\n");
		help_flag = 1;
		rtn = -1;
	}
	if (!(imagefn || xylsfn)) {
		printf("Require either an image (-i / --image) or an XYlist (-x / --xylist) input file.\n");
		help_flag = 1;
		rtn = -1;
	}
	if (!((!scaleunits) ||
		  (!strcasecmp(scaleunits, "degwidth")) ||
		  (!strcasecmp(scaleunits, "arcminwidth")) ||
		  (!strcasecmp(scaleunits, "arcsecperpix")))) {
		printf("Unknown scale units \"%s\".\n", scaleunits);
		help_flag = 1;
		rtn = -1;
	}
	if (help_flag) {
		print_help(args[0]);
		exit(rtn);
	}

	scales = dl_new(16);

	if (imagefn) {
		// if --image is given:
		//	 -run image2pnm.py
		//	 -if it's a FITS image, keep the original (well, sanitized version)
		//	 -otherwise, run ppmtopgm (if necessary) and pnmtofits.
		//	 -run image2xy to generate xylist
		char *uncompressedfn;
		char *sanitizedfn;
		char *pnmfn;				
		sl* lines;
		bool isfits;
        bool iscompressed;
		char *fitsimgfn;
		char* line;
		char pnmtype;
		int maxval;
		char typestr[256];

        uncompressedfn = create_temp_file("uncompressed", tempdir);
		sanitizedfn = create_temp_file("sanitized", tempdir);
        sl_append_nocopy(tempfiles, uncompressedfn);
        sl_append_nocopy(tempfiles, sanitizedfn);

		if (savepnmfn)
			pnmfn = savepnmfn;
		else {
            pnmfn = create_temp_file("pnm", tempdir);
            sl_append_nocopy(tempfiles, pnmfn);
        }

        append_executable(cmd, "image2pnm.py", me);
        if (!verbose)
            sl_append(cmd, "--quiet");
        if (nof2f)
            sl_append(cmd, "--no-fits2fits");
        sl_append(cmd, "--infile");
        append_escape(cmd, imagefn);
        sl_append(cmd, "--uncompressed-outfile");
        append_escape(cmd, uncompressedfn);
        sl_append(cmd, "--sanitized-fits-outfile");
        append_escape(cmd, sanitizedfn);
        sl_append(cmd, "--outfile");
        append_escape(cmd, pnmfn);
        if (force_ppm)
            sl_append(cmd, "--ppm");

        lines = backtick(cmd, verbose);

		isfits = FALSE;
        iscompressed = FALSE;
		for (i=0; i<sl_size(lines); i++) {
            if (verbose)
                printf("  %s\n", sl_get(lines, i));
			if (!strcmp("fits", sl_get(lines, i)))
				isfits = TRUE;
			if (!strcmp("compressed", sl_get(lines, i)))
				iscompressed = TRUE;
		}
		sl_free2(lines);

		// Get image W, H, depth.
        sl_append(cmd, "pnmfile");
        append_escape(cmd, pnmfn);

        lines = backtick(cmd, verbose);

		if (sl_size(lines) == 0) {
			fprintf(stderr, "No output from pnmfile.\n");
			exit(-1);
		}
		line = sl_get(lines, 0);
		// eg	"/tmp/pnm:	 PPM raw, 800 by 510  maxval 255"
		if (strlen(pnmfn) + 1 >= strlen(line)) {
			fprintf(stderr, "Failed to parse output from pnmfile: %s\n", line);
			exit(-1);
		}
		line += strlen(pnmfn) + 1;
		if (sscanf(line, " P%cM %255s %d by %d maxval %d",
				   &pnmtype, typestr, &W, &H, &maxval) != 5) {
			fprintf(stderr, "Failed to parse output from pnmfile: %s\n", line);
			exit(-1);
		}
		sl_free2(lines);

		if (isfits) {
            if (nof2f) {
                if (iscompressed)
                    fitsimgfn = uncompressedfn;
                else
                    fitsimgfn = imagefn;
            } else
                fitsimgfn = sanitizedfn;

			if (guess_scale) {
                dl* estscales = NULL;
                fits_guess_scale(fitsimgfn, NULL, &estscales);
				for (i=0; i<dl_size(estscales); i++) {
					double scale = dl_get(estscales, i);
                    if (verbose)
                        printf("Scale estimate: %g\n", scale);
                    dl_append(scales, scale * 0.99);
                    dl_append(scales, scale * 1.01);
                    guessed_scale = TRUE;
                }
				dl_free(estscales);
			}

		} else {
			fitsimgfn = create_temp_file("fits", tempdir);
            sl_append_nocopy(tempfiles, fitsimgfn);
            
			if (pnmtype == 'P') {
                if (verbose)
                    printf("Converting PPM image to FITS...\n");

                sl_append(cmd, "ppmtopgm");
                append_escape(cmd, pnmfn);
                sl_append(cmd, "| pnmtofits >");
                append_escape(cmd, fitsimgfn);

                run(cmd, verbose);

			} else {
                if (verbose)
                    printf("Converting PGM image to FITS...\n");

                sl_append(cmd, "pnmtofits");
                append_escape(cmd, pnmfn);
                sl_append(cmd, ">");
                append_escape(cmd, fitsimgfn);

                run(cmd, verbose);
			}
		}

        printf("Extracting sources...\n");
        if (verbose)
            printf("Running image2xy...\n");

		xylsfn = create_temp_file("xyls", tempdir);
        sl_append_nocopy(tempfiles, xylsfn);

        append_executable(cmd, "image2xy", me);
        if (!verbose)
            sl_append(cmd, "-q");
        sl_append(cmd, "-O");
        sl_append(cmd, "-o");
        append_escape(cmd, xylsfn);
        append_escape(cmd, fitsimgfn);

        if (scaledown > 1) {
            if (scaledown == 2)
                sl_append(cmd, "-H");
            else {
                // FIXME
                fprintf(stderr, "Can only downsample image by 2.\n");
                exit(-1);
            }
        }

        run(cmd, verbose);

        dosort = TRUE;

	} else {
		// xylist.
		// if --xylist is given:
		//	 -fits2fits.py sanitize

        if (sortcol)
            dosort = TRUE;

        if (!nof2f) {
            char* sanexylsfn;

            if (keep_xylist && !dosort) {
                sanexylsfn = keep_xylist;
            } else {
                sanexylsfn = create_temp_file("sanexyls", tempdir);
                sl_append_nocopy(tempfiles, sanexylsfn);
            }

            append_executable(cmd, "fits2fits.py", me);
            if (verbose)
                sl_append(cmd, "--verbose");
            append_escape(cmd, xylsfn);
            append_escape(cmd, sanexylsfn);

            run(cmd, verbose);
            xylsfn = sanexylsfn;
        }

	}

    if (dosort) {
        char* sortedxylsfn;

        if (!sortcol)
            sortcol = "FLUX";

        if (keep_xylist) {
            sortedxylsfn = keep_xylist;
        } else {
            sortedxylsfn = create_temp_file("sorted", tempdir);
            sl_append_nocopy(tempfiles, sortedxylsfn);
        }

        if (resort) {
            append_executable(cmd, "resort-xylist", me);
            sl_append(cmd, "-f");
            append_escape(cmd, sortcol);
            if (descending)
                sl_append(cmd, "-d");
            append_escape(cmd, xylsfn);
            append_escape(cmd, sortedxylsfn);
            run(cmd, verbose);

        } else {
            tabsort(sortcol, xylsfn, sortedxylsfn, descending);
        }

		xylsfn = sortedxylsfn;
    }

    if (!doaugment)
        // done!
        goto cleanup;

	// start piling FITS headers in there.
	hdr = qfits_header_read(xylsfn);
	if (!hdr) {
		fprintf(stderr, "Failed to read FITS header from file %s.\n", xylsfn);
		exit(-1);
	}

	orig_nheaders = qfits_header_n(hdr);

    if (!(W && H)) {
        // Look for existing IMAGEW and IMAGEH in primary header.
        W = qfits_header_getint(hdr, "IMAGEW", 0);
        H = qfits_header_getint(hdr, "IMAGEH", 0);
        if (W && H) {
            addwh = FALSE;
        } else {
            // Look for IMAGEW and IMAGEH headers in first extension, else bail.
            qfits_header* hdr2 = qfits_header_readext(xylsfn, 1);
            W = qfits_header_getint(hdr2, "IMAGEW", 0);
            H = qfits_header_getint(hdr2, "IMAGEH", 0);
            qfits_header_destroy(hdr2);
        }
        if (!(W && H)) {
            fprintf(stderr, "Error: image width and height must be specified for XYLS inputs.\n");
            exit(-1);
        }
    }

    // we may write long filenames.
    fits_header_add_longstring_boilerplate(hdr);

    if (addwh) {
        fits_header_add_int(hdr, "IMAGEW", W, "image width");
        fits_header_add_int(hdr, "IMAGEH", H, "image height");
    }
	qfits_header_add(hdr, "ANRUN", "T", "Solve this field!", NULL);

    if (xcol)
        qfits_header_add(hdr, "ANXCOL", xcol, "Name of column containing X coords", NULL);
    if (ycol)
        qfits_header_add(hdr, "ANYCOL", ycol, "Name of column containing Y coords", NULL);

	if ((scalelo > 0.0) || (scalehi > 0.0)) {
		double appu, appl;
		if (!scaleunits || !strcasecmp(scaleunits, "degwidth")) {
			appl = deg2arcsec(scalelo) / (double)W;
			appu = deg2arcsec(scalehi) / (double)W;
		} else if (!strcasecmp(scaleunits, "arcminwidth")) {
			appl = arcmin2arcsec(scalelo) / (double)W;
			appu = arcmin2arcsec(scalehi) / (double)W;
		} else if (!strcasecmp(scaleunits, "arcsecperpix")) {
			appl = scalelo;
			appu = scalehi;
		} else
			exit(-1);

		dl_append(scales, appl);
		dl_append(scales, appu);
	}

	if ((dl_size(scales) > 0) && guessed_scale)
		qfits_header_add(hdr, "ANAPPDEF", "T", "try the default scale range too.", NULL);

	for (i=0; i<dl_size(scales)/2; i++) {
		char key[64];
        double lo = dl_get(scales, 2*i);
        double hi = dl_get(scales, 2*i + 1);
        if (lo > 0.0) {
            sprintf(key, "ANAPPL%i", i+1);
            fits_header_add_double(hdr, key, lo, "scale: arcsec/pixel min");
        }
        if (hi > 0.0) {
            sprintf(key, "ANAPPU%i", i+1);
            fits_header_add_double(hdr, key, hi, "scale: arcsec/pixel max");
        }
	}

	qfits_header_add(hdr, "ANTWEAK", (tweak ? "T" : "F"), (tweak ? "Tweak: yes please!" : "Tweak: no, thanks."), NULL);
	if (tweak && tweak_order)
		fits_header_add_int(hdr, "ANTWEAKO", tweak_order, "Tweak order");

	if (solvedfile)
		fits_header_addf_longstring(hdr, "ANSOLVED", "solved output file", "%s", solvedfile);
	if (solvedin)
		fits_header_addf_longstring(hdr, "ANSOLVIN", "solved input file", "%s", solvedin);
	if (cancelfile)
		fits_header_addf_longstring(hdr, "ANCANCEL", "cancel output file", "%s", cancelfile);
	if (matchfile)
		fits_header_addf_longstring(hdr, "ANMATCH", "match output file", "%s", matchfile);
	if (rdlsfile)
		fits_header_addf_longstring(hdr, "ANRDLS", "ra-dec output file", "%s", rdlsfile);
	if (wcsfile)
		fits_header_addf_longstring(hdr, "ANWCS", "WCS header output filename", "%s", wcsfile);
    if (codetol > 0.0)
		fits_header_add_double(hdr, "ANCTOL", codetol, "code tolerance");
    if (pixerr > 0.0)
		fits_header_add_double(hdr, "ANPOSERR", pixerr, "star pos'n error (pixels)");

    for (i=0; i<il_size(depths)/2; i++) {
        int depthlo, depthhi;
        char key[64];
        depthlo = il_get(depths, 2*i);
        depthhi = il_get(depths, 2*i + 1);
        sprintf(key, "ANDPL%i", (i+1));
		fits_header_addf(hdr, key, "", "%i", depthlo);
        sprintf(key, "ANDPU%i", (i+1));
		fits_header_addf(hdr, key, "", "%i", depthhi);
    }

    for (i=0; i<il_size(fields)/2; i++) {
        int lo = il_get(fields, 2*i);
        int hi = il_get(fields, 2*i + 1);
        char key[64];
        if (lo == hi) {
            sprintf(key, "ANFD%i", (i+1));
            fits_header_add_int(hdr, key, lo, "field to solve");
        } else {
            sprintf(key, "ANFDL%i", (i+1));
            fits_header_add_int(hdr, key, lo, "field range: low");
            sprintf(key, "ANFDU%i", (i+1));
            fits_header_add_int(hdr, key, hi, "field range: high");
        }
    }

    I = 0;
    for (i=0; i<sl_size(verifywcs); i++) {
        char* fn;
        sip_t sip;
		int j;

        fn = sl_get(verifywcs, i);
        if (!sip_read_header_file(fn, &sip)) {
            fprintf(stderr, "Failed to parse WCS header from file \"%s\".\n", fn);
            continue;
        }
        I++;
        {
            tan_t* wcs = &(sip.wcstan);
            // note, this initialization has to happen *after* you read the WCS header :)
            double vals[] = { wcs->crval[0], wcs->crval[1],
                              wcs->crpix[0], wcs->crpix[1],
                              wcs->cd[0][0], wcs->cd[0][1],
                              wcs->cd[1][0], wcs->cd[1][1] };
            char key[64];
            char* keys[] = { "ANW%iPIX1", "ANW%iPIX2", "ANW%iVAL1", "ANW%iVAL2",
                             "ANW%iCD11", "ANW%iCD12", "ANW%iCD21", "ANW%iCD22" };
            int m, n, order;
            for (j = 0; j < 8; j++) {
                sprintf(key, keys[j], I);
                fits_header_add_double(hdr, key, vals[j], "");
            }

            if (sip.a_order) {
                sprintf(key, "ANW%iSAO", I);
                order = sip.a_order;
                fits_header_add_int(hdr, key, order, "SIP forward polynomial order");
                for (m=0; m<=order; m++) {
                    for (n=0; (m+n)<=order; n++) {
                        if (m+n < 1)
                            continue;
                        sprintf(key, "ANW%iA%i%i", I, m, n);
                        fits_header_add_double(hdr, key, sip.a[m][n], "");
                        sprintf(key, "ANW%iB%i%i", I, m, n);
                        fits_header_add_double(hdr, key, sip.b[m][n], "");
                    }
                }
            }
            if (sip.ap_order) {
                order = sip.ap_order;
                sprintf(key, "ANW%iSAPO", I);
                fits_header_add_int(hdr, key, order, "SIP reverse polynomial order");
                for (m=0; m<=order; m++) {
                    for (n=0; (m+n)<=order; n++) {
                        if (m+n < 1)
                            continue;
                        sprintf(key, "ANW%iAP%i%i", I, m, n);
                        fits_header_add_double(hdr, key, sip.ap[m][n], "");
                        sprintf(key, "ANW%iBP%i%i", I, m, n);
                        fits_header_add_double(hdr, key, sip.bp[m][n], "");
                    }
                }
            }
        }
    }
    sl_free2(verifywcs);

	fout = fopen(outfn, "wb");
	if (!fout) {
		fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
		exit(-1);
	}

    if (verbose)
        printf("Writing headers to file %s.\n", outfn);

	if (qfits_header_dump(hdr, fout)) {
		fprintf(stderr, "Failed to write FITS header.\n");
		exit(-1);
	}
    qfits_header_destroy(hdr);

	// copy blocks from xyls to output.
	{
		FILE* fin;
		int start;
		int nb;
		struct stat st;

        if (verbose)
            printf("Copying body of file %s to output %s.\n", xylsfn, outfn);

		start = fits_blocks_needed(orig_nheaders * FITS_LINESZ) * FITS_BLOCK_SIZE;

		if (stat(xylsfn, &st)) {
			fprintf(stderr, "Failed to stat() xyls file \"%s\": %s\n", xylsfn, strerror(errno));
			exit(-1);
		}
		nb = st.st_size;

		fin = fopen(xylsfn, "rb");
		if (!fin) {
			fprintf(stderr, "Failed to open xyls file \"%s\": %s\n", xylsfn, strerror(errno));
			exit(-1);
		}

        if (pipe_file_offset(fin, start, nb - start, fout)) {
            fprintf(stderr, "Failed to copy the data segment of xylist file %s to %s.\n", xylsfn, outfn);
            exit(-1);
        }
		fclose(fin);
	}
    fclose(fout);

 cleanup:
	for (i=0; i<sl_size(tempfiles); i++) {
		char* fn = sl_get(tempfiles, i);
		if (unlink(fn)) {
			fprintf(stderr, "Failed to delete temp file \"%s\": %s.\n", fn, strerror(errno));
		}
	}

    dl_free(scales);
    il_free(depths);
    il_free(fields);
    sl_free2(cmd);
    sl_free2(tempfiles);

	return 0;
}

static int parse_depth_string(il* depths, const char* str) {
    return parse_positive_range_string(depths, str, 0, 0, "Depth");
}

static int parse_fields_string(il* fields, const char* str) {
    // 10,11,20-25,30,40-50
    while (str && *str) {
        unsigned int lo, hi;
        int nread;
        if (sscanf(str, "%u-%u", &lo, &hi) == 2) {
            sscanf(str, "%*u-%*u%n", &nread);
        } else if (sscanf(str, "%u", &lo) == 1) {
            sscanf(str, "%*u%n", &nread);
            hi = lo;
        } else {
            fprintf(stderr, "Failed to parse fragment: \"%s\"\n", str);
            return -1;
        }
        if (lo <= 0) {
            fprintf(stderr, "Field number %i is invalid: must be >= 1.\n", lo);
            return -1;
        }
        if (lo > hi) {
            fprintf(stderr, "Field range %i to %i is invalid: max must be >= min!\n", lo, hi);
            return -1;
        }
        il_append(fields, lo);
        il_append(fields, hi);
        str += nread;
        while ((*str == ',') || isspace(*str))
            str++;
    }
    return 0;
}

