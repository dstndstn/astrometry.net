/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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


/**
   A command-line interface to the blind solver system.

TODO:

(2) It assumes you have netpbm tools installed which the main build
doesn't require.

> I think it will only complain if it needs one of the netpbm programs to do
> its work - and it cannot do anything sensible (except print a friendly
> error message) if they don't exist.

(6)  by default, we do not produce an entirely new fits file but this can
be turned on

(7) * by default, we output to stdout a single line for each file something like:
myimage.png: unsolved using X field objects
or
myimage.png: solved using X field objects, RA=rr,DEC=dd, size=AxB
pixels=UxV arcmin

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <libgen.h>
#include <errors.h>

#include <getopt.h>

#include "an-bool.h"
#include "bl.h"
#include "ioutils.h"
#include "xylist.h"
#include "matchfile.h"
#include "scriptutils.h"
#include "fitsioutils.h"

static const char* OPTIONS = "2C:D:E:F:GH:I:KL:OPSTU:V:W:X:Y:ac:d:fghi:k:m:o:rs:t:u:vz";


static struct option long_options[] = {
	{"help",           no_argument,       0, 'h'},
	{"verbose",        no_argument,       0, 'v'},
	{"width",          required_argument, 0, 'W'},
	{"height",         required_argument, 0, 'H'},
	{"scale-low",	   required_argument, 0, 'L'},
	{"scale-high",	   required_argument, 0, 'U'},
	{"scale-units",    required_argument, 0, 'u'},
    {"fields",         required_argument, 0, 'F'},
	{"depth",          required_argument, 0, 'D'},
	{"no-tweak",       no_argument,       0, 'T'},
	{"no-guess-scale", no_argument,       0, 'G'},
	{"tweak-order",    required_argument, 0, 't'},
	{"dir",            required_argument, 0, 'd'},
	{"out",            required_argument, 0, 'o'},
	{"backend-config", required_argument, 0, 'c'},
	{"files-on-stdin", no_argument,       0, 'f'},
	{"overwrite",      no_argument,       0, 'O'},
	{"no-plots",       no_argument,       0, 'P'},
	{"no-fits2fits",   no_argument,       0, '2'},
	{"temp-dir",       required_argument, 0, 'm'},
	{"x-column",       required_argument, 0, 'X'},
	{"y-column",       required_argument, 0, 'Y'},
    {"sort-column",    required_argument, 0, 's'},
    {"sort-ascending", no_argument,       0, 'a'},
    {"keep-xylist",    required_argument, 0, 'k'},
	{"solved-in",      required_argument, 0, 'I'},
	{"solved-in-dir",  required_argument, 0, 'i'},
    {"verify",         required_argument, 0, 'V'},
    {"code-tolerance", required_argument, 0, 'C'},
    {"pixel-error",    required_argument, 0, 'E'},
    {"use-wget",       no_argument,       0, 'g'},
    {"resort",         no_argument,       0, 'r'},
    {"downsample",     no_argument,       0, 'z'},
    {"continue",       no_argument,       0, 'K'},
    {"skip-solved",    no_argument,       0, 'S'},
	{0, 0, 0, 0}
};

static void print_help(const char* progname) {
	printf("Usage:   %s [options]\n"
	       "  [--dir <directory>]: place all output files in this directory\n"
	       "  [--out <filename>]: name the output files with this base name\n"
	       "  [--scale-units <units>]: in what units are the lower and upper bound specified?   (-u)\n"
	       "     choices:  \"degwidth\"    : width of the image, in degrees\n"
	       "               \"arcminwidth\" : width of the image, in arcminutes\n"
	       "               \"arcsecperpix\": arcseconds per pixel\n"
	       "  [--scale-low  <number>]: lower bound of image scale estimate   (-L)\n"
	       "  [--scale-high <number>]: upper bound of image scale estimate   (-U)\n"
           "  [--fields <number>]: specify a field (ie, FITS extension) to solve\n"
           "  [--fields <min>/<max>]: specify a range of fields (FITS extensions) to solve; inclusive\n"
	       "  [--width  <number>]: (mostly for xyls inputs): the original image width   (-W)\n"
	       "  [--height <number>]: (mostly for xyls inputs): the original image height  (-H)\n"
           "  [--x-column <name>]: for xyls inputs: the name of the FITS column containing the X coordinate of the sources.  (-X)\n"
           "  [--y-column <name>]: for xyls inputs: the name of the FITS column containing the Y coordinate of the sources.  (-Y)\n"
           "  [--sort-column <name>]: for xyls inputs: the name of the FITS column that should be used to sort the sources  (-s)\n"
           "  [--sort-ascending]: when sorting, sort in ascending (smallest first) order   (-a)\n"
		   "  [--depth <number1>-<number2>]: consider hypotheses generated from field objects num1-num2   (-D)\n"
	       "  [--tweak-order <integer>]: polynomial order of SIP WCS corrections.  (-t <#>)\n"
	       "  [--no-tweak]: don't fine-tune WCS by computing a SIP polynomial  (-T)\n"
	       "  [--no-guess-scale]: don't try to guess the image scale from the FITS headers  (-G)\n"
           "  [--no-plots]: don't create any PNG plots.  (-P)\n"
           "  [--no-fits2fits]: don't sanitize FITS files; assume they're already sane.  (-2)\n"
	       "  [--backend-config <filename>]: use this config file for the \"backend\" program.  (-c <file>)\n"
	       "  [--overwrite]: overwrite output files if they already exist.  (-O)\n"
	       "  [--continue]: don't overwrite output files if they already exist; continue a previous run  (-K)\n"
	       "  [--skip-solved]: skip input files for which the 'solved' output file already exists;\n"
           "                  NOTE: this assumes single-field input files.  (-S)\n"
           "  [--continue]: don't overwrite output files if they already exist; continue a previous run  (-K)\n"
	       "  [--files-on-stdin]: read filenames to solve on stdin, one per line (-f)\n"
           "  [--temp-dir <dir>]: where to put temp files, default /tmp  (-m)\n"
           "  [--verbose]: be more chatty!  (-v)\n"
           "  [--keep-xylist <filename>]: save the (unaugmented) xylist to <filename>  (-k)\n"
           "  [--solved-in-dir <dir>]: directory containing input solved files  (-i)\n"
           "  [--solved-in <filename>]: input filename for solved file  (-I)\n"
           "  [--verify <wcs-file>]: try to verify an existing WCS file  (-V)\n"
           "  [--code-tolerance <tol>]: matching distance for quads (default 0.01) (-c)\n"
           "  [--pixel-error <pix>]: for verification, size of pixel positional error, default 1  (-E)\n"
           "  [--use-wget]: use wget instead of curl.  (-g)\n"
           "  [--resort]: sort the star brightnesses using a compromise between background-subtraction and no background-subtraction (-r). \n"
           "  [--downsample]: downsample the image by half before doing source extraction  (-z)\n"
	       "\n"
	       "  [<image-file-1> <image-file-2> ...] [<xyls-file-1> <xyls-file-2> ...]\n"
           "\n"
           "You can specify http:// or ftp:// URLs instead of filenames.  The \"wget\" or \"curl\" program will be used to retrieve the URL.\n"
	       "\n", progname);
}

static int run_command(const char* cmd, bool* ctrlc) {
	int rtn;
	//printf("Running command:\n  %s\n", cmd);
	rtn = system(cmd);
	if (rtn == -1) {
		fprintf(stderr, "Failed to run command: %s\n", strerror(errno));
		return -1;
	}
	if (WIFSIGNALED(rtn)) {
        if (ctrlc && (WTERMSIG(rtn) == SIGTERM))
            *ctrlc = TRUE;
		return -1;
	}
	rtn = WEXITSTATUS(rtn);
	if (rtn) {
		fprintf(stderr, "Command exited with exit status %i.\n", rtn);
	}
	return rtn;
}

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

int main(int argc, char** args) {
	int c;
	bool help = FALSE;
	sl* augmentxyargs;
	char* outdir = NULL;
	char* image = NULL;
	char* xyls = NULL;
	char* cmd;
	int i, f;
    int inputnum;
	int rtn;
	sl* backendargs;
	const char* errmsg;
	bool guess_scale = TRUE;
	int width = 0, height = 0;
	int nllargs;
	int nbeargs;
	bool fromstdin = FALSE;
	bool overwrite = FALSE;
	bool cont = FALSE;
    bool skip_solved = FALSE;
    bool makeplots = TRUE;
    char* me;
    char* tempdir = "/tmp";
    bool verbose = FALSE;
    char* baseout = NULL;
    char* xcol = NULL;
    char* ycol = NULL;
    char* solvedin = NULL;
    char* solvedindir = NULL;
	bool usecurl = TRUE;
    bool resort = FALSE;

    me = find_executable(args[0], NULL);

	augmentxyargs = sl_new(16);
	append_executable(augmentxyargs, "augment-xylist", me);

	backendargs = sl_new(16);
	append_executable(backendargs, "backend", me);

	rtn = 0;
	while (1) {
		int option_index = 0;
		// getopt_long_only doesn't exist on my MacOS setup...
		//c = getopt_long_only(argc, args, OPTIONS, long_options, &option_index);
		c = getopt_long(argc, args, OPTIONS, long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
        case 'i':
            solvedindir = optarg;
            break;
        case 'z':
            sl_append(augmentxyargs, "--downsample");
            break;
		case 'h':
			help = TRUE;
			break;
        case 'r':
            resort = TRUE;
            sl_append(augmentxyargs, "--resort");
            break;
        case 'g':
            usecurl = FALSE;
            break;
        case 'C':
            sl_append(augmentxyargs, "--code-tolerance");
            sl_append(augmentxyargs, optarg);
            break;
        case 'E':
            sl_append(augmentxyargs, "--pixel-error");
            sl_append(augmentxyargs, optarg);
            break;
        case 'v':
            sl_append(augmentxyargs, "--verbose");
            sl_append(backendargs, "--verbose");
            verbose = TRUE;
            break;
        case 'V':
            sl_append(augmentxyargs, "--verify");
            append_escape(augmentxyargs, optarg);
            break;
        case 'I':
            solvedin = optarg;
            break;
        case 'k':
            sl_append(augmentxyargs, "--keep-xylist");
            append_escape(augmentxyargs, optarg);
            break;
        case 'o':
            baseout = optarg;
            break;
        case 'X':
            sl_append(augmentxyargs, "--x-column");
            append_escape(augmentxyargs, optarg);
            xcol = optarg;
            break;
        case 'Y':
            sl_append(augmentxyargs, "--y-column");
            append_escape(augmentxyargs, optarg);
            ycol = optarg;
            break;
        case 's':
            sl_append(augmentxyargs, "--sort-column");
            append_escape(augmentxyargs, optarg);
            break;
        case 'a':
            sl_append(augmentxyargs, "--sort-ascending");
            break;
        case 'm':
            sl_append(augmentxyargs, "--temp-dir");
            append_escape(augmentxyargs, optarg);
            tempdir = optarg;
            break;
        case '2':
            sl_append(augmentxyargs, "--no-fits2fits");
            break;
        case 'F':
            sl_append(augmentxyargs, "--fields");
            append_escape(augmentxyargs, optarg);
            break;
        case 'D':
            sl_append(augmentxyargs, "--depth");
            append_escape(augmentxyargs, optarg);
            break;
        case 'P':
            makeplots = FALSE;
            break;
        case 'O':
            overwrite = TRUE;
            break;
        case 'K':
            cont = TRUE;
            break;
        case 'S':
            skip_solved = TRUE;
            break;
		case 'G':
			guess_scale = FALSE;
			break;
		case 'W':
			width = atoi(optarg);
			break;
		case 'H':
			height = atoi(optarg);
			break;
		case 'T':
			sl_append(augmentxyargs, "--no-tweak");
			break;
		case 'L':
			sl_append(augmentxyargs, "--scale-low");
			append_escape(augmentxyargs, optarg);
			break;
		case 'U':
			sl_append(augmentxyargs, "--scale-high");
			append_escape(augmentxyargs, optarg);
			break;
		case 'u':
			sl_append(augmentxyargs, "--scale-units");
			append_escape(augmentxyargs, optarg);
			break;
		case 't':
			sl_append(augmentxyargs, "--tweak-order");
			append_escape(augmentxyargs, optarg);
			break;
		case 'c':
			sl_append(backendargs, "--config");
			append_escape(backendargs, optarg);
			break;
		case 'd':
			outdir = optarg;
			break;
		case 'f':
			fromstdin = TRUE;
			break;

        case '?':
            printf("\nTry \"--help\" to get a list of options.\n");
            exit(-1);
		}
	}

	if (optind == argc) {
		printf("ERROR: You didn't specify any files to process.\n");
		help = TRUE;
	}

	if (help) {
		print_help(args[0]);
		exit(rtn);
	}

	if (outdir) {
        char* escout;
        escout = shell_escape(outdir);
		asprintf_safe(&cmd, "mkdir -p %s", escout);
        free(escout);
        if (run_command(cmd, NULL)) {
            free(cmd);
			fprintf(stderr, "Failed to create output directory %s.\n", outdir);
			exit(-1);
        }
        free(cmd);
	}

	if (guess_scale)
		sl_append(augmentxyargs, "--guess-scale");

	// number of low-level and backend args (not specific to a particular file)
	nllargs = sl_size(augmentxyargs);
	nbeargs = sl_size(backendargs);

	f = optind;
    inputnum = 0;
	while (1) {
		char fnbuf[1024];
		char* infile = NULL;
		bool isxyls;
		char* reason;
		int len;
		char* cpy;
		char* base;
		char *matchfn, *rdlsfn, *solvedfn, *wcsfn, *axyfn, *objsfn, *redgreenfn;
        char* solvedinfn = NULL;
		char *ngcfn, *ppmfn=NULL, *indxylsfn;
        char* downloadfn;
        char* suffix = NULL;
		sl* outfiles;
		sl* tempfiles;
		sl* cmdline;
        bool ctrlc = FALSE;

		if (fromstdin) {
			if (!fgets(fnbuf, sizeof(fnbuf), stdin)) {
				if (ferror(stdin))
					fprintf(stderr, "Failed to read a filename!\n");
				break;
			}
			len = strlen(fnbuf);
			if (fnbuf[len-1] == '\n')
				fnbuf[len-1] = '\0';
			infile = fnbuf;
            printf("Reading input file \"%s\"...\n", infile);
		} else {
			if (f == argc)
				break;
			infile = args[f];
			f++;
            printf("Reading input file %i of %i: \"%s\"...\n",
                   f - optind, argc - optind, infile);
		}
        inputnum++;

        cmdline = sl_new(16);

        // Remove arguments that might have been added in previous trips through this loop
		sl_remove_from(augmentxyargs, nllargs);
		sl_remove_from(backendargs,  nbeargs);

		// Choose the base path/filename for output files.
        if (baseout)
            asprintf_safe(&cpy, baseout, inputnum, infile);
        else
            cpy = strdup(infile);
		if (outdir)
			asprintf_safe(&base, "%s/%s", outdir, basename(cpy));
		else
			base = strdup(basename(cpy));
		free(cpy);
		len = strlen(base);
		// trim .xxx / .xxxx
		if (len > 4) {
			if (base[len - 4] == '.') {
				base[len-4] = '\0';
                suffix = base + len - 3;
            }
			if (base[len - 5] == '.') {
				base[len-5] = '\0';
                suffix = base + len - 4;
            }
		}

		// the output filenames.
		outfiles = sl_new(16);
		tempfiles = sl_new(4);

		axyfn      = sl_appendf(outfiles, "%s.axy",       base);
		matchfn    = sl_appendf(outfiles, "%s.match",     base);
		rdlsfn     = sl_appendf(outfiles, "%s.rdls",      base);
		wcsfn      = sl_appendf(outfiles, "%s.wcs",       base);
		objsfn     = sl_appendf(outfiles, "%s-objs.png",  base);
		redgreenfn = sl_appendf(outfiles, "%s-indx.png",  base);
		ngcfn      = sl_appendf(outfiles, "%s-ngc.png",   base);
		indxylsfn  = sl_appendf(outfiles, "%s-indx.xyls", base);
        if (suffix)
            downloadfn = sl_appendf(outfiles, "%s-downloaded.%s", base, suffix);
        else
            downloadfn = sl_appendf(outfiles, "%s-downloaded", base);

		solvedfn   = sl_appendf(outfiles, "%s.solved",    base);

        if (solvedin || solvedindir) {
            if (solvedin && solvedindir)
                asprintf(&solvedinfn, "%s/%s", solvedindir, solvedin);
            else if (solvedin)
                solvedinfn = strdup(solvedin);
            else {
                char* bc = strdup(base);
                char* bn = strdup(basename(bc));
                asprintf(&solvedinfn, "%s/%s.solved", solvedindir, bn);
                free(bn);
                free(bc);
            }
        }
        if (solvedinfn && (strcmp(solvedfn, solvedinfn) == 0)) {
            // solved input and output files are the same: don't delete the input!
            sl_pop(outfiles);
            // MEMLEAK
        }

		free(base);
		base = NULL;

        if (skip_solved) {
            if (solvedinfn) {
                if (verbose)
                    printf("Checking for solved file %s\n", solvedinfn);
                if (file_exists(solvedinfn)) {
                    printf("Solved file exists: %s; skipping this input file.\n", solvedinfn);
                    goto nextfile;
                }
            }
            if (verbose)
                printf("Checking for solved file %s\n", solvedfn);
            if (file_exists(solvedfn)) {
                printf("Solved file exists: %s; skipping this input file.\n", solvedfn);
                goto nextfile;
            }
        }

		// Check for (and possibly delete) existing output filenames.
		for (i = 0; i < sl_size(outfiles); i++) {
			char* fn = sl_get(outfiles, i);
			if (!file_exists(fn))
				continue;
            if (cont) {
            } else if (overwrite) {
				if (unlink(fn)) {
					printf("Failed to delete an already-existing output file: \"%s\": %s\n", fn, strerror(errno));
					exit(-1);
				}
			} else {
				printf("Output file \"%s\" already exists.  Bailing out.  "
				       "Use the --overwrite flag to overwrite existing files,\n"
                       " or the --continue  flag to not overwrite existing files.\n", fn);
				printf("Continuing to next input file.\n");
                goto nextfile;
			}
		}

        // Download URL...
        if (!file_exists(infile) &&
            ((strncasecmp(infile, "http://", 7) == 0) ||
             (strncasecmp(infile, "ftp://", 6) == 0))) {
			if (usecurl) {
				sl_append(cmdline, "curl");
				if (!verbose)
					sl_append(cmdline, "--silent");
				sl_append(cmdline, "--output");
			} else {
				sl_append(cmdline, "wget");
				if (!verbose)
					sl_append(cmdline, "--quiet");
                //sl_append(cmdline, "--no-verbose");
				sl_append(cmdline, "-O");
			}
            append_escape(cmdline, downloadfn);
            append_escape(cmdline, infile);

            cmd = sl_implode(cmdline, " ");
            sl_remove_all(cmdline);

            if (verbose)
                printf("Running:\n  %s\n", cmd);
            else
                printf("Downloading...\n");
            fflush(NULL);
            if (run_command(cmd, &ctrlc)) {
                fflush(NULL);
                fprintf(stderr, "wget command %s; exiting.\n",
                        (ctrlc ? "was cancelled" : "failed"));
                exit(-1);
            }
            free(cmd);

            infile = downloadfn;
        }

        if (verbose)
            printf("Checking if file \"%s\" is xylist or image: ", infile);
        fflush(NULL);
        fits_use_error_system();
        reason = NULL;
		isxyls = xylist_is_file_xylist(infile, xcol, ycol, &reason);
        if (verbose) {
            printf(isxyls ? "xyls\n" : "image\n");
            if (!isxyls) {
                printf("  (%s)\n", reason);
            }
        }
        free(reason);
        fflush(NULL);

		if (isxyls) {
			xyls = infile;
			image = NULL;
		} else {
			xyls = NULL;
			image = infile;
		}

		if (image) {
			sl_append(augmentxyargs, "--image");
			append_escape(augmentxyargs, image);
		} else {
			sl_append(augmentxyargs, "--xylist");
			append_escape(augmentxyargs, xyls);
			/*
			 if (!width || !height) {
			 // Load the xylist and compute the min/max.
			 }
			 */
		}

		if (width) {
			sl_appendf(augmentxyargs, "--width %i", width);
		}
		if (height) {
			sl_appendf(augmentxyargs, "--height %i", height);
		}

		if (image) {
            ppmfn = create_temp_file("ppm", tempdir);
            sl_append_nocopy(tempfiles, ppmfn);

			sl_append(augmentxyargs, "--pnm");
			append_escape(augmentxyargs, ppmfn);
			sl_append(augmentxyargs, "--force-ppm");
		}

		sl_append(augmentxyargs, "--out");
        append_escape(augmentxyargs, axyfn);
		sl_append(augmentxyargs, "--match");
        append_escape(augmentxyargs, matchfn);
		sl_append(augmentxyargs, "--rdls");
        append_escape(augmentxyargs, rdlsfn);
		sl_append(augmentxyargs, "--solved");
        append_escape(augmentxyargs, solvedfn);
		sl_append(augmentxyargs, "--wcs");
        append_escape(augmentxyargs, wcsfn);

        if (solvedinfn) {
            sl_append(augmentxyargs, "--solved-in");
            append_escape(augmentxyargs, solvedinfn);
        }

		cmd = sl_implode(augmentxyargs, " ");
        if (verbose)
            printf("Running:\n  %s\n", cmd);
        fflush(NULL);
		if (run_command(cmd, &ctrlc)) {
            fflush(NULL);
            fprintf(stderr, "augment-xylist %s; exiting.\n",
                    (ctrlc ? "was cancelled" : "failed"));
			exit(-1);
        }
		free(cmd);

        if (makeplots) {
            // source extraction overlay
            // plotxy -i harvard.axy -I /tmp/pnm -C red -P -w 2 -N 50 | plotxy -w 2 -r 3 -I - -i harvard.axy -C red -n 50 > harvard-objs.png
            append_executable(cmdline, "plotxy", me);
            sl_append(cmdline, "-i");
            append_escape(cmdline, axyfn);
            if (image) {
                sl_append(cmdline, "-I");
                append_escape(cmdline, ppmfn);
            }
            if (xcol) {
                sl_append(cmdline, "-X");
                append_escape(cmdline, xcol);
            }
            if (ycol) {
                sl_append(cmdline, "-Y");
                append_escape(cmdline, ycol);
            }
            sl_append(cmdline, "-P");
            sl_append(cmdline, "-C red -w 2 -N 50 -x 1 -y 1");
            
            sl_append(cmdline, "|");

            append_executable(cmdline, "plotxy", me);
            sl_append(cmdline, "-i");
            append_escape(cmdline, axyfn);
            if (xcol) {
                sl_append(cmdline, "-X");
                append_escape(cmdline, xcol);
            }
            if (ycol) {
                sl_append(cmdline, "-Y");
                append_escape(cmdline, ycol);
            }
            sl_append(cmdline, "-I - -w 2 -r 3 -C red -n 50 -N 200 -x 1 -y 1");

            sl_append(cmdline, ">");
            append_escape(cmdline, objsfn);

            cmd = sl_implode(cmdline, " ");
            sl_remove_all(cmdline);
            
            if (verbose)
                printf("Running:\n  %s\n", cmd);
            fflush(NULL);
            if (run_command(cmd, &ctrlc)) {
                fflush(NULL);
                fprintf(stderr, "Plotting command %s.\n",
                        (ctrlc ? "was cancelled" : "failed"));
                if (ctrlc)
                    exit(-1);
                // don't try any more plots...
                fprintf(stderr, "Maybe you didn't build the plotting programs?\n");
                makeplots = FALSE;
            }
            free(cmd);
        }

		append_escape(backendargs, axyfn);
		cmd = sl_implode(backendargs, " ");
        if (verbose)
            printf("Running:\n  %s\n", cmd);
        else
            printf("Solving...\n");
        fflush(NULL);
		if (run_command_get_outputs(cmd, NULL, NULL, &errmsg)) {
            fflush(NULL);
			fprintf(stderr, "backend failed: %s\n", errmsg);
			fprintf(stderr, "exiting.\n");
			exit(-1);
		}
		free(cmd);
        fflush(NULL);

		if (!file_exists(solvedfn)) {
			// boo hoo.
			//printf("Field didn't solve.\n");
		} else {
			matchfile* mf;
			MatchObj* mo;
            sl* lines;

			// index rdls to xyls.
            append_executable(cmdline, "wcs-rd2xy", me);
            if (!verbose)
                sl_append(cmdline, "-q");
			sl_append(cmdline, "-w");
			append_escape(cmdline, wcsfn);
			sl_append(cmdline, "-i");
			append_escape(cmdline, rdlsfn);
			sl_append(cmdline, "-o");
			append_escape(cmdline, indxylsfn);

			cmd = sl_implode(cmdline, " ");
			sl_remove_all(cmdline);
            if (verbose)
                printf("Running:\n  %s\n", cmd);
            fflush(NULL);
			if (run_command(cmd, &ctrlc)) {
                fflush(NULL);
                fprintf(stderr, "wcs-rd2xy %s; exiting.\n",
                        (ctrlc ? "was cancelled" : "failed"));
				exit(-1);
            }
			free(cmd);


            append_executable(cmdline, "wcsinfo", me);
            append_escape(cmdline, wcsfn);

			cmd = sl_implode(cmdline, " ");
			sl_remove_all(cmdline);
            if (verbose)
                printf("Running:\n  %s\n", cmd);
            fflush(NULL);
            if (run_command_get_outputs(cmd, &lines, NULL, &errmsg)) {
                fflush(NULL);
                fprintf(stderr, "wcsinfo failed: %s\n", errmsg);
                fprintf(stderr, "exiting.\n");
                exit(-1);
            }
			free(cmd);

            if (lines) {
                int i;
                char* parity;
                double rac, decc;
                char* rahms=NULL;
                char* decdms=NULL;
                double fieldw, fieldh;
                char* fieldunits = NULL;

                for (i=0; i<sl_size(lines); i++) {
                    char* line;
                    char* nextword;
                    line = sl_get(lines, i);
                    if (is_word(line, "parity ", &nextword)) {
                        if (nextword[0] == '1') {
                            parity = "flipped";
                        } else {
                            parity = "normal";
                        }
                    } else if (is_word(line, "ra_center ", &nextword)) {
                        rac = atof(nextword);
                    } else if (is_word(line, "dec_center ", &nextword)) {
                        decc = atof(nextword);
                    } else if (is_word(line, "ra_center_hms ", &nextword)) {
                        rahms = strdup(nextword);
                    } else if (is_word(line, "dec_center_dms ", &nextword)) {
                        decdms = strdup(nextword);
                    } else if (is_word(line, "fieldw ", &nextword)) {
                        fieldw = atof(nextword);
                    } else if (is_word(line, "fieldh ", &nextword)) {
                        fieldh = atof(nextword);
                    } else if (is_word(line, "fieldunits ", &nextword)) {
                        fieldunits = strdup(nextword);
                    }
                }

                printf("Field center: (RA,Dec) = (%.4g, %.4g) deg.\n", rac, decc);
                printf("Field center: (RA H:M:S, Dec D:M:S) = (%s, %s).\n", rahms, decdms);
                printf("Field size: %g x %g %s\n", fieldw, fieldh, fieldunits);

                free(fieldunits);
                free(rahms);
                free(decdms);

                sl_free2(lines);
            }




            if (makeplots) {
                // sources + index overlay
                append_executable(cmdline, "plotxy", me);
                sl_append(cmdline, "-i");
                append_escape(cmdline, axyfn);
                if (image) {
                    sl_append(cmdline, "-I");
                    append_escape(cmdline, ppmfn);
                }
                if (xcol) {
                    sl_append(cmdline, "-X");
                    append_escape(cmdline, xcol);
                }
                if (ycol) {
                    sl_append(cmdline, "-Y");
                    append_escape(cmdline, ycol);
                }
                sl_append(cmdline, "-P");
                sl_append(cmdline, "-C red -w 2 -r 6 -N 200 -x 1 -y 1");
                sl_append(cmdline, "|");
                append_executable(cmdline, "plotxy", me);
                sl_append(cmdline, "-i");
                append_escape(cmdline, indxylsfn);
                sl_append(cmdline, "-I - -w 2 -r 4 -C green -x 1 -y 1");

                mf = matchfile_open(matchfn);
                if (!mf) {
                    fprintf(stderr, "Failed to read matchfile %s.\n", matchfn);
                    exit(-1);
                }
                // just read the first match...
                mo = matchfile_read_match(mf);
                if (!mo) {
                    fprintf(stderr, "Failed to read a match from matchfile %s.\n", matchfn);
                    exit(-1);
                }

                sl_append(cmdline, " -P |");
                append_executable(cmdline, "plotquad", me);
                sl_append(cmdline, "-I -");
                sl_append(cmdline, "-C green");
                sl_append(cmdline, "-w 2");
				sl_appendf(cmdline, "-d %i", mo->dimquads);
                for (i=0; i<(2 * mo->dimquads); i++)
                    sl_appendf(cmdline, " %g", mo->quadpix[i]);

                matchfile_close(mf);
			
                sl_append(cmdline, ">");
                append_escape(cmdline, redgreenfn);

                cmd = sl_implode(cmdline, " ");
                sl_remove_all(cmdline);
                if (verbose)
                    printf("Running:\n  %s\n", cmd);
                fflush(NULL);
                if (run_command(cmd, &ctrlc)) {
                    fflush(NULL);
                    fprintf(stderr, "Plotting commands %s; exiting.\n",
                            (ctrlc ? "were cancelled" : "failed"));
                    exit(-1);
                }
                free(cmd);
            }

            if (image && makeplots) {
                sl* lines;

                append_executable(cmdline, "plot-constellations", me);
                if (verbose)
                    sl_append(cmdline, "-v");
				sl_append(cmdline, "-w");
				append_escape(cmdline, wcsfn);
				sl_append(cmdline, "-i");
				append_escape(cmdline, ppmfn);
				sl_append(cmdline, "-N");
				sl_append(cmdline, "-C");
				sl_append(cmdline, "-o");
				append_escape(cmdline, ngcfn);

				cmd = sl_implode(cmdline, " ");
				sl_remove_all(cmdline);
				if (verbose)
                    printf("Running:\n  %s\n", cmd);
                fflush(NULL);
                if (run_command_get_outputs(cmd, &lines, NULL, &errmsg)) {
                    fflush(NULL);
                    fprintf(stderr, "plot-constellations failed: %s\n", errmsg);
                    fprintf(stderr, "exiting.\n");
                    exit(-1);
                }
				free(cmd);
                if (lines && sl_size(lines)) {
                    int i;
                    printf("Your field contains:\n");
                    for (i=0; i<sl_size(lines); i++) {
                        printf("  %s\n", sl_get(lines, i));
                    }
                }
                if (lines)
                    sl_free2(lines);
			}

			// create field rdls?
		}
        fflush(NULL);

    nextfile:        // clean up and move on to the next file.
        free(solvedinfn);
		for (i=0; i<sl_size(tempfiles); i++) {
			char* fn = sl_get(tempfiles, i);
			if (unlink(fn))
				fprintf(stderr, "Failed to delete temp file \"%s\": %s.\n", fn, strerror(errno));
		}
        sl_free2(cmdline);
		sl_free2(outfiles);
		sl_free2(tempfiles);
	}

	sl_free2(augmentxyargs);
	sl_free2(backendargs);
    free(me);

	return 0;
}

