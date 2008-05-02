/*
  This file is part of the Astrometry.net suite.
  Copyright 2007-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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
#include "augment-xylist.h"
#include "an-opts.h"
#include "log.h"
#include "errors.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "wcs-rd2xy.h"

static an_option_t options[] = {
	{'h', "help",		   no_argument, NULL,
     "print this help message" },
	{'v', "verbose",       no_argument, NULL,
     "be more chatty" },
    {'D', "dir", required_argument, "directory",
     "place all output files in this directory"},
    {'o', "out", required_argument, "base-filename",
     "name the output files with this base name"},
    {'b', "backend-config", required_argument, "filename",
     "use this config file for the \"backend\" program"},
	{'f', "files-on-stdin", no_argument, NULL,
     "read filenames to solve on stdin, one per line"},
	{'p', "no-plots",       no_argument, NULL,
     "don't create any plots of the results"},
    //{"solved-in-dir",  required_argument, 0, 'i'},
    //directory containing input solved files  (-i)\n"
    {'G', "use-wget",       no_argument, NULL,
     "use wget instead of curl"},
  	{'O', "overwrite",      no_argument, NULL,
     "overwrite output files if they already exist"},
    {'K', "continue",       no_argument, NULL,
     "don't overwrite output files if they already exist; continue a previous run"},
    {'J', "skip-solved",    no_argument, NULL,
     "skip input files for which the 'solved' output file already exists;\n"
     "                  NOTE: this assumes single-field input files"},
};

static void print_help(const char* progname, bl* opts) {
	printf("\nUsage:   %s [options]  [<image-file-1> <image-file-2> ...] [<xyls-file-1> <xyls-file-2> ...]\n"
           "\n"
           "You can specify http:// or ftp:// URLs instead of filenames.  The \"wget\" or \"curl\" program will be used to retrieve the URL.\n"
	       "\n", progname);
    printf("Options include:\n");
    opts_print_help(opts, stdout, augment_xylist_print_special_opts, NULL);
    printf("\n\n");
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
	char* outdir = NULL;
	char* cmd;
	int i, j, f;
    int inputnum;
	int rtn;
	sl* backendargs;
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
    bl* opts;
    augment_xylist_t theallaxy;
    augment_xylist_t* allaxy = &theallaxy;
    int nmyopts;
    char* removeopts = "ixo\x01";

    errors_print_on_exit(stderr);

    me = find_executable(args[0], NULL);

	backendargs = sl_new(16);
	append_executable(backendargs, "backend", me);

	rtn = 0;

    nmyopts = sizeof(options)/sizeof(an_option_t);
    opts = opts_from_array(options, nmyopts, NULL);
    augment_xylist_add_options(opts);

    // remove duplicate short options.
    for (i=0; i<nmyopts; i++) {
        an_option_t* opt1 = bl_access(opts, i);
        for (j=nmyopts; j<bl_size(opts); j++) {
            an_option_t* opt2 = bl_access(opts, j);
            if (opt2->shortopt == opt1->shortopt)
                bl_remove_index(opts, j);
        }
    }

    // remove unwanted augment-xylist options.
    for (i=0; i<strlen(removeopts); i++) {
        for (j=nmyopts; j<bl_size(opts); j++) {
            an_option_t* opt2 = bl_access(opts, j);
            if (opt2->shortopt == removeopts[i])
                bl_remove_index(opts, j);
        }
    }

    augment_xylist_init(allaxy);

	while (1) {
        int res;
		c = opts_getopt(opts, argc, args);
        if (c == -1)
            break;
        switch (c) {
            /*
             case 'i':
             solvedindir = optarg;
             break;
             }
             */
		case 'h':
			help = TRUE;
			break;
        case 'v':
            sl_append(backendargs, "--verbose");
            verbose = TRUE;
            break;
		case 'D':
			outdir = optarg;
			break;
        case 'o':
            baseout = optarg;
            break;
		case 'b':
			sl_append(backendargs, "--config");
			append_escape(backendargs, optarg);
			break;
		case 'f':
			fromstdin = TRUE;
			break;
        case 'O':
            overwrite = TRUE;
            break;
        case 'p':
            makeplots = FALSE;
            break;
        case 'G':
            usecurl = FALSE;
            break;
        case 'K':
            cont = TRUE;
            break;
        case 'J':
            skip_solved = TRUE;
            break;
        default:
            res = augment_xylist_parse_option(c, optarg, allaxy);
            if (res) {
                rtn = -1;
                goto dohelp;
            }
        }
    }

	if (optind == argc) {
		printf("ERROR: You didn't specify any files to process.\n");
		help = TRUE;
	}

	if (help) {
    dohelp:
		print_help(args[0], opts);
		exit(rtn);
	}












    /*









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

        case '?':
            printf("\nTry \"--help\" to get a list of options.\n");
            exit(-1);
		}
	}
     */

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

	// number of backend args not specific to a particular file
	nbeargs = sl_size(backendargs);

	f = optind;
    inputnum = 0;
	while (1) {
		char* infile = NULL;
		bool isxyls;
		char* reason;
		int len;
		char* cpy;
		char* base;
		char *objsfn, *redgreenfn;
		char *ngcfn, *ppmfn=NULL, *indxylsfn;
        char* downloadfn;
        char* suffix = NULL;
		sl* outfiles;
		sl* tempfiles;
		sl* cmdline;
        bool ctrlc;
        augment_xylist_t theaxy;
        augment_xylist_t* axy = &theaxy;

        // reset augment-xylist args.
        memcpy(axy, allaxy, sizeof(augment_xylist_t));

		if (fromstdin) {
            char fnbuf[1024];
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

		axy->outfn    = sl_appendf(outfiles, "%s.axy",       base);
		axy->matchfn  = sl_appendf(outfiles, "%s.match",     base);
		axy->rdlsfn   = sl_appendf(outfiles, "%s.rdls",      base);
		axy->solvedfn = sl_appendf(outfiles, "%s.solved",    base);
		axy->wcsfn    = sl_appendf(outfiles, "%s.wcs",       base);
		objsfn     = sl_appendf(outfiles, "%s-objs.png",  base);
		redgreenfn = sl_appendf(outfiles, "%s-indx.png",  base);
		ngcfn      = sl_appendf(outfiles, "%s-ngc.png",   base);
		indxylsfn  = sl_appendf(outfiles, "%s-indx.xyls", base);
        if (suffix)
            downloadfn = sl_appendf(outfiles, "%s-downloaded.%s", base, suffix);
        else
            downloadfn = sl_appendf(outfiles, "%s-downloaded", base);


        if (solvedin || solvedindir) {
            if (solvedin && solvedindir)
                asprintf(&axy->solvedinfn, "%s/%s", solvedindir, solvedin);
            else if (solvedin)
                axy->solvedinfn = strdup(solvedin);
            else {
                char* bc = strdup(base);
                char* bn = strdup(basename(bc));
                asprintf(&axy->solvedinfn, "%s/%s.solved", solvedindir, bn);
                free(bn);
                free(bc);
            }
        }
        if (axy->solvedinfn && (strcmp(axy->solvedfn, axy->solvedinfn) == 0)) {
            // solved input and output files are the same: don't delete the input!
            sl_pop(outfiles);
            // MEMLEAK
        }

		free(base);
		base = NULL;

        if (skip_solved) {
            if (axy->solvedinfn) {
                if (verbose)
                    printf("Checking for solved file %s\n", axy->solvedinfn);
                if (file_exists(axy->solvedinfn)) {
                    printf("Solved file exists: %s; skipping this input file.\n", axy->solvedinfn);
                    goto nextfile;
                }
            }
            if (verbose)
                printf("Checking for solved file %s\n", axy->solvedfn);
            if (file_exists(axy->solvedfn)) {
                printf("Solved file exists: %s; skipping this input file.\n", axy->solvedfn);
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
			axy->xylsfn = infile;
		} else {
			axy->imagefn = infile;
		}

		if (axy->imagefn) {
            ppmfn = create_temp_file("ppm", tempdir);
            sl_append_nocopy(tempfiles, ppmfn);

            axy->pnmfn = ppmfn;
            axy->force_ppm = TRUE;
		}

        if (augment_xylist(axy, me)) {
            ERROR("augment-xylist failed");
            exit(-1);
        }

        if (makeplots) {
            // source extraction overlay
            // plotxy -i harvard.axy -I /tmp/pnm -C red -P -w 2 -N 50 | plotxy -w 2 -r 3 -I - -i harvard.axy -C red -n 50 > harvard-objs.png
            append_executable(cmdline, "plotxy", me);
            sl_append(cmdline, "-i");
            append_escape(cmdline, axy->outfn);
            if (axy->imagefn) {
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
            append_escape(cmdline, axy->outfn);
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

		append_escape(backendargs, axy->outfn);
		cmd = sl_implode(backendargs, " ");
        if (verbose)
            printf("Running:\n  %s\n", cmd);
        else
            printf("Solving...\n");
        fflush(NULL);
		if (run_command_get_outputs(cmd, NULL, NULL)) {
            fflush(NULL);
            ERROR("backend failed.  Command that failed was:\n  %s", cmd);
			exit(-1);
		}
		free(cmd);
        fflush(NULL);

		if (!file_exists(axy->solvedfn)) {
			// boo hoo.
			//printf("Field didn't solve.\n");
		} else {
			matchfile* mf;
			MatchObj* mo;
            sip_t wcs;
            double ra, dec, fieldw, fieldh;
            char rastr[32], decstr[32];
            char* fieldunits;

			// index rdls to xyls.
            if (wcs_rd2xy(axy->wcsfn, axy->rdlsfn, indxylsfn,
                          NULL, NULL, FALSE, NULL)) {
                ERROR("Failed to project index stars into field coordinates using wcs-rd2xy");
                exit(-1);
            }

            // print info about the field.
            if (!sip_read_header_file(axy->wcsfn, &wcs)) {
                ERROR("Failed to read WCS header from file %s", axy->wcsfn);
                exit(-1);
            }
            sip_get_radec_center(&wcs, &ra, &dec);
            sip_get_radec_center_hms_string(&wcs, rastr, decstr);
            sip_get_field_size(&wcs, &fieldw, &fieldh, &fieldunits);
            printf("Field center: (RA,Dec) = (%.4g, %.4g) deg.\n", ra, dec);
            printf("Field center: (RA H:M:S, Dec D:M:S) = (%s, %s).\n", rastr, decstr);
            printf("Field size: %g x %g %s\n", fieldw, fieldh, fieldunits);

            if (makeplots) {
                // sources + index overlay
                append_executable(cmdline, "plotxy", me);
                sl_append(cmdline, "-i");
                append_escape(cmdline, axy->outfn);
                if (axy->imagefn) {
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

                mf = matchfile_open(axy->matchfn);
                if (!mf) {
                    fprintf(stderr, "Failed to read matchfile %s.\n", axy->matchfn);
                    exit(-1);
                }
                // just read the first match...
                mo = matchfile_read_match(mf);
                if (!mo) {
                    fprintf(stderr, "Failed to read a match from matchfile %s.\n", axy->matchfn);
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

            if (axy->imagefn && makeplots) {
                sl* lines;

                append_executable(cmdline, "plot-constellations", me);
                if (verbose)
                    sl_append(cmdline, "-v");
				sl_append(cmdline, "-w");
				append_escape(cmdline, axy->wcsfn);
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
                if (run_command_get_outputs(cmd, &lines, NULL)) {
                    fflush(NULL);
                    ERROR("plot-constellations failed");
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
        free(axy->solvedinfn);
		for (i=0; i<sl_size(tempfiles); i++) {
			char* fn = sl_get(tempfiles, i);
			if (unlink(fn))
				fprintf(stderr, "Failed to delete temp file \"%s\": %s.\n", fn, strerror(errno));
		}
        sl_free2(cmdline);
		sl_free2(outfiles);
		sl_free2(tempfiles);
	}

	sl_free2(backendargs);
    free(me);

    augment_xylist_free_contents(allaxy);

	return 0;
}

