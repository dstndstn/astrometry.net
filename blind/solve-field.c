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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
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
#include "fitsioutils.h"
#include "augment-xylist.h"
#include "an-opts.h"
#include "log.h"
#include "errors.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "wcs-rd2xy.h"
#include "new-wcs.h"
#include "scamp.h"

// hvD:o:b:fpGOKJN:Z:i:L:H:u:d:l:rz:C:S:I:M:R:j:B:W:P:k:AV:ygTt:c:E:m:q:Q:F:w:e:2X:Y:s:an:U:

static an_option_t options[] = {
	{'h', "help",		   no_argument, NULL,
     "print this help message" },
	{'v', "verbose",       no_argument, NULL,
     "be more chatty -- repeat for even more verboseness" },
    {'D', "dir", required_argument, "directory",
     "place all output files in the specified directory"},
    {'o', "out", required_argument, "base-filename",
     "name the output files with this base name"},
    {'b', "backend-config", required_argument, "filename",
     "use this config file for the \"backend\" program"},
	{'f', "files-on-stdin", no_argument, NULL,
     "read filenames to solve on stdin, one per line"},
	{'p', "no-plots",       no_argument, NULL,
     "don't create any plots of the results"},
    {'G', "use-wget",       no_argument, NULL,
     "use wget instead of curl"},
  	{'O', "overwrite",      no_argument, NULL,
     "overwrite output files if they already exist"},
    {'K', "continue",       no_argument, NULL,
     "don't overwrite output files if they already exist; continue a previous run"},
    {'J', "skip-solved",    no_argument, NULL,
     "skip input files for which the 'solved' output file already exists;\n"
     "                  NOTE: this assumes single-field input files"},
    {'N', "new-fits",    required_argument, "filename",
     "output filename of the new FITS file containing the WCS header; \"none\" to not create this file"},
    {'Z', "kmz",            required_argument, "filename",
     "create KMZ file for Google Sky.  (requires wcs2kml)"},
    {'i', "scamp",          required_argument, "filename",
     "create image object catalog for SCAMP"},
    {'n', "scamp-config",   required_argument, "filename",
     "create SCAMP config file snippet"},
    {'U', "index-xyls",     required_argument, "filename",
     "xylist containing the image coordinate of stars from the index"},
};

static void print_help(const char* progname, bl* opts) {
	printf("\nUsage:   %s [options]  [<image-file-1> <image-file-2> ...] [<xyls-file-1> <xyls-file-2> ...]\n"
           "\n"
           "You can specify http:// or ftp:// URLs instead of filenames.  The \"wget\" or \"curl\" program will be used to retrieve the URL.\n"
	       "\n", progname);
    printf("Options include:\n");
    opts_print_help(opts, stdout, augment_xylist_print_special_opts, NULL);
    printf("\n");
    printf("Note that most output files can be disabled by setting the filename to \"none\".\n"
           " (If you have a sick sense of humour and you really want to name your output file \"none\",\n"
           "  you can use \"./none\" instead.)\n");
    printf("\n\n");
}

static int run_command(const char* cmd, bool* ctrlc) {
	int rtn;
    logverb("Running: %s\n", cmd);
    fflush(NULL);
	rtn = system(cmd);
    fflush(NULL);
	if (rtn == -1) {
		SYSERROR("Failed to run command \"%s\"", cmd);
		return -1;
	}
	if (WIFSIGNALED(rtn)) {
        if (ctrlc && (WTERMSIG(rtn) == SIGTERM))
            *ctrlc = TRUE;
		return -1;
	}
	rtn = WEXITSTATUS(rtn);
	if (rtn) {
		ERROR("Command exited with exit status %i", rtn);
        ERROR("Command was: \"%s\"\n", cmd);
    }
	return rtn;
}

static void append_escape(sl* list, const char* fn) {
    sl_append_nocopy(list, shell_escape(fn));
}
static void appendf_escape(sl* list, const char* fmt, const char* fn) {
    char* esc = shell_escape(fn);
    sl_appendf(list, fmt, esc);
    free(esc);
}
static void append_executable(sl* list, const char* fn, const char* me) {
    char* exec = find_executable(fn, me);
    if (!exec) {
        ERROR("Error, couldn't find executable \"%s\"", fn);
        exit(-1);
    }
    sl_append_nocopy(list, shell_escape(exec));
    free(exec);
}

static int write_kmz(const augment_xylist_t* axy, const char* kmzfn,
                     const char* tempdir, sl* tempdirs, sl* tempfiles) {
    char* pngfn = NULL;
    char* kmlfn = NULL;
    char* warpedpngfn = NULL;
    char* tmpdir;
    char* cmd = NULL;
    sl* cmdline = sl_new(16);
            
    tmpdir = create_temp_dir("kmz", tempdir);
    if (!tmpdir) {
        ERROR("Failed to create temp dir for KMZ output");
        sl_free2(cmdline);
        return -1;
    }
    sl_append_nocopy(tempdirs, tmpdir);

    pngfn = create_temp_file("png", tempdir);
    sl_append_nocopy(tempfiles, pngfn);

    sl_append(cmdline, "pnmtopng");
    append_escape(cmdline, axy->pnmfn);
    sl_append(cmdline, ">");
    append_escape(cmdline, pngfn);
    // run it
    cmd = sl_implode(cmdline, " ");
    sl_remove_all(cmdline);
    logverb("Running:\n  %s\n", cmd);
    if (run_command_get_outputs(cmd, NULL, NULL)) {
        ERROR("pnmtopng failed");
        free(cmd);
        sl_free2(cmdline);
        return -1;
    }
    free(cmd);

    kmlfn       = sl_appendf(tempfiles, "%s/%s", tmpdir, "doc.kml");
    warpedpngfn = sl_appendf(tempfiles, "%s/%s", tmpdir, "warped.png");

    logverb("Trying to run wcs2kml to generate KMZ output.\n");
    sl_append(cmdline, "wcs2kml");
    // FIXME - if parity?
    sl_append(cmdline, "--input_image_origin_is_upper_left");
    appendf_escape(cmdline, "--fitsfile=%s", axy->wcsfn);
    appendf_escape(cmdline, "--imagefile=%s", pngfn);
    appendf_escape(cmdline, "--kmlfile=%s", kmlfn);
    appendf_escape(cmdline, "--outfile=%s", warpedpngfn);
    // run it
    cmd = sl_implode(cmdline, " ");
    sl_remove_all(cmdline);
    logverb("Running:\n  %s\n", cmd);
    if (run_command_get_outputs(cmd, NULL, NULL)) {
        ERROR("wcs2kml failed");
        free(cmd);
        sl_free2(cmdline);
        return -1;
    }
    free(cmd);

    sl_append(cmdline, "zip");
    sl_append(cmdline, "-j"); // no paths, just filenames
    //if (!verbose)
    //sl_append(cmdline, "-q");
    // pipe to stdout, because zip likes to add ".zip" to the
    // output filename, and provides no way to turn off this
    // behaviour.
    sl_append(cmdline, "-");
    appendf_escape(cmdline, "%s", warpedpngfn);
    appendf_escape(cmdline, "%s", kmlfn);
    sl_append(cmdline, ">");
    append_escape(cmdline, kmzfn);

    // run it
    cmd = sl_implode(cmdline, " ");
    sl_remove_all(cmdline);
    logverb("Running:\n  %s\n", cmd);
    if (run_command_get_outputs(cmd, NULL, NULL)) {
        ERROR("zip failed");
        free(cmd);
        sl_free2(cmdline);
        return -1;
    }
    free(cmd);
    sl_free2(cmdline);
    return 0;
}

static int plot_source_overlay(augment_xylist_t* axy, const char* me,
                               const char* objsfn) {
    // plotxy -i harvard.axy -I /tmp/pnm -C red -P -w 2 -N 50 | plotxy -w 2 -r 3 -I - -i harvard.axy -C red -n 50 > harvard-objs.png
    sl* cmdline = sl_new(16);
    char* cmd;
    bool ctrlc;

    append_executable(cmdline, "plotxy", me);
    sl_append(cmdline, "-i");
    append_escape(cmdline, axy->outfn);
    if (axy->imagefn) {
        sl_append(cmdline, "-I");
        append_escape(cmdline, axy->pnmfn);
    }
    if (axy->xcol) {
        sl_append(cmdline, "-X");
        append_escape(cmdline, axy->xcol);
    }
    if (axy->ycol) {
        sl_append(cmdline, "-Y");
        append_escape(cmdline, axy->ycol);
    }
    sl_append(cmdline, "-P");
    sl_append(cmdline, "-C red -w 2 -N 50 -x 1 -y 1");
            
    sl_append(cmdline, "|");

    append_executable(cmdline, "plotxy", me);
    sl_append(cmdline, "-i");
    append_escape(cmdline, axy->outfn);
    if (axy->xcol) {
        sl_append(cmdline, "-X");
        append_escape(cmdline, axy->xcol);
    }
    if (axy->ycol) {
        sl_append(cmdline, "-Y");
        append_escape(cmdline, axy->ycol);
    }
    sl_append(cmdline, "-I - -w 2 -r 3 -C red -n 50 -N 200 -x 1 -y 1");

    sl_append(cmdline, ">");
    append_escape(cmdline, objsfn);

    cmd = sl_implode(cmdline, " ");
    sl_free2(cmdline);

    if (run_command(cmd, &ctrlc)) {
        ERROR("Plotting command %s", (ctrlc ? "was cancelled" : "failed"));
        if (!ctrlc) {
            errors_print_stack(stdout);
            errors_clear_stack();
        }
        free(cmd);
        return -1;
    }
    free(cmd);
    return 0;
}

static int plot_index_overlay(augment_xylist_t* axy, const char* me,
                              const char* indxylsfn, const char* redgreenfn) {
    sl* cmdline = sl_new(16);
    char* cmd;
    matchfile* mf;
    MatchObj* mo;
    int i;
    bool ctrlc;

    mf = matchfile_open(axy->matchfn);
    if (!mf) {
        ERROR("Failed to read matchfile %s", axy->matchfn);
        return -1;
    }
    // just read the first match...
    mo = matchfile_read_match(mf);
    if (!mo) {
        ERROR("Failed to read a match from matchfile %s", axy->matchfn);
        return -1;
    }

    // sources + index overlay
    append_executable(cmdline, "plotxy", me);
    sl_append(cmdline, "-i");
    append_escape(cmdline, axy->outfn);
    if (axy->imagefn) {
        sl_append(cmdline, "-I");
        append_escape(cmdline, axy->pnmfn);
    }
    if (axy->xcol) {
        sl_append(cmdline, "-X");
        append_escape(cmdline, axy->xcol);
    }
    if (axy->ycol) {
        sl_append(cmdline, "-Y");
        append_escape(cmdline, axy->ycol);
    }
    sl_append(cmdline, "-P");
    sl_append(cmdline, "-C red -w 2 -r 6 -N 200 -x 1 -y 1");
    sl_append(cmdline, "|");
    append_executable(cmdline, "plotxy", me);
    sl_append(cmdline, "-i");
    append_escape(cmdline, indxylsfn);
    sl_append(cmdline, "-I - -w 2 -r 4 -C green -x 1 -y 1");

    // if we solved by verifying an existing WCS, there is no quad.
    if (mo->dimquads) {
        sl_append(cmdline, " -P |");
        append_executable(cmdline, "plotquad", me);
        sl_append(cmdline, "-I -");
        sl_append(cmdline, "-C green");
        sl_append(cmdline, "-w 2");
        sl_appendf(cmdline, "-d %i", mo->dimquads);
        for (i=0; i<(2 * mo->dimquads); i++)
            sl_appendf(cmdline, " %g", mo->quadpix[i]);
    }

    matchfile_close(mf);
			
    sl_append(cmdline, ">");
    append_escape(cmdline, redgreenfn);
    
    cmd = sl_implode(cmdline, " ");
    sl_free2(cmdline);
    logverb("Running:\n  %s\n", cmd);
    if (run_command(cmd, &ctrlc)) {
        ERROR("Plotting commands %s; exiting.", (ctrlc ? "were cancelled" : "failed"));
        return -1;
    }
    free(cmd);
    return 0;
}

static int plot_annotations(augment_xylist_t* axy, const char* me, bool verbose,
                            const char* annfn) {
    sl* cmdline = sl_new(16);
    char* cmd;
    sl* lines;

    append_executable(cmdline, "plot-constellations", me);
    if (verbose)
        sl_append(cmdline, "-v");
    sl_append(cmdline, "-w");
    append_escape(cmdline, axy->wcsfn);
    sl_append(cmdline, "-i");
    append_escape(cmdline, axy->pnmfn);
    sl_append(cmdline, "-N");
    sl_append(cmdline, "-C");
    sl_append(cmdline, "-o");
    append_escape(cmdline, annfn);
    cmd = sl_implode(cmdline, " ");
    sl_free2(cmdline);
    logverb("Running:\n  %s\n", cmd);
    if (run_command_get_outputs(cmd, &lines, NULL)) {
        ERROR("plot-constellations failed");
        return -1;
    }
    free(cmd);
    if (lines && sl_size(lines)) {
        int i;
        if (strlen(sl_get(lines, 0))) {
            logmsg("Your field contains:\n");
            for (i=0; i<sl_size(lines); i++)
                logmsg("  %s\n", sl_get(lines, i));
        }
    }
    if (lines)
        sl_free2(lines);
    return 0;
}

// "none" => NULL
static char* none_is_null(char* in) {
    return streq(in, "none") ? NULL : in;
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
    int loglvl = LOG_MSG;
    char* outbase = NULL;
    char* solvedin = NULL;
    char* solvedindir = NULL;
	bool usecurl = TRUE;
    bl* opts;
    augment_xylist_t theallaxy;
    augment_xylist_t* allaxy = &theallaxy;
    int nmyopts;
    char* removeopts = "ixo\x01";
    char* newfits;
    char* kmzfn = NULL;
    char* scampfn = NULL;
    char* scampconfigfn = NULL;
    char* index_xyls_template;

    errors_print_on_exit(stderr);
    fits_use_error_system();

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

    // default output filename patterns.
    allaxy->outfn    = "%s.axy";
    allaxy->matchfn  = "%s.match";
    allaxy->rdlsfn   = "%s.rdls";
    allaxy->solvedfn = "%s.solved";
    allaxy->wcsfn    = "%s.wcs";
    allaxy->corrfn   = "%s.corr";
    newfits          = "%s.new";
    index_xyls_template = "%s-indx.xyls";

	while (1) {
        int res;
		c = opts_getopt(opts, argc, args);
        if (c == -1)
            break;
        switch (c) {
        case 'U':
            index_xyls_template = optarg;
            break;
        case 'n':
            scampconfigfn = optarg;
            break;
        case 'i':
            scampfn = optarg;
            break;
        case 'Z':
            kmzfn = optarg;
            break;
        case 'N':
            newfits = optarg;
            break;
		case 'h':
			help = TRUE;
			break;
        case 'v':
            sl_append(backendargs, "--verbose");
            verbose = TRUE;
            loglvl++;
            break;
		case 'D':
			outdir = optarg;
			break;
        case 'o':
            outbase = optarg;
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

	if ((optind == argc) && !fromstdin) {
		printf("ERROR: You didn't specify any files to process.\n");
		help = TRUE;
	}

	if (help) {
    dohelp:
		print_help(args[0], opts);
		exit(rtn);
	}
    bl_free(opts);

    log_init(loglvl);

    if (kmzfn && starts_with(kmzfn, "-")) {
        logmsg("Do you really want to save KMZ to the file named \"%s\" ??\n", kmzfn);
    }
    if (starts_with(newfits, "-")) {
        logmsg("Do you really want to save the new FITS file to the file named \"%s\" ??\n", newfits);
    }

    // Allow (some of the) default filenames to be disabled by setting them to "none".
    allaxy->matchfn  = none_is_null(allaxy->matchfn);
    allaxy->rdlsfn   = none_is_null(allaxy->rdlsfn);
    allaxy->solvedfn = none_is_null(allaxy->solvedfn);
    allaxy->wcsfn    = none_is_null(allaxy->wcsfn);
    allaxy->corrfn   = none_is_null(allaxy->corrfn);
    newfits          = none_is_null(newfits);
    index_xyls_template = none_is_null(index_xyls_template);

	if (outdir) {
        if (mkdir_p(outdir)) {
            ERROR("Failed to create output directory %s", outdir);
            exit(-1);
        }
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
        char* basedir;
        char* basefile;
		char *objsfn=NULL, *redgreenfn=NULL;
		char *ngcfn=NULL, *ppmfn=NULL, *indxylsfn=NULL;
        char* newfitsfn = NULL;
        char* downloadfn = NULL;
        char* suffix = NULL;
		sl* outfiles;
		sl* tempfiles;
		sl* tempdirs;
		sl* cmdline;
        bool ctrlc;
        bool isurl;
        augment_xylist_t theaxy;
        augment_xylist_t* axy = &theaxy;
        int j;

        // reset augment-xylist args.
        memcpy(axy, allaxy, sizeof(augment_xylist_t));

		if (fromstdin) {
            char fnbuf[1024];
			if (!fgets(fnbuf, sizeof(fnbuf), stdin)) {
				if (ferror(stdin))
					SYSERROR("Failed to read a filename from stdin");
				break;
			}
			len = strlen(fnbuf);
			if (fnbuf[len-1] == '\n')
				fnbuf[len-1] = '\0';
			infile = fnbuf;
            logmsg("Reading input file \"%s\"...\n", infile);
		} else {
			if (f == argc)
				break;
			infile = args[f];
			f++;
            logmsg("Reading input file %i of %i: \"%s\"...\n",
                   f - optind, argc - optind, infile);
		}
        inputnum++;

        cmdline = sl_new(16);

        // Remove arguments that might have been added in previous trips through this loop
		sl_remove_from(backendargs,  nbeargs);

		// Choose the base path/filename for output files.
        if (outbase)
            asprintf_safe(&basefile, outbase, inputnum, infile);
        else {
            cpy = strdup(infile);
            basefile = strdup(basename(cpy));
            free(cpy);
        }
        //logverb("Base filename: %s\n", basefile);

        isurl = (!file_exists(infile) &&
                 (starts_with(infile, "http://") ||
                  starts_with(infile, "ftp://")));

		if (outdir)
            basedir = strdup(outdir);
		else {
            if (isurl)
                basedir = strdup(".");
            else {
                cpy = strdup(infile);
                basedir = strdup(dirname(cpy));
                free(cpy);
            }
        }
        //logverb("Base directory: %s\n", basedir);

        asprintf_safe(&base, "%s/%s", basedir, basefile);
        //logverb("Base name for output files: %s\n", base);

        // trim .gz, .bz2
        // hmm, we drop the suffix in this case...
		len = strlen(base);
        if (ends_with(base, ".gz"))
            base[len-3] = '\0';
        else if (ends_with(base, ".bz2"))
            base[len-4] = '\0';
		len = strlen(base);
		// trim .xx / .xxx / .xxxx
		if (len >= 5) {
            for (j=3; j<=5; j++) {
                if (base[len - j] == '.') {
                    base[len - j] = '\0';
                    suffix = base + len - j + 1;
                    break;
                }
            }
		}
        logverb("Base: %s, basefile %s, basedir %s, suffix %s\n", base, basefile, basedir, suffix);

		// the output filenames.
		outfiles = sl_new(16);
		tempfiles = sl_new(4);
		tempdirs = sl_new(4);

		axy->outfn    = sl_appendf(outfiles, axy->outfn,       base);
        if (axy->matchfn)
            axy->matchfn  = sl_appendf(outfiles, axy->matchfn,     base);
        if (axy->rdlsfn)
            axy->rdlsfn   = sl_appendf(outfiles, axy->rdlsfn,      base);
        if (axy->solvedfn)
            axy->solvedfn = sl_appendf(outfiles, axy->solvedfn,    base);
        if (axy->wcsfn)
            axy->wcsfn    = sl_appendf(outfiles, axy->wcsfn,       base);
        if (axy->corrfn)
            axy->corrfn   = sl_appendf(outfiles, axy->corrfn,      base);
        if (newfits)
            newfitsfn  = sl_appendf(outfiles, newfits,  base);
        if (axy->cancelfn)
            axy->cancelfn  = sl_appendf(outfiles, axy->cancelfn, base);
        if (axy->keepxylsfn)
            axy->keepxylsfn  = sl_appendf(outfiles, axy->keepxylsfn, base);
        if (axy->pnmfn)
            axy->pnmfn  = sl_appendf(outfiles, axy->pnmfn, base);
        if (makeplots) {
            objsfn     = sl_appendf(outfiles, "%s-objs.png",  base);
            redgreenfn = sl_appendf(outfiles, "%s-indx.png",  base);
            ngcfn      = sl_appendf(outfiles, "%s-ngc.png",   base);
        }
        if (index_xyls_template)
            indxylsfn  = sl_appendf(outfiles, index_xyls_template, base);
        if (isurl) {
            if (suffix)
                downloadfn = sl_appendf(outfiles, "%s.%s", base, suffix);
            else
                downloadfn = sl_appendf(outfiles, "%s", base);
        }

        // Do %s replacement on --verify-wcs entries...
        if (sl_size(axy->verifywcs)) {
            sl* newlist = sl_new(4);
            for (j=0; j<sl_size(axy->verifywcs); j++)
                sl_appendf(newlist, sl_get(axy->verifywcs, j), base);
            axy->verifywcs = newlist;
        }

        if (solvedin || solvedindir) {
            char* dir = (solvedindir ? solvedindir : basedir);
            if (solvedin) {
                // "solvedin" might contain "%s"...
                char* tmpstr;
                asprintf(&tmpstr, "%s/%s", dir, solvedin);
                asprintf(&axy->solvedinfn, tmpstr, base);
                free(tmpstr);
            } else
                asprintf(&axy->solvedinfn, dir, axy->solvedfn);
        }
        if (axy->solvedinfn && axy->solvedfn && (strcmp(axy->solvedfn, axy->solvedinfn) == 0)) {
            // solved input and output files are the same: don't delete the input!
            sl_remove_string(outfiles, axy->solvedfn);
            free(axy->solvedfn);
            axy->solvedfn = axy->solvedinfn;
        }

        free(basedir);
        free(basefile);

        if (skip_solved) {
            char* tocheck[] = { axy->solvedinfn, axy->solvedfn };
            for (j=0; j<sizeof(tocheck)/sizeof(char*); j++) {
                if (!tocheck[j])
                    continue;
                logverb("Checking for solved file %s\n", tocheck[j]);
                if (file_exists(tocheck[j])) {
                    logmsg("Solved file exists: %s; skipping this input file.\n", tocheck[j]);
                    goto nextfile;
                } else {
                    logverb("File \"%s\" does not exist.\n", tocheck[j]);
                }
            }
        }

        // Check for overlap between input and output filenames
		for (i = 0; i < sl_size(outfiles); i++) {
			char* fn = sl_get(outfiles, i);
            if (streq(fn, infile)) {
                logmsg("Output filename \"%s\" is the same as your input file.\n"
                       "Refusing to continue.\n"
                       "You can either choose a different output filename, or\n"
                       "rename your input file to have a different extension.\n", fn);
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
					SYSERROR("Failed to delete an already-existing output file \"%s\"", fn);
					exit(-1);
				}
			} else {
				logmsg("Output file already exists: \"%s\".\n"
				       "Use the --overwrite flag to overwrite existing files,\n"
                       " or the --continue  flag to not overwrite existing files but still try solving.\n", fn);
				logmsg("Continuing to next input file.\n");
                goto nextfile;
			}
		}

        // Download URL...
        if (isurl) {

            sl_append(cmdline, usecurl ? "curl" : "wget");
            if (!verbose)
                sl_append(cmdline, usecurl ? "--silent" : "--quiet");
            sl_append(cmdline, usecurl ? "--output" : "-O");
            append_escape(cmdline, downloadfn);
            append_escape(cmdline, infile);

            cmd = sl_implode(cmdline, " ");

            logmsg("Downloading...\n");
            if (run_command(cmd, &ctrlc)) {
                ERROR("%s command %s", sl_get(cmdline, 0),
                      (ctrlc ? "was cancelled" : "failed"));
                exit(-1);
            }
            sl_remove_all(cmdline);
            free(cmd);

            infile = downloadfn;
        }

        logverb("Checking if file \"%s\" is xylist or image: ", infile);
        fflush(NULL);
        reason = NULL;
		isxyls = xylist_is_file_xylist(infile, axy->xcol, axy->ycol, &reason);
        logverb(isxyls ? "xyls\n" : "image\n");
        if (!isxyls)
            logverb("  (not xyls because: %s)\n", reason);
        free(reason);
        fflush(NULL);

		if (isxyls)
			axy->xylsfn = infile;
        else
			axy->imagefn = infile;

		if (axy->imagefn) {
            ppmfn = create_temp_file("ppm", tempdir);
            sl_append_nocopy(tempfiles, ppmfn);
            axy->pnmfn = ppmfn;
            axy->force_ppm = TRUE;
		}

        axy->keep_fitsimg = (newfits || scampfn);

        if (augment_xylist(axy, me)) {
            ERROR("augment-xylist failed");
            exit(-1);
        }

        if (makeplots) {
            // Check that the plotting executables were built...
            char* exec = find_executable("plotxy", me);
            free(exec);
            if (!exec) {
                logmsg("Couldn't find \"plotxy\" executable - maybe you didn't build the plotting programs?\n");
                logmsg("Disabling plots.\n");
                makeplots = FALSE;
            }
        }
        if (makeplots) {
            // source extraction overlay
            if (plot_source_overlay(axy, me, objsfn))
                makeplots = FALSE;
        }

		append_escape(backendargs, axy->outfn);
		cmd = sl_implode(backendargs, " ");

        logmsg("Solving...\n");
        logverb("Running:\n  %s\n", cmd);
        fflush(NULL);
        if (run_command_get_outputs(cmd, NULL, NULL)) {
            ERROR("backend failed.  Command that failed was:\n  %s", cmd);
			exit(-1);
		}
        free(cmd);
        fflush(NULL);

		if (!file_exists(axy->solvedfn)) {
			// boo hoo.
			//printf("Field didn't solve.\n");
		} else {
            sip_t wcs;
            double ra, dec, fieldw, fieldh;
            char rastr[32], decstr[32];
            char* fieldunits;

            // create new FITS file...
            if (axy->fitsimgfn && newfitsfn) {
                logmsg("Creating new FITS file \"%s\"...\n", newfitsfn);
                if (new_wcs(axy->fitsimgfn, axy->wcsfn, newfitsfn, TRUE)) {
                    ERROR("Failed to create FITS image with new WCS headers");
                    exit(-1);
                }
            }

            if (makeplots && !indxylsfn) {
                // write index xyls to temp file for overlay plot.
                indxylsfn = create_temp_file("indxyls", tempdir);
                sl_append_nocopy(tempfiles, indxylsfn);
            }

            if (indxylsfn) {
                // index rdls to xyls.
                if (wcs_rd2xy(axy->wcsfn, axy->rdlsfn, indxylsfn,
                              NULL, NULL, FALSE, NULL)) {
                    ERROR("Failed to project index stars into field coordinates using wcs-rd2xy");
                    exit(-1);
                }
            }

            // print info about the field.
            if (!sip_read_header_file(axy->wcsfn, &wcs)) {
                ERROR("Failed to read WCS header from file %s", axy->wcsfn);
                exit(-1);
            }
            sip_get_radec_center(&wcs, &ra, &dec);
            sip_get_radec_center_hms_string(&wcs, rastr, decstr);
            sip_get_field_size(&wcs, &fieldw, &fieldh, &fieldunits);
            logmsg("Field center: (RA,Dec) = (%.4g, %.4g) deg.\n", ra, dec);
            logmsg("Field center: (RA H:M:S, Dec D:M:S) = (%s, %s).\n", rastr, decstr);
            logmsg("Field size: %g x %g %s\n", fieldw, fieldh, fieldunits);

            if (makeplots) {
                logmsg("Creating plots...\n");
                if (plot_index_overlay(axy, me, indxylsfn, redgreenfn)) {
                    ERROR("Plot index overlay failed.");
                }
            }

            if (axy->imagefn && makeplots) {
                if (plot_annotations(axy, me, verbose, ngcfn)) {
                    ERROR("Plot annotations failed.");
                }

			}

            if (axy->imagefn && kmzfn) {
                char* realkmzfn;
                asprintf(&realkmzfn, kmzfn, base);
                if (write_kmz(axy, realkmzfn, tempdir, tempdirs, tempfiles)) {
                    ERROR("Failed to write KMZ.");
                    exit(-1);
                }
                free(realkmzfn);
            }

            if (scampfn) {
                char* hdrfile = NULL;
                qfits_header* imageheader = NULL;
                starxy_t* xy;
                xylist_t* xyls = xylist_open(axy->outfn);

                if (!xyls) {
                    ERROR("Failed to read xylist to write SCAMP catalog");
                    exit(-1);
                }
                if (axy->xcol)
                    xylist_set_xname(xyls, axy->xcol);
                if (axy->ycol)
                    xylist_set_yname(xyls, axy->ycol);
                //xylist_set_include_flux(xyls, FALSE);
                xylist_set_include_background(xyls, FALSE);
                xy = xylist_read_field(xyls, NULL);
                xylist_close(xyls);

                if (axy->fitsimgfn)
                    hdrfile = axy->fitsimgfn;
                if (axy->xylsfn)
                    hdrfile = axy->xylsfn;
                if (hdrfile)
                    imageheader = qfits_header_read(hdrfile);

                if (scamp_write_field(imageheader, &wcs, xy, scampfn)) {
                    ERROR("Failed to write SCAMP catalog");
                    exit(-1);
                }
                starxy_free(xy);
                if (imageheader)
                    qfits_header_destroy(imageheader);
            }

            if (scampconfigfn) {
                if (scamp_write_config_file(axy->scampfn, scampconfigfn)) {
                    ERROR("Failed to write SCAMP config file snippet to %s", scampconfigfn);
                    exit(-1);
                }
            }
		}
        fflush(NULL);

        // clean up and move on to the next file.
    nextfile:        
		free(base);
        free(axy->fitsimgfn);
        //free(axy->solvedinfn);
		for (i=0; i<sl_size(tempfiles); i++) {
			char* fn = sl_get(tempfiles, i);
			if (unlink(fn))
				SYSERROR("Failed to delete temp file \"%s\"", fn);
		}
		for (i=0; i<sl_size(tempdirs); i++) {
			char* fn = sl_get(tempdirs, i);
			if (rmdir(fn))
				SYSERROR("Failed to delete temp dir \"%s\"", fn);
		}
        // erm.
        if (axy->verifywcs != allaxy->verifywcs)
            sl_free2(axy->verifywcs);

        sl_free2(cmdline);
		sl_free2(outfiles);
		sl_free2(tempfiles);
		sl_free2(tempdirs);

        errors_print_stack(stdout);
        errors_clear_stack();
        logmsg("\n");
	}

	sl_free2(backendargs);
    free(me);

    augment_xylist_free_contents(allaxy);

	return 0;
}

