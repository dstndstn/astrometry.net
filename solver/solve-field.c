/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */


/**
 A command-line interface to the astrometry solver system.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <libgen.h>
#include <errors.h>
#include <getopt.h>
#include <assert.h>

#include "boilerplate.h"
#include "an-bool.h"
#include "bl.h"
#include "ioutils.h"
#include "fileutils.h"
#include "xylist.h"
#include "matchfile.h"
#include "fitsioutils.h"
#include "augment-xylist.h"
#include "an-opts.h"
#include "log.h"
#include "errors.h"
#include "anqfits.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "wcs-rd2xy.h"
#include "new-wcs.h"
#include "scamp.h"

static an_option_t options[] = {
    {'h', "help",		   no_argument, NULL,
     "print this help message" },
    {'\x95', "version", no_argument, NULL, "print version string and exit"},
    {'v', "verbose",       no_argument, NULL,
     "be more chatty -- repeat for even more verboseness" },
    {'D', "dir", required_argument, "directory",
     "place all output files in the specified directory"},
    {'o', "out", required_argument, "base-filename",
     "name the output files with this base name"},
    // DEPRECATED
    {'b', "backend-config", required_argument, "filename",
     "use this config file for the \"astrometry-engine\" program"},
    {'\x89', "config", required_argument, "filename",
     "use this config file for the \"astrometry-engine\" program"},
    {'\x96', "index-dir", required_argument, "dirname",
     "search for index files in the given directory, for \"astrometry-engine\""},
    {'\x97', "index-file", required_argument, "filename",
     "add this index file to the \"astrometry-engine\" program; you can quote this and include wildcards."},
    {'(', "batch",  no_argument, NULL,
     "run astrometry-engine once, rather than once per input file"},
    {'f', "files-on-stdin", no_argument, NULL,
     "read filenames to solve on stdin, one per line"},
    {'p', "no-plots",       no_argument, NULL,
     "don't create any plots of the results"},
    {'\x84', "plot-scale",  required_argument, "scale",
     "scale the plots by this factor (eg, 0.25)"},
    {'\x85', "plot-bg",  required_argument, "filename (JPEG)",
     //.jpg, .jpeg, .ppm, .pnm, .png)",
     "set the background image to use for plots"},
    {'G', "use-wget",       no_argument, NULL,
     "use wget instead of curl"},
    {'O', "overwrite",      no_argument, NULL,
     "overwrite output files if they already exist"},
    {'K', "continue",       no_argument, NULL,
     "don't overwrite output files if they already exist; continue a previous run"},
    {'J', "skip-solved",    no_argument, NULL,
     "skip input files for which the 'solved' output file already exists;"
     " NOTE: this assumes single-field input files"},
    {'\x87', "fits-image", no_argument, NULL,
     "assume the input files are FITS images"},
    {'N', "new-fits",    required_argument, "filename",
     "output filename of the new FITS file containing the WCS header; \"none\" to not create this file"},
    {'Z', "kmz",            required_argument, "filename",
     "create KMZ file for Google Sky.  (requires wcs2kml)"},
    {'i', "scamp",          required_argument, "filename",
     "create image object catalog for SCAMP"},
    {'n', "scamp-config",   required_argument, "filename",
     "create SCAMP config file snippet"},
    {'U', "index-xyls",     required_argument, "filename",
     "output filename for xylist containing the image coordinate of stars from the index"},
    {'@', "just-augment",   no_argument, NULL,
     "just write the augmented xylist files; don't run astrometry-engine."},
    {'\x91', "axy", required_argument, "filename",
     "output filename for augment xy list (axy)"},
    {'\x90', "temp-axy", no_argument, NULL,
     "write 'augmented xy list' (axy) file to a temp file"},
    {'\x88', "timestamp", no_argument, NULL,
     "add timestamps to log messages"},
};

static void print_help(const char* progname, bl* opts) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage:   %s [options]  [<image-file-1> <image-file-2> ...] [<xyls-file-1> <xyls-file-2> ...]\n"
           "\n"
           "You can specify http:// or https:// or ftp:// URLs instead of filenames.  The \"wget\" or \"curl\" program will be used to retrieve the URL.\n"
           "\n", progname);
    printf("Options include:\n");
    opts_print_help(opts, stdout, augment_xylist_print_special_opts, NULL);
    printf("\n");
    printf("Note that most output files can be disabled by setting the filename to \"none\".\n"
           " (If you have a sick sense of humour and you really want to name your output\n"
           "  file \"none\", you can use \"./none\" instead.)\n");
    printf("\n\n");
}

static int run_command(const char* cmd, anbool* ctrlc) {
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
    char* basekmlfn = NULL;
    char* basewarpedpngfn = NULL;
    char* tmpdir;
    char* cmd = NULL;
    sl* cmdline = sl_new(16);
    char* wcsbase = NULL;

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

    basekmlfn       = "doc.kml";
    basewarpedpngfn = "warped.png";
    kmlfn       = sl_appendf(tempfiles, "%s/%s", tmpdir, basekmlfn);
    warpedpngfn = sl_appendf(tempfiles, "%s/%s", tmpdir, basewarpedpngfn);
    // delete the wcs we create with "cp" below.
    assert(axy->wcsfn);
    wcsbase = basename_safe(axy->wcsfn);
    sl_appendf(tempfiles, "%s/%s", tmpdir, wcsbase);
    free(wcsbase);

    logverb("Trying to run wcs2kml to generate KMZ output.\n");
    sl_appendf(cmdline, "cp %s %s; cd %s; ", axy->wcsfn, tmpdir, tmpdir);
    sl_append(cmdline, "wcs2kml");
    // FIXME - if parity?
    sl_append(cmdline, "--input_image_origin_is_upper_left");
    appendf_escape(cmdline, "--fitsfile=%s", axy->wcsfn);
    appendf_escape(cmdline, "--imagefile=%s", pngfn);
    appendf_escape(cmdline, "--kmlfile=%s", basekmlfn);
    appendf_escape(cmdline, "--outfile=%s", basewarpedpngfn);
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

static int plot_source_overlay(const char* plotxy, augment_xylist_t* axy, const char* me,
                               const char* objsfn, double plotscale, const char* bgfn) {
    // plotxy -i harvard.axy -I /tmp/pnm -C red -P -w 2 -N 50 | plotxy -w 2 -r 3 -I - -i harvard.axy -C red -n 50 > harvard-objs.png
    sl* cmdline = sl_new(16);
    char* cmd;
    anbool ctrlc;
    char* imgfn;

    if (bgfn) {
        append_executable(cmdline, "jpegtopnm", me);
        append_escape(cmdline, bgfn);
        sl_append(cmdline, "|");
        imgfn = "-";
    } else {
        imgfn = axy->pnmfn;
        if (axy->imagefn && plotscale != 1.0) {
            append_executable(cmdline, "pnmscale", me);
            sl_appendf(cmdline, "%f", plotscale);
            append_escape(cmdline, axy->pnmfn);
            sl_append(cmdline, "|");
            imgfn = "-";
        }
    }

    append_executable(cmdline, plotxy, me);
    if (imgfn) {
        sl_append(cmdline, "-I");
        append_escape(cmdline, imgfn);
    } else {
        sl_appendf(cmdline, "-W %i -H %i", (int)(plotscale * axy->W), (int)(plotscale * axy->H));
    }
    sl_append(cmdline, "-i");
    append_escape(cmdline, axy->axyfn);
    if (axy->xcol) {
        sl_append(cmdline, "-X");
        append_escape(cmdline, axy->xcol);
    }
    if (axy->ycol) {
        sl_append(cmdline, "-Y");
        append_escape(cmdline, axy->ycol);
    }
    if (plotscale != 1.0) {
        sl_append(cmdline, "-S");
        sl_appendf(cmdline, "%f", plotscale);
    }
    sl_append(cmdline, "-C red -w 2 -N 50 -x 1 -y 1");
    sl_append(cmdline, "-P");
    sl_append(cmdline, "|");

    append_executable(cmdline, plotxy, me);
    sl_append(cmdline, "-i");
    append_escape(cmdline, axy->axyfn);
    if (axy->xcol) {
        sl_append(cmdline, "-X");
        append_escape(cmdline, axy->xcol);
    }
    if (axy->ycol) {
        sl_append(cmdline, "-Y");
        append_escape(cmdline, axy->ycol);
    }
    sl_append(cmdline, "-I - -w 2 -r 3 -C red -n 50 -N 200 -x 1 -y 1");
    if (plotscale != 1.0) {
        sl_append(cmdline, "-S");
        sl_appendf(cmdline, "%f", plotscale);
    }

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

static int plot_index_overlay(const char* plotxy, augment_xylist_t* axy, const char* me,
                              const char* indxylsfn, const char* redgreenfn,
                              double plotscale, const char* bgfn) {
    sl* cmdline = sl_new(16);
    char* cmd;
    matchfile* mf;
    MatchObj* mo;
    int i;
    anbool ctrlc;
    char* imgfn;
    char* plotquad = NULL;

    assert(axy->matchfn);
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
    imgfn = axy->pnmfn;

    if (bgfn) {
        append_executable(cmdline, "jpegtopnm", me);
        append_escape(cmdline, bgfn);
        sl_append(cmdline, "|");
        imgfn = "-";
    } else {
        if (axy->imagefn && plotscale != 1.0) {
            append_executable(cmdline, "pnmscale", me);
            sl_appendf(cmdline, "%f", plotscale);
            append_escape(cmdline, axy->pnmfn);
            sl_append(cmdline, "|");
            imgfn = "-";
        }
    }

    append_executable(cmdline, plotxy, me);
    if (imgfn) {
        sl_append(cmdline, "-I");
        append_escape(cmdline, imgfn);
    } else {
        sl_appendf(cmdline, "-W %i -H %i", (int)(plotscale * axy->W), (int)(plotscale * axy->H));
    }
    sl_append(cmdline, "-i");
    append_escape(cmdline, axy->axyfn);
    if (axy->xcol) {
        sl_append(cmdline, "-X");
        append_escape(cmdline, axy->xcol);
    }
    if (axy->ycol) {
        sl_append(cmdline, "-Y");
        append_escape(cmdline, axy->ycol);
    }
    if (plotscale != 1.0) {
        sl_append(cmdline, "-S");
        sl_appendf(cmdline, "%f", plotscale);
    }
    sl_append(cmdline, "-C red -w 2 -r 6 -N 200 -x 1 -y 1");
    sl_append(cmdline, "-P");
    sl_append(cmdline, "|");
    append_executable(cmdline, plotxy, me);
    sl_append(cmdline, "-i");
    append_escape(cmdline, indxylsfn);
    sl_append(cmdline, "-I - -w 2 -r 4 -C green -x 1 -y 1");
    if (plotscale != 1.0) {
        sl_append(cmdline, "-S");
        sl_appendf(cmdline, "%f", plotscale);
    }

    // if we solved by verifying an existing WCS, there is no quad.
    if (mo->dimquads > 0) {
	plotquad = find_executable("plotquad", me);
	if (!plotquad) {
	    // Try ../plot/plotquad
	    plotquad = find_executable("../plot/plotquad", me);
	}
	if (plotquad) {
            sl_append(cmdline, " -P |");
            append_executable(cmdline, plotquad, me);
            sl_append(cmdline, "-I -");
            sl_append(cmdline, "-C green");
            sl_append(cmdline, "-w 2");
            sl_appendf(cmdline, "-d %i", mo->dimquads);
            if (plotscale != 1.0) {
                sl_append(cmdline, "-s");
                sl_appendf(cmdline, "%f", plotscale);
            }
            for (i=0; i<(2 * mo->dimquads); i++)
                sl_appendf(cmdline, " %g", mo->quadpix_orig[i]);
	}
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
    free(plotquad);
    return 0;
}

static int plot_annotations(augment_xylist_t* axy, const char* me, anbool verbose,
                            const char* annfn, double plotscale, const char* bgfn) {
    sl* cmdline = sl_new(16);
    char* cmd;
    sl* lines;
    char* imgfn;
    char* plotconst = NULL;

    imgfn = axy->pnmfn;
    if (bgfn) {
        append_executable(cmdline, "jpegtopnm", me);
        append_escape(cmdline, bgfn);
        sl_append(cmdline, "|");
        imgfn = "-";
    } else if (axy->imagefn && plotscale != 1.0) {
        append_executable(cmdline, "pnmscale", me);
        sl_appendf(cmdline, "%f", plotscale);
        append_escape(cmdline, axy->pnmfn);
        sl_append(cmdline, "|");
        imgfn = "-";
    }

    plotconst = find_executable("plot-constellations", me);
    if (!plotconst) {
	// Try ../plot/
	plotconst = find_executable("../plot/plot-constellations", me);
    }
    if (!plotconst) {
	logerr("Failed to find plot-constellations program, not creating overlay plot.");
	return -1;
    }
    append_executable(cmdline, plotconst, me);
    if (verbose)
        sl_append(cmdline, "-v");
    sl_append(cmdline, "-w");
    assert(axy->wcsfn);
    append_escape(cmdline, axy->wcsfn);

    if (imgfn) {
	sl_append(cmdline, "-i");
	append_escape(cmdline, imgfn);
    }
    if (plotscale != 1.0) {
        sl_append(cmdline, "-s");
        sl_appendf(cmdline, "%f", plotscale);
    }
    sl_append(cmdline, "-N");
    sl_append(cmdline, "-B");
    sl_append(cmdline, "-C");
    sl_append(cmdline, "-o");
    assert(annfn);
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
    free(plotconst);
    return 0;
}

// "none" => NULL
static char* none_is_null(char* in) {
    return streq(in, "none") ? NULL : in;
}

static void run_engine(sl* engineargs) {
    char* cmd;
    cmd = sl_implode(engineargs, " ");
    logmsg("Solving...\n");
    logverb("Running:\n  %s\n", cmd);
    fflush(NULL);
    if (run_command_get_outputs(cmd, NULL, NULL)) {
        ERROR("engine failed.  Command that failed was:\n  %s", cmd);
        exit(-1);
    }
    free(cmd);
    fflush(NULL);
}

struct solve_field_args {
    char* newfitsfn;
    char* indxylsfn;
    char* redgreenfn;
    char* ngcfn;
    char* kmzfn;
    char* scampfn;
    char* scampconfigfn;
};
typedef struct solve_field_args solve_field_args_t;


// This runs after "astrometry-engine" is run on the file.
static void after_solved(augment_xylist_t* axy,
                         solve_field_args_t* sf,
                         anbool makeplots,
                         const char* me,
                         anbool verbose,
                         const char* tempdir,
                         sl* tempdirs,
                         sl* tempfiles,
			 const char* plotxy,
                         double plotscale,
                         const char* bgfn) {
    sip_t wcs;
    double ra, dec, fieldw, fieldh;
    char rastr[32], decstr[32];
    char* fieldunits;

    // print info about the field.
    logmsg("Field: %s\n", axy->imagefn ? axy->imagefn : axy->xylsfn);
    if (file_exists(axy->wcsfn)) {
        double orient;
        if (axy->wcs_last_mod) {
            time_t t = file_get_last_modified_time(axy->wcsfn);
            if (t == axy->wcs_last_mod) {
                logmsg("Warning: there was already a WCS file, and its timestamp has not changed.\n");
            }
        }
        if (!sip_read_header_file(axy->wcsfn, &wcs)) {
            ERROR("Failed to read WCS header from file %s", axy->wcsfn);
            exit(-1);
        }
        sip_get_radec_center(&wcs, &ra, &dec);
        sip_get_radec_center_hms_string(&wcs, rastr, decstr);
        sip_get_field_size(&wcs, &fieldw, &fieldh, &fieldunits);
        orient = sip_get_orientation(&wcs);
        logmsg("Field center: (RA,Dec) = (%3.6f, %3.6f) deg.\n", ra, dec);
        logmsg("Field center: (RA H:M:S, Dec D:M:S) = (%s, %s).\n", rastr, decstr);
        logmsg("Field size: %g x %g %s\n", fieldw, fieldh, fieldunits);
        logmsg("Field rotation angle: up is %g degrees E of N\n", orient);
        // Note, negative determinant = positive parity.
        double det = sip_det_cd(&wcs);
        logmsg("Field parity: %s\n", (det < 0 ? "pos" : "neg"));

    } else {
        logmsg("Did not solve (or no WCS file was written).\n");
    }

    // create new FITS file...
    if (axy->fitsimgfn && sf->newfitsfn && file_exists(axy->wcsfn)) {
        int ext = axy->isfits ? axy->fitsimgext : 0;
        logmsg("Creating new FITS file \"%s\"...\n", sf->newfitsfn);
        //logmsg("From image %s, ext
        if (new_wcs(axy->fitsimgfn, ext, axy->wcsfn,
                    sf->newfitsfn, TRUE)) {
            ERROR("Failed to create FITS image with new WCS headers");
            exit(-1);
        }
    }

    // write list of index stars in image coordinates
    if (sf->indxylsfn && file_exists(axy->wcsfn) && file_exists(axy->rdlsfn)) {
        assert(axy->wcsfn);
        assert(axy->rdlsfn);
        // index rdls to xyls.
        if (wcs_rd2xy(axy->wcsfn, 0, axy->rdlsfn, sf->indxylsfn,
                      NULL, NULL, FALSE, FALSE, NULL)) {
            ERROR("Failed to project index stars into field coordinates using wcs-rd2xy");
            exit(-1);
        }
    }

    if (makeplots && file_exists(sf->indxylsfn) && file_readable(axy->matchfn) && file_readable(axy->wcsfn)) {
        logmsg("Creating index object overlay plot...\n");
        if (plot_index_overlay(plotxy, axy, me, sf->indxylsfn, sf->redgreenfn, plotscale, bgfn)) {
            ERROR("Plot index overlay failed.");
        }
    }

    if (makeplots && file_readable(axy->wcsfn)) {
        logmsg("Creating annotation plot...\n");
        if (plot_annotations(axy, me, verbose, sf->ngcfn, plotscale, bgfn)) {
            ERROR("Plot annotations failed.");
        }
    }

    if (axy->imagefn && sf->kmzfn && file_exists(axy->wcsfn)) {
        logmsg("Writing kmz file...\n");
        if (write_kmz(axy, sf->kmzfn, tempdir, tempdirs, tempfiles)) {
            ERROR("Failed to write KMZ.");
            exit(-1);
        }
    }

    if (sf->scampfn && file_exists(axy->wcsfn)) {
        //char* hdrfile = NULL;
        qfits_header* imageheader = NULL;
        starxy_t* xy;
        xylist_t* xyls;

        xyls = xylist_open(axy->axyfn);
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

        if (axy->fitsimgfn) {
            //hdrfile = axy->fitsimgfn;
            imageheader = anqfits_get_header2(axy->fitsimgfn, 0);
        }
        if (axy->xylsfn) {
            char val[32];
            //hdrfile = axy->xylsfn;
            imageheader = anqfits_get_header2(axy->xylsfn, 0);
            // Set NAXIS=2, NAXIS1=IMAGEW, NAXIS2=IMAGEH
            fits_header_mod_int(imageheader, "NAXIS", 2, NULL);
            sprintf(val, "%i", axy->W);
            qfits_header_add_after(imageheader, "NAXIS",  "NAXIS1", val, "image width", NULL);
            sprintf(val, "%i", axy->H);
            qfits_header_add_after(imageheader, "NAXIS1", "NAXIS2", val, "image height", NULL);
            //fits_header_add_int(imageheader, "NAXIS1", axy->W, NULL);
            //fits_header_add_int(imageheader, "NAXIS2", axy->H, NULL);
            logverb("Using NAXIS 1,2 = %i,%i\n", axy->W, axy->H);
        }

        if (scamp_write_field(imageheader, &wcs, xy, sf->scampfn)) {
            ERROR("Failed to write SCAMP catalog");
            exit(-1);
        }
        starxy_free(xy);
        if (imageheader)
            qfits_header_destroy(imageheader);
    }

    if (sf->scampconfigfn) {
        if (scamp_write_config_file(axy->scampfn, sf->scampconfigfn)) {
            ERROR("Failed to write SCAMP config file snippet to %s", sf->scampconfigfn);
            exit(-1);
        }
    }
}

static void delete_temp_files(sl* tempfiles, sl* tempdirs) {
    int i;
    if (tempfiles) {
        for (i=0; i<sl_size(tempfiles); i++) {
            char* fn = sl_get(tempfiles, i);
            logverb("Deleting temp file %s\n", fn);
            if (unlink(fn))
                SYSERROR("Failed to delete temp file \"%s\"", fn);
        }
        sl_remove_all(tempfiles);
    }
    if (tempdirs) {
        for (i=0; i<sl_size(tempdirs); i++) {
            char* fn = sl_get(tempdirs, i);
            logverb("Deleting temp dir %s\n", fn);
            if (rmdir(fn))
                SYSERROR("Failed to delete temp dir \"%s\"", fn);
        }
        sl_remove_all(tempdirs);
    }
}


int main(int argc, char** args) {
    int c;
    anbool help = FALSE;
    char* outdir = NULL;
    char* cmd;
    int i, j, f;
    int inputnum;
    int rtn;
    sl* engineargs;
    int nbeargs;
    anbool fromstdin = FALSE;
    anbool overwrite = FALSE;
    anbool cont = FALSE;
    anbool skip_solved = FALSE;
    anbool makeplots = TRUE;
    double plotscale = 1.0;
    char* inbgfn = NULL;
    char* bgfn = NULL;
    char* me;
    anbool verbose = FALSE;
    int loglvl = LOG_MSG;
    char* outbase = NULL;
    anbool usecurl = TRUE;
    bl* opts;
    augment_xylist_t theallaxy;
    augment_xylist_t* allaxy = &theallaxy;
    int nmyopts;
    char* removeopts = "ixo\x01";
    char* newfits;
    char* kmz = NULL;
    char* scamp = NULL;
    char* scampconfig = NULL;
    char* index_xyls;
    anbool just_augment = FALSE;
    anbool engine_batch = FALSE;
    bl* batchaxy = NULL;
    bl* batchsf = NULL;
    sl* outfiles;
    sl* tempfiles;
    // these are deleted after the outer loop over input files
    sl* tempfiles2;
    sl* tempdirs;
    anbool timestamp = FALSE;
    anbool tempaxy = FALSE;
    char* plotxy = NULL;

    errors_print_on_exit(stderr);
    fits_use_error_system();

    me = find_executable(args[0], NULL);

    engineargs = sl_new(16);
    append_executable(engineargs, "astrometry-engine", me);

    // output filenames.
    outfiles = sl_new(16);
    tempfiles = sl_new(4);
    tempfiles2 = sl_new(4);
    tempdirs = sl_new(4);

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

    // which options are left?
    /*{
     char options[256];
     memset(options, 0, 256);
     printf("options:\n");
     for (i=0; i<bl_size(opts); i++) {
     an_option_t* opt = bl_access(opts, i);
     printf("  %c (%i) %s\n", opt->shortopt, (int)opt->shortopt, opt->name);
     options[(int)((opt->shortopt + 256) % 256)] = 1;
     }
     printf("Remaining short opts:\n");
     for (i=0; i<256; i++) {
     if (!options[i])
     printf("  %c (%i, 0x%x)\n", (char)i, i, i);
     }
     }*/

    augment_xylist_init(allaxy);

    // default output filename patterns.
    allaxy->axyfn    = "%s.axy";
    allaxy->matchfn  = "%s.match";
    allaxy->rdlsfn   = "%s.rdls";
    allaxy->solvedfn = "%s.solved";
    allaxy->wcsfn    = "%s.wcs";
    allaxy->corrfn   = "%s.corr";
    newfits          = "%s.new";
    index_xyls = "%s-indx.xyls";

    while (1) {
        int res;
        c = opts_getopt(opts, argc, args);
        //printf("option %c (%i)\n", c, (int)c);
        if (c == -1)
            break;
        switch (c) {
        case '\x91':
            allaxy->axyfn = optarg;
            break;
        case '\x90':
            tempaxy = TRUE;
            break;
        case '\x88':
            timestamp = TRUE;
            break;
        case '\x84':
            plotscale = atof(optarg);
            break;
        case '\x85':
            inbgfn = optarg;
            break;
        case '\x87':
            allaxy->assume_fits_image = TRUE;
            break;
        case '(':
            engine_batch = TRUE;
            break;
        case '@':
            just_augment = TRUE;
            break;
        case 'U':
            index_xyls = optarg;
            break;
        case 'n':
            scampconfig = optarg;
            break;
        case 'i':
            scamp = optarg;
            break;
        case 'Z':
            kmz = optarg;
            break;
        case 'N':
            newfits = optarg;
            break;
        case 'h':
            help = TRUE;
            break;
        case 'v':
            sl_append(engineargs, "--verbose");
            verbose = TRUE;
            allaxy->verbosity++;
            loglvl++;
            break;
        case 'D':
            outdir = optarg;
            break;
        case 'o':
            outbase = optarg;
            break;
        case 'b':
        case '\x89':
            sl_append(engineargs, "--config");
            append_escape(engineargs, optarg);
            break;
        case '\x96':
            sl_append(engineargs, "--index-dir");
            append_escape(engineargs, optarg);
            break;
        case '\x97':
            sl_append(engineargs, "--index");
            append_escape(engineargs, optarg);
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
        case '\x95':
            printf("%s\n", AN_GIT_REVISION);
            exit(rtn);
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

    // --dont-augment: advertised as just write xy file,
    // so quit after doing that.
    if (allaxy->dont_augment) {
        just_augment = TRUE;
    }

    log_init(loglvl);
    if (timestamp)
        log_set_timestamp(TRUE);

    if (kmz && starts_with(kmz, "-"))
        logmsg("Do you really want to save KMZ to the file named \"%s\" ??\n", kmz);

    if (starts_with(newfits, "-")) {
        logmsg("Do you really want to save the new FITS file to the file named \"%s\" ??\n", newfits);
    }

    if (engine_batch) {
        batchaxy = bl_new(16, sizeof(augment_xylist_t));
        batchsf  = bl_new(16, sizeof(solve_field_args_t));
    }

    // Allow (some of the) default filenames to be disabled by setting them to "none".
    allaxy->matchfn  = none_is_null(allaxy->matchfn);
    allaxy->rdlsfn   = none_is_null(allaxy->rdlsfn);
    allaxy->solvedfn = none_is_null(allaxy->solvedfn);
    allaxy->solvedinfn = none_is_null(allaxy->solvedinfn);
    allaxy->wcsfn    = none_is_null(allaxy->wcsfn);
    allaxy->corrfn   = none_is_null(allaxy->corrfn);
    newfits          = none_is_null(newfits);
    index_xyls = none_is_null(index_xyls);

    if (outdir) {
        if (mkdir_p(outdir)) {
            ERROR("Failed to create output directory %s", outdir);
            exit(-1);
        }
    }

    // number of engine args not specific to a particular file
    nbeargs = sl_size(engineargs);

    f = optind;
    inputnum = 0;
    while (1) {
        char* infile = NULL;
        anbool isxyls;
        char* reason;
        int len;
        char* base;
        char* basedir;
        char* basefile = NULL;
        char *objsfn=NULL;
        char *ppmfn=NULL;
        char* downloadfn = NULL;
        char* suffix = NULL;
        sl* cmdline;
        anbool ctrlc;
        anbool isurl;
        augment_xylist_t theaxy;
        augment_xylist_t* axy = &theaxy;
        int j;
        solve_field_args_t thesf;
        solve_field_args_t* sf = &thesf;
        anbool want_pnm = FALSE;

        // reset augment-xylist args.
        memcpy(axy, allaxy, sizeof(augment_xylist_t));

        memset(sf, 0, sizeof(solve_field_args_t));

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

        if (!engine_batch) {
            // Remove arguments that might have been added in previous trips through this loop
            sl_remove_from(engineargs,  nbeargs);
        }

        // Choose the base path/filename for output files.
        if (outbase)
            asprintf_safe(&basefile, outbase, inputnum, infile);
        else
            basefile = basename_safe(infile);
        //logverb("Base filename: %s\n", basefile);

        isurl = (!file_exists(infile) &&
                 (starts_with(infile, "http://") ||
                  starts_with(infile, "https://") ||
                  starts_with(infile, "ftp://")));

        if (outdir)
            basedir = strdup(outdir);
        else {
            if (isurl)
                basedir = strdup(".");
            else
                basedir = dirname_safe(infile);
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
                if (base[len - j] == '/')
                    break;
                if (base[len - j] == '.') {
                    base[len - j] = '\0';
                    suffix = base + len - j + 1;
                    break;
                }
            }
        }
        logverb("Base: \"%s\", basefile \"%s\", basedir \"%s\", suffix \"%s\"\n", base, basefile, basedir, suffix);

        if (tempaxy) {
            axy->axyfn = create_temp_file("axy", axy->tempdir);
            sl_append_nocopy(tempfiles2, axy->axyfn);
        } else
            axy->axyfn    = sl_appendf(outfiles, axy->axyfn,       base);
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
        if (axy->cancelfn)
            axy->cancelfn  = sl_appendf(outfiles, axy->cancelfn, base);
        if (axy->keepxylsfn)
            axy->keepxylsfn  = sl_appendf(outfiles, axy->keepxylsfn, base);
        if (axy->pnmfn)
            axy->pnmfn  = sl_appendf(outfiles, axy->pnmfn, base);
        if (newfits)
            sf->newfitsfn  = sl_appendf(outfiles, newfits,  base);
        if (kmz)
            sf->kmzfn = sl_appendf(outfiles, kmz, base);
        if (index_xyls)
            sf->indxylsfn  = sl_appendf(outfiles, index_xyls, base);
        if (scamp)
            sf->scampfn = sl_appendf(outfiles, scamp, base);
        if (scampconfig)
            sf->scampconfigfn = sl_appendf(outfiles, scampconfig, base);
        if (makeplots) {
            objsfn     = sl_appendf(outfiles, "%s-objs.png",  base);
            sf->redgreenfn = sl_appendf(outfiles, "%s-indx.png",  base);
            sf->ngcfn      = sl_appendf(outfiles, "%s-ngc.png",   base);
        }
        if (isurl) {
            if (suffix)
                downloadfn = sl_appendf(outfiles, "%s.%s", base, suffix);
            else
                downloadfn = sl_appendf(outfiles, "%s", base);
        }

        if (axy->solvedinfn)
            asprintf_safe(&axy->solvedinfn, axy->solvedinfn, base);

        // Do %s replacement on --verify-wcs entries...
        if (sl_size(axy->verifywcs)) {
            sl* newlist = sl_new(4);
            for (j=0; j<sl_size(axy->verifywcs); j++)
                sl_appendf(newlist, sl_get(axy->verifywcs, j), base);
            axy->verifywcs = newlist;
        }

        // ... and plot-bg
        if (inbgfn)
            asprintf_safe(&bgfn, inbgfn, base);

        if (axy->solvedinfn && axy->solvedfn && streq(axy->solvedfn, axy->solvedinfn)) {
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

        // if we're making "redgreen" plot, we need:
        if (sf->redgreenfn) {
            // -- index xylist
            if (!sf->indxylsfn) {
                sf->indxylsfn = create_temp_file("indxyls", axy->tempdir);
                sl_append_nocopy(tempfiles, sf->indxylsfn);
            }
            // -- match file.
            if (!axy->matchfn) {
                axy->matchfn = create_temp_file("match", axy->tempdir);
                sl_append_nocopy(tempfiles, axy->matchfn);
            }
        }

        // if index xyls file is needed, we need:
        if (sf->indxylsfn) {
            // -- wcs
            if (!axy->wcsfn) {
                axy->wcsfn = create_temp_file("wcs", axy->tempdir);
                sl_append_nocopy(tempfiles, axy->wcsfn);
            }
            // -- rdls
            if (!axy->rdlsfn) {
                axy->rdlsfn = create_temp_file("rdls", axy->tempdir);
                sl_append_nocopy(tempfiles, axy->rdlsfn);
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

        if (makeplots)
            want_pnm = TRUE;

        if (axy->assume_fits_image) {
            axy->imagefn = infile;
            if (axy->pnmfn)
                want_pnm = TRUE;
        } else {
            logverb("Checking if file \"%s\" ext %i is xylist or image: ",
                    infile, axy->extension);
            fflush(NULL);
            reason = NULL;
            isxyls = xylist_is_file_xylist(infile, axy->extension,
                                           axy->xcol, axy->ycol, &reason);
            logverb(isxyls ? "xyls\n" : "image\n");
            if (!isxyls)
                logverb("  (not xyls because: %s)\n", reason);
            free(reason);
            fflush(NULL);

            if (isxyls) {
                axy->xylsfn = infile;
		want_pnm = FALSE;
	    } else {
                axy->imagefn = infile;
                want_pnm = TRUE;
            }
        }

        if (want_pnm && !axy->pnmfn) {
            ppmfn = create_temp_file("ppm", axy->tempdir);
            sl_append_nocopy(tempfiles, ppmfn);
            axy->pnmfn = ppmfn;
            axy->force_ppm = TRUE;
        }

        axy->keep_fitsimg = (newfits || scamp);

        if (augment_xylist(axy, me)) {
            ERROR("augment-xylist failed");
            exit(-1);
        }

        if (just_augment)
            goto nextfile;

        if (makeplots) {
            // Check that the plotting executables were built...
            plotxy = find_executable("plotxy", me);
            if (!plotxy) {
                // Try ../plot/plotxy
                plotxy = find_executable("../plot/plotxy", me);
                if (!plotxy) {
                    logmsg("Couldn't find \"plotxy\" executable - maybe you didn't build the plotting programs?\n");
                    logmsg("Disabling plots.\n");
                    makeplots = FALSE;
                }
            }
        }
        if (makeplots) {
            // source extraction overlay
            if (plot_source_overlay(plotxy, axy, me, objsfn, plotscale, bgfn))
                makeplots = FALSE;
        }

        append_escape(engineargs, axy->axyfn);

        if (file_readable(axy->wcsfn))
            axy->wcs_last_mod = file_get_last_modified_time(axy->wcsfn);
        else
            axy->wcs_last_mod = 0;

        if (!engine_batch) {
            run_engine(engineargs);
            after_solved(axy, sf, makeplots, me, verbose,
                         axy->tempdir, tempdirs, tempfiles, plotxy, plotscale, bgfn);
        } else {
            bl_append(batchaxy, axy);
            bl_append(batchsf,  sf );
        }
        fflush(NULL);

        // clean up and move on to the next file.
    nextfile:        
        free(base);
        sl_free2(cmdline);

        if (!engine_batch) {
            free(axy->fitsimgfn);
            free(axy->solvedinfn);
            free(bgfn);
            // erm.
            if (axy->verifywcs != allaxy->verifywcs)
                sl_free2(axy->verifywcs);
            sl_remove_all(outfiles);
            if (!axy->no_delete_temp)
                delete_temp_files(tempfiles, tempdirs);
        }
        errors_print_stack(stdout);
        errors_clear_stack();
        logmsg("\n");
    }

    if (engine_batch) {
        run_engine(engineargs);
        for (i=0; i<bl_size(batchaxy); i++) {
            augment_xylist_t* axy = bl_access(batchaxy, i);
            solve_field_args_t* sf = bl_access(batchsf, i);

            after_solved(axy, sf, makeplots, me, verbose,
                         axy->tempdir, tempdirs, tempfiles, plotxy, plotscale, bgfn);
            errors_print_stack(stdout);
            errors_clear_stack();
            logmsg("\n");

            free(axy->fitsimgfn);
            free(axy->solvedinfn);
            // erm.
            if (axy->verifywcs != allaxy->verifywcs)
                sl_free2(axy->verifywcs);
        }
        if (!allaxy->no_delete_temp)
            delete_temp_files(tempfiles, tempdirs);
        bl_free(batchaxy);
        bl_free(batchsf);
    }

    if (!allaxy->no_delete_temp)
        delete_temp_files(tempfiles2, NULL);

    free(plotxy);
    sl_free2(outfiles);
    sl_free2(tempfiles);
    sl_free2(tempfiles2);
    sl_free2(tempdirs);
    sl_free2(engineargs);
    free(me);
    augment_xylist_free_contents(allaxy);

    return 0;
}

