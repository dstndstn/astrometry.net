/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 * Accepts an image or xylist, plus command-line options, and produces
 * an augmented xylist.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>
#include <getopt.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "ioutils.h"
#include "fileutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "fitsioutils.h"
#include "solverutils.h"
#include "sip_qfits.h"
#include "tabsort.h"
#include "cut-table.h"
#include "errors.h"
#include "fits-guess-scale.h"
#include "image2xy-files.h"
#include "resort-xylist.h"
#include "an-opts.h"
#include "augment-xylist.h"
#include "log.h"
#include "anqfits.h"
#include "mathutil.h"

static void delete_existing_an_headers(qfits_header* hdr);

void augment_xylist_init(augment_xylist_t* axy) {
    memset(axy, 0, sizeof(augment_xylist_t));
    axy->tempdir = NULL;
    axy->tweak = TRUE;
    axy->tweakorder = 2;
    axy->depths = il_new(4);
    axy->fields = il_new(16);
    axy->verifywcs = sl_new(4);
    axy->verifywcs_ext = il_new(4);
    axy->tagalong = sl_new(4);
    axy->try_verify = TRUE;
    axy->resort = TRUE;
    axy->ra_center = LARGE_VAL;
    axy->dec_center = LARGE_VAL;
    axy->parity = PARITY_BOTH;
    axy->uniformize = 10;
    axy->verify_uniformize = TRUE;
}

void augment_xylist_free_contents(augment_xylist_t* axy) {
    sl_free2(axy->verifywcs);
    il_free(axy->verifywcs_ext);
    sl_free2(axy->tagalong);
    il_free(axy->depths);
    il_free(axy->fields);
    if (axy->predistort)
        sip_free(axy->predistort);
}

void augment_xylist_print_special_opts(an_option_t* opt, bl* opts, int index,
                                       FILE* fid, void* extra) {
    if (!strcmp(opt->name, "image")) {
        fprintf(fid, "%s",
                "  (   -i / --image  <image-input-file>\n"
                "   OR -x / --xylist <xylist-input-file>  ): input file\n");
    } else if (!strcmp(opt->name, "scale-units")) {
        fprintf(fid, "%s",
                "  -u / --scale-units <units>: in what units are the lower and upper bounds?\n"
                "     choices:  \"degwidth\", \"degw\", \"dw\"   : width of the image, in degrees (default)\n"
                "               \"arcminwidth\", \"amw\", \"aw\" : width of the image, in arcminutes\n"
                "               \"arcsecperpix\", \"app\": arcseconds per pixel\n"
                "               \"focalmm\": 35-mm (width-based) equivalent focal length\n"
                );
    } else if (!strcmp(opt->name, "depth")) {
        fprintf(fid, "%s",
                "  -d / --depth <number or range>: number of field objects to look at, or range\n"
                "          of numbers; 1 is the brightest star, so \"-d 10\" or \"-d 1-10\" mean look\n"
                "          at the top ten brightest stars only.\n");
    } else if (!strcmp(opt->name, "xylist-only")) {
        fprintf(fid, "%s",
                "The following options are valid for xylist inputs only:\n");
    } else if (!strcmp(opt->name, "fields")) {
        fprintf(fid, "%s",
                "  -F / --fields <number or range>: the FITS extension(s) to solve, inclusive\n");
    } else if (!strcmp(opt->name, "options")) {
        fprintf(fid, "Options include:\n");
    }
}

static an_option_t options[] = {
    {'i', "image",                 required_argument, NULL, NULL},
    {'x', "xylist",                required_argument, NULL, NULL},
    {'o', "out",                   required_argument, "filename",
     "output augmented xylist filename"},
    {'\x01', "options",       no_argument, NULL, NULL},
    {'h', "help",                  no_argument, NULL,
     "print this help message" },
    {'v', "verbose",       no_argument, NULL,
     "be more chatty" },
    {'7', "no-delete-temp", no_argument, NULL,
     "don't delete temp files (for debugging)\n"},
    {'L', "scale-low",     required_argument, "scale",
     "lower bound of image scale estimate"},
    {'H', "scale-high",   required_argument, "scale",
     "upper bound of image scale estimate"},
    {'u', "scale-units",    required_argument, "units", NULL},
    {'8', "parity",         required_argument, "pos/neg",
     "only check for matches with positive/negative parity (default: try both)"},
    {'c', "code-tolerance", required_argument, "distance",
     "matching distance for quads (default: 0.01)"},
    {'E', "pixel-error",    required_argument, "pixels",
     "for verification, size of pixel positional error (default: 1)"},
    {'q', "quad-size-min",  required_argument, "fraction",
     "minimum size of quads to try, as a fraction of the smaller image dimension, default: 0.1"},
    {'Q', "quad-size-max",  required_argument, "fraction",
     "maximum size of quads to try, as a fraction of the image hypotenuse, default 1.0"},
    {'\x82', "odds-to-tune-up",  required_argument, "odds",
     "odds ratio at which to try tuning up a match that isn't good enough to solve (default: 1e6)"},
    {'[', "odds-to-solve",  required_argument, "odds",
     "odds ratio at which to consider a field solved (default: 1e9)"},
    {'#', "odds-to-reject",   required_argument, "odds",
     "odds ratio at which to reject a hypothesis (default: 1e-100)"},
    {'%', "odds-to-stop-looking", required_argument, "odds",
     "odds ratio at which to stop adding stars when evaluating a hypothesis (default: LARGE_VAL)"},
    {'^', "use-source-extractor", no_argument, NULL,
     "use SourceExtractor rather than built-in image2xy to find sources"},
    {'&', "source-extractor-config", required_argument, "filename",
     "use the given SourceExtractor config file.  "
     "Note that CATALOG_NAME and CATALOG_TYPE values will be over-ridden by command-line values.  "
     "This option implies --use-source-extractor."},
    {'*', "source-extractor-path", required_argument, "filename",
     "use the given path to the SourceExtractor executable.  Default: just 'source-extractor', assumed to be in your PATH."
     "  Note that you can give command-line args here too (but put them in quotes), eg: --source-extractor-path 'source-extractor -DETECT_TYPE CCD'.  "
     "This option implies --use-source-extractor."},
    {'3', "ra",             required_argument, "degrees or hh:mm:ss",
     "only search in indexes within 'radius' of the field center given by 'ra' and 'dec'"},
    {'4', "dec",            required_argument, "degrees or [+-]dd:mm:ss",
     "only search in indexes within 'radius' of the field center given by 'ra' and 'dec'"},
    {'5', "radius",         required_argument, "degrees",
     "only search in indexes within 'radius' of the field center given by ('ra', 'dec')"},
    {'d', "depth",                 required_argument, NULL, NULL},
    {'|', "objs",           required_argument, "int",
     "cut the source list to have this many items (after sorting, if applicable)."},
    {'l', "cpulimit",       required_argument, "seconds",
     "give up solving after the specified number of seconds of CPU time"},
    {'r', "resort",         no_argument, NULL,
     "sort the star brightnesses by background-subtracted flux; the default is to sort using a"
     "compromise between background-subtracted and non-background-subtracted flux"},
    {'6', "extension",      required_argument, "int",
     "FITS extension to read image from."},
    {';', "invert",         no_argument, NULL,
     "invert the image (for black-on-white images)"},
    {'z', "downsample",     required_argument, "int",
     "downsample the image by factor <int> before running source extraction"},
    {']', "no-background-subtraction", no_argument, NULL,
     "don't try to estimate a smoothly-varying sky background during source extraction."},
    {'{', "sigma", required_argument, "float",
     "set the noise level in the image"},
    {'\x93', "nsigma", required_argument, "float",
     "number of sigma for a source detection; default 8 for FITS images, 4 for JPEG, etc"},
    {'9', "no-remove-lines", no_argument, NULL,
     "don't remove horizontal and vertical overdensities of sources."},
    {':', "uniformize", required_argument, "int",
     "select sources uniformly using roughly this many boxes (0=disable; default 10)"},
    {'\x81', "no-verify-uniformize", no_argument, NULL,
     "don't uniformize the field stars during verification"},
    {'\x83', "no-verify-dedup", no_argument, NULL,
     "don't deduplicate the field stars during verification"},
    {'C', "cancel",                required_argument, "filename",
     "filename whose creation signals the process to stop"},
    {'S', "solved",                required_argument, "filename",
     "output file to mark that the solver succeeded"},
    {'I', "solved-in",     required_argument, "filename",
     "input filename for solved file"},
    {'M', "match",                 required_argument, "filename",
     "output filename for match file"},
    {'R', "rdls",                  required_argument, "filename",
     "output filename for RDLS file"},
    {'\x80', "sort-rdls",    required_argument, "column",
     "sort the RDLS file by this column; default is ascending; use "
     "\"-column\" to sort \"column\" in descending order instead."},
    {'}',"tag",           required_argument, "column",
     "grab tag-along column from index into RDLS file"},
    {'<', "tag-all",       no_argument, NULL,
     "grab all tag-along columns from index into RDLS file"},
    {'j', "scamp-ref",     required_argument, "filename",
     "output filename for SCAMP reference catalog"},
    {'B', "corr",          required_argument, "filename",
     "output filename for correspondences"},
    {'W', "wcs",                   required_argument, "filename",
     "output filename for WCS file"},
    {'P', "pnm",                   required_argument, "filename",
     "save the PNM file as <filename>"},
    {'f', "force-ppm",     no_argument, NULL,
     "force the PNM file to be a PPM"},
    {'k', "keep-xylist",   required_argument, "filename",
     "save the (unaugmented) xylist to <filename>"},
    {'A', "dont-augment",   no_argument, NULL,
     "quit after writing the unaugmented xylist"},
    {'V', "verify",         required_argument, "filename",
     "try to verify an existing WCS file"},
    {'\x92', "verify-ext",     required_argument, "extension",
     "HDU from which to read WCS to verify; set this BEFORE --verify"},
    {'y', "no-verify",     no_argument, NULL,
     "ignore existing WCS headers in FITS input images"},
    {'g', "guess-scale",   no_argument, NULL,
     "try to guess the image scale from the FITS headers"},
    {'>', "crpix-center",  no_argument, NULL,
     "set the WCS reference point to the image center"},
    {'/', "crpix-x",  required_argument, "pix",
     "set the WCS reference point to the given position"},
    {'\\', "crpix-y",  required_argument, "pix",
     "set the WCS reference point to the given position"},
    {'T', "no-tweak",      no_argument, NULL,
     "don't fine-tune WCS by computing a SIP polynomial"},
    {'t', "tweak-order",    required_argument, "int",
     "polynomial order of SIP WCS corrections"},
    {'\x86', "predistort",  required_argument, "filename",
     "apply the inverse distortion in this SIP WCS header before solving"},
    {'\x94', "xscale", required_argument, "factor",
     "for rectangular pixels: factor to apply to measured X positions to make pixels square"},
    {'m', "temp-dir",       required_argument, "dir",
     "where to put temp files, default /tmp"},
    // placeholder for printing "The following are for xylist inputs only"
    {'\0', "xylist-only", no_argument, NULL, NULL},
    {'F', "fields",         required_argument, NULL, NULL},
    {'w', "width",                 required_argument, "pixels",
     "specify the field width"},
    {'e', "height",                required_argument, "pixels", 
     "specify the field height"},
    {'X', "x-column",       required_argument, "column-name",
     "the FITS column containing the X coordinate of the sources"},
    {'Y', "y-column",       required_argument, "column-name",
     "the FITS column containing the Y coordinate of the sources"},
    {'s', "sort-column",    required_argument, "column-name",
     "the FITS column that should be used to sort the sources"},
    {'a', "sort-ascending", no_argument, NULL,
     "sort in ascending order (smallest first); default is descending order"},
};

void augment_xylist_print_help(FILE* fid) {
    bl* opts;
    opts = opts_from_array(options, sizeof(options)/sizeof(an_option_t), NULL);
    opts_print_help(opts, fid, augment_xylist_print_special_opts, NULL);
    bl_free(opts);
}

void augment_xylist_add_options(bl* opts) {
    bl* myopts = opts_from_array(options, sizeof(options)/sizeof(an_option_t), NULL);
    bl_append_list(opts, myopts);
    bl_free(myopts);
}

static int parse_fields_string(il* fields, const char* str);

int augment_xylist_parse_option(char argchar, char* optarg,
                                augment_xylist_t* axy) {
    double d;
    int verify_extension = -1;
    //printf("parsing option %c (%i)\n", argchar, (int)argchar);
    switch (argchar) {
    case '\x80':
        axy->sort_rdls = optarg;
        break;
    case '\x81':
        axy->verify_uniformize = FALSE;
        break;
    case '\x82':
        axy->odds_to_tune_up = atof(optarg);
        break;
    case '\x83':
        axy->verify_dedup = FALSE;
        break;
    case '\x86':
        axy->predistort = sip_read_header_file(optarg, NULL);
        if (!axy->predistort) {
            ERROR("Failed to read SIP header file \"%s\" for pre-distortion values", optarg);
            return -1;
        }
        break;
    case '\x94':
        axy->pixel_xscale = atof(optarg);
        break;
    case ';':
        axy->invert_image = TRUE;
        break;
    case '>':
        axy->set_crpix_center = TRUE;
        axy->set_crpix = TRUE;
        break;
    case '/':
        axy->crpix[0] = atof(optarg);
        axy->set_crpix = TRUE;
        break;
    case '\\':
        axy->crpix[1] = atof(optarg);
        axy->set_crpix = TRUE;
        break;
    case '}':
        sl_append(axy->tagalong, optarg);
        break;
    case '<':
        axy->tagalong_all = TRUE;
        break;
    case '{':
        axy->image_sigma = atof(optarg);
        break;
    case '\x93':
        axy->image_nsigma = atof(optarg);
        break;
    case '^':
        axy->use_source_extractor = TRUE;
        break;
    case '&':
        axy->source_extractor_config = optarg;
        axy->use_source_extractor = TRUE;
        break;
    case '*':
        axy->source_extractor_path = optarg;
        axy->use_source_extractor = TRUE;
        break;
    case '9':
        axy->no_removelines = TRUE;
        break;
    case ':':
        axy->uniformize = atoi(optarg);
        break;
    case '3':
        axy->ra_center = atora(optarg);
        if (axy->ra_center == LARGE_VAL) {
            ERROR("Couldn't understand your RA center argument \"%s\"", optarg);
            return -1;
        }
        break;
    case '4':
        axy->dec_center = atodec(optarg);
        if (axy->dec_center == LARGE_VAL) {
            ERROR("Couldn't understand your Dec center argument \"%s\"", optarg);
            return -1;
        }
        break;
    case '5':
        axy->search_radius = atof(optarg);
        break;
    case '6':
        axy->extension = atoi(optarg);
        break;
    case '[':
        axy->odds_to_solve = atof(optarg);
        break;
    case '#':
        axy->odds_to_bail = atof(optarg);
        break;
    case '%':
        axy->odds_to_stoplooking = atof(optarg);
        break;
    case ']':
        axy->no_bg_subtraction = TRUE;
        break;
    case '8':
        if (streq(optarg, "pos")) {
            axy->parity = PARITY_NORMAL;
        } else if (streq(optarg, "neg")) {
            axy->parity = PARITY_FLIP;
        } else {
            ERROR("Couldn't understand your Parity argument \"%s\": must be \"pos\" or \"neg\"", optarg);
            return -1;
        }
        break;
    case 'B':
        axy->corrfn = optarg;
        break;
    case 'y':
        axy->try_verify = FALSE;
        break;
    case 'q':
        d = atof(optarg);
        if (d < 0.0 || d > 1.0) {
            ERROR("quad size fraction (-q / --quad-size-min) must be between 0.0 and 1.0");
            return -1;
        }
        axy->quadsize_min = d;
        break;
    case 'Q':
        d = atof(optarg);
        if (d < 0.0 || d > 1.0) {
            ERROR("quad size fraction (-Q / --quad-size-max) must be between 0.0 and 1.0");
            return -1;
        }
        axy->quadsize_max = d;
        break;
    case 'l':
        axy->cpulimit = atof(optarg);
        break;
    case 'A':
        axy->dont_augment = TRUE;
        break;
    case 'z':
        axy->downsample = atoi(optarg);
        break;
    case 'r':
        axy->resort = FALSE;
        break;
    case 'E':
        axy->pixelerr = atof(optarg);
        break;
    case 'c':
        axy->codetol = atof(optarg);
        break;
    case 'v':
        axy->verbosity++;
        break;
    case '7':
        axy->no_delete_temp = TRUE;
        break;
    case '\x92':
        verify_extension = atoi(optarg);
        break;
    case 'V':
        sl_append(axy->verifywcs, optarg);
        il_append(axy->verifywcs_ext, 
                  (verify_extension >= 0 ? verify_extension : axy->extension));
        break;
    case 'I':
        axy->solvedinfn = optarg;
        break;
    case 'k':
        axy->keepxylsfn = optarg;
        break;
    case 's':
        axy->sortcol = optarg;
        axy->resort = FALSE;
        break;
    case 'a':
        axy->sort_ascending = TRUE;
        break;
    case 'X':
        axy->xcol = optarg;
        break;
    case 'Y':
        axy->ycol = optarg;
        break;
    case 'm':
        axy->tempdir = optarg;
        break;
    case 'F':
        if (parse_fields_string(axy->fields, optarg)) {
            ERROR("Failed to parse fields specification \"%s\"", optarg);
            return -1;
        }
        break;
    case 'd':
        if (parse_depth_string(axy->depths, optarg)) {
            ERROR("Failed to parse depth specification: \"%s\"", optarg);
            return -1;
        }
        break;
    case '|':
        axy->cutobjs = atoi(optarg);
        break;
    case 'o':
        axy->axyfn = optarg;
        break;
    case 'i':
        axy->imagefn = optarg;
        break;
    case 'x':
        axy->xylsfn = optarg;
        break;
    case 'L':
        axy->scalelo = atof(optarg);
        break;
    case 'H':
        axy->scalehi = atof(optarg);
        break;
    case 'u':
        axy->scaleunit = parse_scale_units(optarg);
        if (axy->scaleunit == -1) {
            ERROR("Unknown image scale units \"%s\".  See \"solve-field -h\" for allowed values.\n", optarg);
            return -1;
        }
        break;
    case 'w':
        axy->W = atoi(optarg);
        break;
    case 'e':
        axy->H = atoi(optarg);
        break;
    case 'T':
        axy->tweak = FALSE;
        break;
    case 't':
        axy->tweakorder = atoi(optarg);
        break;
    case 'P':
        axy->pnmfn = optarg;
        break;
    case 'f':
        axy->force_ppm = TRUE;
        break;
    case 'g':
        axy->guess_scale = TRUE;
        break;
    case 'S':
        axy->solvedfn = optarg;
        break;
    case 'C':
        axy->cancelfn = optarg;
        break;
    case 'M':
        axy->matchfn = optarg;
        break;
    case 'R':
        axy->rdlsfn = optarg;
        break;
    case 'j':
        axy->scampfn = optarg;
        break;
    case 'W':
        axy->wcsfn = optarg;
        break;
    default:
        return 1;
    }

    return 0;
}

int parse_scale_units(const char* units) {
    if (strcaseeq(units, "degwidth") ||
        strcaseeq(units, "degw") ||
        strcaseeq(units, "dw")) {
        return SCALE_UNITS_DEG_WIDTH;
    } else if (strcaseeq(units, "arcminwidth") ||
               strcaseeq(units, "amw") ||
               strcaseeq(units, "aw")) {
        return SCALE_UNITS_ARCMIN_WIDTH;
    } else if (strcaseeq(units, "arcsecperpix") ||
               strcaseeq(units, "app")) {
        return SCALE_UNITS_ARCSEC_PER_PIX;
    } else if (strcaseeq(units, "focalmm")) {
        return SCALE_UNITS_FOCAL_MM;
    }
    return -1;
}

// run(): ppmtopgm, pnmtofits, source_extractor
// backtick(): pnmfile, image2pnm

static void append_escape(sl* list, const char* fn) {
    sl_append_nocopy(list, shell_escape(fn));
}
static void append_executable(sl* list, const char* fn, const char* me) {
    char* exec = find_executable(fn, me);
    if (!exec) {
        char* binfn = NULL;
        asprintf_safe(&binfn, "../bin/%s", fn);
        exec = find_executable(binfn, me);
        free(binfn);
    }
    if (!exec) {
        ERROR("Couldn't find executable \"%s\"", fn);
        exit(-1);
    }
    sl_append_nocopy(list, shell_escape(exec));
    free(exec);
}

static sl* backtick(sl* cmd, anbool verbose) {
    char* cmdstr = sl_implode(cmd, " ");
    sl* lines;
    logverb("Running: %s\n", cmdstr);
    if (run_command_get_outputs(cmdstr, &lines, NULL)) {
        ERROR("Failed to run command: %s", cmdstr);
        free(cmdstr);
        exit(-1);
    }
    free(cmdstr);
    sl_remove_all(cmd);
    return lines;
}

static void run(sl* cmd, anbool verbose) {
    if (verbose) {
        char* cmdstr = sl_implode(cmd, " ");
        logverb("Running: %s\n", cmdstr);
        if (run_command_get_outputs(cmdstr, NULL, NULL)) {
            ERROR("Failed to run command: %s", cmdstr);
            free(cmdstr);
            exit(-1);
        }
        free(cmdstr);
        sl_remove_all(cmd);
    } else {
        sl* lines = backtick(cmd, verbose);
        if (verbose) {
            int i;
            for (i=0; i<sl_size(lines); i++)
                logverb("  %s\n", sl_get(lines, i));
        }
        sl_free2(lines);
    }
}

static void add_sip_coeffs(qfits_header* hdr, const char* prefix, const sip_t* sip) {
    char key[64];
    int m, n, order;

    if (sip->a_order) {
        sprintf(key, "%sSAO", prefix);
        order = sip->a_order;
        fits_header_add_int(hdr, key, order, "SIP forward polynomial order");
        for (m=0; m<=order; m++) {
            for (n=0; (m+n)<=order; n++) {
                if (m+n < 1)
                    continue;
                sprintf(key, "%sA%i%i", prefix, m, n);
                fits_header_add_double(hdr, key, sip->a[m][n], "");
                sprintf(key, "%sB%i%i", prefix, m, n);
                fits_header_add_double(hdr, key, sip->b[m][n], "");
            }
        }
    }
    if (sip->ap_order) {
        order = sip->ap_order;
        sprintf(key, "%sSAPO", prefix);
        fits_header_add_int(hdr, key, order, "SIP reverse polynomial order");
        for (m=0; m<=order; m++) {
            for (n=0; (m+n)<=order; n++) {
                if (m+n < 1)
                    continue;
                sprintf(key, "%sAP%i%i", prefix, m, n);
                fits_header_add_double(hdr, key, sip->ap[m][n], "");
                sprintf(key, "%sBP%i%i", prefix, m, n);
                fits_header_add_double(hdr, key, sip->bp[m][n], "");
            }
        }
    }
}

int augment_xylist(augment_xylist_t* axy,
                   const char* me) {
    // tempfiles to delete when we finish
    sl* tempfiles;
    sl* cmd;
    anbool verbose = axy->verbosity > 0;
    int i, I;
    //anbool guessed_scale = FALSE;
    anbool dosort = FALSE;
    char* xylsfn;
    qfits_header* hdr = NULL;
    anbool addwh = TRUE;
    FILE* fout = NULL;
    char *fitsimgfn = NULL;
    dl* scales;
    char* nolinesfn = NULL;
    char* sortedxylsfn = NULL;
    char* unixylsfn = NULL;
    char* cutxylsfn = NULL;
    // as we process the image (uncompress it, eg), keep track of extension
    // (along with axy->fitsimgfn if keep_fitsimg is set)
    axy->fitsimgext = axy->extension;

    cmd = sl_new(16);
    tempfiles = sl_new(4);
    scales = dl_new(4);

    if (axy->imagefn) {
        // if --image is given:
        //       -run image2pnm
        //       -if it's a FITS image, keep the original
        //       -otherwise, run ppmtopgm (if necessary) and pnmtofits.
        //       -run image2xy to generate xylist
        char *uncompressedfn;
        char *pnmfn = NULL;
        sl* lines;
        anbool iscompressed = FALSE;
        char* line;
        char pnmtype;
        char typestr[256];
        anbool want_pnm = TRUE;

        uncompressedfn = create_temp_file("uncompressed", axy->tempdir);
        sl_append_nocopy(tempfiles, uncompressedfn);

        if (axy->assume_fits_image) {
            axy->isfits = TRUE;
            if (!axy->pnmfn) {
                qfits_header* hdr;
                want_pnm = FALSE;
                // We need to get image W,H from the FITS header.
                logverb("Reading FITS image \"%s\" to find image size\n", axy->imagefn);
                hdr = anqfits_get_header2(axy->imagefn, axy->extension);
                axy->W = qfits_header_getint(hdr, "NAXIS1", -1);
                axy->H = qfits_header_getint(hdr, "NAXIS2", -1);
                qfits_header_destroy(hdr);
                if (axy->W == -1 || axy->H == -1) {
                    ERROR("Failed to find size of FITS image \"%s\": got NAXIS1 = %i, NAXIS2 = %i\n",
                          axy->imagefn, axy->W, axy->H);
                    return -1;
                }
                logverb("  got FITS image size %i x %i\n", axy->W, axy->H);
            }
        }

        if (want_pnm) {
            if (axy->pnmfn)
                pnmfn = axy->pnmfn;
            else {
                pnmfn = create_temp_file("pnm", axy->tempdir);
                sl_append_nocopy(tempfiles, pnmfn);
            }

            append_executable(cmd, "image2pnm", me);
            if (axy->extension) {
                sl_append(cmd, "--extension");
                sl_appendf(cmd, "%i", axy->extension);
            }
            sl_append(cmd, "--infile");
            append_escape(cmd, axy->imagefn);
            sl_append(cmd, "--uncompressed-outfile");
            append_escape(cmd, uncompressedfn);
            sl_append(cmd, "--outfile");
            append_escape(cmd, pnmfn);
            if (axy->force_ppm)
                sl_append(cmd, "--ppm");
            sl_append(cmd, "--mydir");
            append_escape(cmd, me);
            lines = backtick(cmd, verbose);
            axy->isfits = FALSE;
            for (i=0; i<sl_size(lines); i++) {
                logverb("  %s\n", sl_get(lines, i));
                if (streq("fits", sl_get(lines, i)))
                    axy->isfits = TRUE;
                if (streq("compressed", sl_get(lines, i)))
                    iscompressed = TRUE;
                if (streq("fz", sl_get(lines, i)))
                    axy->fitsimgext = 0;
            }
            sl_free2(lines);

            // Get image W, H, depth.
            sl_append(cmd, "pnmfile");
            append_escape(cmd, pnmfn);

            lines = backtick(cmd, verbose);

            if (sl_size(lines) == 0) {
                ERROR("Got no output from the \"pnmfile\" program.");
                exit(-1);
            }
            line = sl_get(lines, 0);
            // eg       "/tmp/pnm:       PPM raw, 800 by 510  maxval 255"
            if (strlen(pnmfn) + 1 >= strlen(line)) {
                ERROR("Failed to parse output from pnmfile: \"%s\"", line);
                exit(-1);
            }
            line += strlen(pnmfn) + 1;
            // "PBM raw, 750 by 864"
            if (sscanf(line, " P%cM %255s %d by %d",
                       &pnmtype, typestr, &axy->W, &axy->H) != 4) {
                ERROR("Failed to parse output from pnmfile: \"%s\"\n", line);
                exit(-1);
            }
            sl_free2(lines);
        }

        if (axy->isfits) {

            if (iscompressed)
                fitsimgfn = uncompressedfn;
            else
                fitsimgfn = axy->imagefn;

            if (axy->try_verify) {
                char* errstr;
                sip_t sip;
                anbool ok;
                // Try to read WCS header from FITS image; if successful,
                // add it to the list of WCS headers to verify.
                logverb("Looking for a WCS header in FITS input image %s ext %i\n", fitsimgfn, axy->fitsimgext);

                // FIXME - Right now we just try to read SIP/TAN -
                // obviously this should be more flexible and robust.
                errors_start_logging_to_string();
                memset(&sip, 0, sizeof(sip_t));
                ok = (sip_read_header_file_ext(fitsimgfn, axy->fitsimgext, &sip) != NULL);
                errstr = errors_stop_logging_to_string(": ");
                if (ok) {
                    logmsg("Found an existing WCS header, will try to verify it.\n");
                    sl_append(axy->verifywcs, fitsimgfn);
                    il_append(axy->verifywcs_ext, axy->fitsimgext);
                } else {
                    logverb("Failed to read a SIP or TAN header from FITS image.\n");
                    logverb("  (reason: %s)\n", errstr);
                }
                free(errstr);
            }

        } else {
            fitsimgfn = create_temp_file("fits", axy->tempdir);
            sl_append_nocopy(tempfiles, fitsimgfn);
            
            if (pnmtype == 'P') {
                logverb("Converting PPM image to FITS...\n");

                sl_append(cmd, "ppmtopgm");
                append_escape(cmd, pnmfn);
                sl_append(cmd, "|");
                append_executable(cmd, "an-pnmtofits", me);
                sl_append(cmd, ">");
                append_escape(cmd, fitsimgfn);

                run(cmd, verbose);

            } else if (pnmtype == 'G') {
                logverb("Converting PGM image to FITS...\n");

                append_executable(cmd, "an-pnmtofits", me);
                append_escape(cmd, pnmfn);
                sl_append(cmd, ">");
                append_escape(cmd, fitsimgfn);

                run(cmd, verbose);

            } else if (pnmtype == 'B') {
                logverb("Converting PBM image to FITS...\n");

                sl_append(cmd, "pbmtopgm 3 3");
                append_escape(cmd, pnmfn);
                sl_append(cmd, "|");
                append_executable(cmd, "an-pnmtofits", me);
                sl_append(cmd, ">");
                append_escape(cmd, fitsimgfn);
                run(cmd, verbose);
            } else {
                assert(0);
            }

        }

        if (axy->keep_fitsimg) {
            axy->fitsimgfn = strdup(fitsimgfn);
            sl_remove_string(tempfiles, fitsimgfn);
        }

        logmsg("Extracting sources...\n");
        xylsfn = create_temp_file("xyls", axy->tempdir);
        sl_append_nocopy(tempfiles, xylsfn);

        if (axy->use_source_extractor) {
            if (axy->source_extractor_path)
                sl_append(cmd, axy->source_extractor_path);
            else
                sl_append(cmd, "source-extractor");

            if (axy->source_extractor_config)
                sl_appendf(cmd, "-c %s", axy->source_extractor_config);
            else {
                char* paramfn;
                char* paramstr;
                char* filterfn;
                char* filterstr;

                paramfn = create_temp_file("param", axy->tempdir);
                sl_append_nocopy(tempfiles, paramfn);
                paramstr = "X_IMAGE\nY_IMAGE\nMAG_AUTO\nFLUX_AUTO";
                if (write_file(paramfn, paramstr, strlen(paramstr))) {
                    ERROR("Failed to write Source Extractor parameters to temp file \"%s\"", paramfn);
                    exit(-1);
                }
                sl_appendf(cmd, "-PARAMETERS_NAME %s", paramfn);
                if (verbose)
                    sl_append(cmd, "-VERBOSE_TYPE FULL");

                axy->xcol = "X_IMAGE";
                axy->ycol = "Y_IMAGE";
                axy->sortcol = "MAG_AUTO";
                axy->sort_ascending = TRUE;

                filterfn = create_temp_file("filter", axy->tempdir);
                sl_append_nocopy(tempfiles, filterfn);
                filterstr = "CONV NORM\n"
                    "# 5x5 convolution mask of a gaussian PSF with FWHM = 2.0 pixels.\n"
                    "0.006319 0.040599 0.075183 0.040599 0.006319\n"
                    "0.040599 0.260856 0.483068 0.260856 0.040599\n"
                    "0.075183 0.483068 0.894573 0.483068 0.075183\n"
                    "0.040599 0.260856 0.483068 0.260856 0.040599\n"
                    "0.006319 0.040599 0.075183 0.040599 0.006319\n";
                if (write_file(filterfn, filterstr, strlen(filterstr))) {
                    ERROR("Failed to write Source Extractor convolution filter to temp file \"%s\"", filterfn);
                    exit(-1);
                }
                sl_appendf(cmd, "-FILTER_NAME %s", filterfn);
            }

            sl_append(cmd, "-CATALOG_TYPE FITS_1.0");
            sl_appendf(cmd, "-CATALOG_NAME %s", xylsfn);
            append_escape(cmd, fitsimgfn);

            logverb("Running Source Extractor: output file is %s\n", xylsfn);
            run(cmd, verbose);

        } else {
            simplexy_t sxyparams;
            logverb("Running image2xy: input=%s, output=%s, ext=%i\n", fitsimgfn, xylsfn, axy->fitsimgext);

            // we have to delete the temp file because otherwise image2xy is too timid to overwrite it.
            if (unlink(xylsfn)) {
                SYSERROR("Failed to delete temp file %s", xylsfn);
                exit(-1);
            }

            memset(&sxyparams, 0, sizeof(simplexy_t));
            // The other params get set to defaults for float or u8 images.
            sxyparams.nobgsub = axy->no_bg_subtraction;
            sxyparams.sigma = axy->image_sigma;
            sxyparams.invert = axy->invert_image;
            sxyparams.plim = axy->image_nsigma;

            // MAGIC 3: downsample by a factor of 2, up to 3 times.
            if (image2xy_files(fitsimgfn, xylsfn, TRUE, axy->downsample, 3,
                               axy->fitsimgext, 0, &sxyparams)) {
                ERROR("Source extraction failed");
                exit(-1);
            }
        }
        dosort = TRUE;

    } else {
        // xylist.
        xylsfn = axy->xylsfn;
        if (axy->sortcol)
            dosort = TRUE;

        if (axy->extension && (axy->extension != 1)) {
            // Copy just this extension to a temp file.
            FILE* fout;
            FILE* fin;
            off_t offset;
            off_t nbytes;
            off_t nprimary;
            anqfits_t* anq;
            char* extfn;

            anq = anqfits_open_hdu(xylsfn, axy->extension);
            if (!anq) {
                ERROR("Failed to open xyls file %s up to extension %i",
                      xylsfn, axy->extension);
                exit(-1);
            }

            fin = fopen(xylsfn, "rb");
            if (!fin) {
                SYSERROR("Failed to open xyls file \"%s\"", xylsfn);
                exit(-1);
            }

            nprimary = anqfits_header_size(anq, 0) + anqfits_data_size(anq, 0);
            offset = anqfits_header_start(anq, axy->extension);
            nbytes = anqfits_header_size(anq, axy->extension) +
                anqfits_data_size(anq, axy->extension);
            anqfits_close(anq);
            extfn = create_temp_file("ext", axy->tempdir);
            sl_append_nocopy(tempfiles, extfn);
            fout = fopen(extfn, "wb");
            if (!fout) {
                SYSERROR("Failed to open temp file \"%s\" to write extension",
                         extfn);
                exit(-1);
            }
            logverb("Copying ext %i of %s to temp %s\n", axy->extension,
                    xylsfn, extfn);
            if (pipe_file_offset(fin, 0, nprimary, fout)) {
                ERROR("Failed to copy the primary HDU of xylist file %s to %s",
                      xylsfn, extfn);
                exit(-1);
            }
            if (pipe_file_offset(fin, offset, nbytes, fout)) {
                ERROR("Failed to copy HDU %i of xylist file %s to %s",
                      axy->extension, xylsfn, extfn);
                exit(-1);
            }
            fclose(fin);
            if (fclose(fout)) {
                SYSERROR("Failed to close %s", extfn);
                exit(-1);
            }
            xylsfn = extfn;
        }
    }

    if (axy->guess_scale && (fitsimgfn || !axy->imagefn)) {
        dl* estscales = NULL;
        char* infn = (fitsimgfn ? fitsimgfn : xylsfn);
        fits_guess_scale(infn, NULL, &estscales);
        for (i=0; i<dl_size(estscales); i++) {
            double scale = dl_get(estscales, i);
            logverb("Scale estimate: %g\n", scale);
            dl_append(scales, scale * 0.99);
            dl_append(scales, scale * 1.01);
            //guessed_scale = TRUE;
        }
        dl_free(estscales);
    }

    // remove lines
    // sort
    // uniformize
    // cut
    if (axy->keepxylsfn) {
        // Figure out which is the last stage to run, and set its output
        // file to "keepxylsfn".
        if (axy->cutobjs) {
            cutxylsfn = axy->keepxylsfn;
        } else if (axy->uniformize) {
            unixylsfn = axy->keepxylsfn;
        } else if (dosort) {
            sortedxylsfn = axy->keepxylsfn;
        } else if (!axy->no_removelines) {
            nolinesfn = axy->keepxylsfn;
        } else {
            // copy xylsfn to axy->keepxylsfn.
            if (copy_file(xylsfn, axy->keepxylsfn)) {
                ERROR("Failed to copy xyls file \"%s\" to \"%s\"",
                      xylsfn, axy->keepxylsfn);
                return -1;
            }
        }
    }

    if (!axy->no_removelines) {
        if (!nolinesfn) {
            nolinesfn = create_temp_file("removelines", axy->tempdir);
            sl_append_nocopy(tempfiles, nolinesfn);
        }
        logverb("Removing lines of (spurious) sources from xylist \"%s\", writing to \"%s\"\n",
                xylsfn, nolinesfn);
        append_executable(cmd, "removelines", me);
        if (axy->xcol)
            sl_appendf(cmd, "-X %s", axy->xcol);
        if (axy->ycol)
            sl_appendf(cmd, "-Y %s", axy->ycol);
        //if (axy->extension)
        //    sl_appendf(cmd, "-e %i", axy->extension);
        append_escape(cmd, xylsfn);
        append_escape(cmd, nolinesfn);
        run(cmd, verbose);
        xylsfn = nolinesfn;
    }

    if (dosort) {
        anbool do_tabsort = FALSE;

        if (!axy->sortcol)
            axy->sortcol = "FLUX";
        if (!axy->bgcol)
            axy->bgcol = "BACKGROUND";

        if (!sortedxylsfn) {
            sortedxylsfn = create_temp_file("sorted", axy->tempdir);
            sl_append_nocopy(tempfiles, sortedxylsfn);
        }

        if (axy->resort) {
            char* err;
            int rtn;
            logverb("Sorting file \"%s\" to \"%s\" using columns flux (%s) and background (%s), %sscending\n",
                    xylsfn, sortedxylsfn, axy->sortcol, axy->bgcol, axy->sort_ascending?"a":"de");
            errors_start_logging_to_string();
            rtn = resort_xylist(xylsfn, sortedxylsfn, axy->sortcol, axy->bgcol, axy->sort_ascending);
            err = errors_stop_logging_to_string(": ");
            if (rtn) {
                logmsg("Sorting brightness using %s and BACKGROUND columns failed; falling back to %s.\n",
                       axy->sortcol, axy->sortcol);
                logverb("Reason: %s\n", err);
                do_tabsort = TRUE;
            }
            free(err);

        } else
            do_tabsort = TRUE;

        if (do_tabsort) {
            logverb("Sorting by brightness: input=%s, output=%s, column=%s.\n",
                    xylsfn, sortedxylsfn, axy->sortcol);
            tabsort(xylsfn, sortedxylsfn,
                    axy->sortcol, !axy->sort_ascending);
        }
        xylsfn = sortedxylsfn;
    }

    if (axy->uniformize) {
        if (!unixylsfn) {
            unixylsfn = create_temp_file("uniform", axy->tempdir);
            sl_append_nocopy(tempfiles, unixylsfn);
        }
        append_executable(cmd, "uniformize", me);
        sl_appendf(cmd, "-n %i", axy->uniformize);
        if (axy->xcol)
            sl_appendf(cmd, "-X %s", axy->xcol);
        if (axy->ycol)
            sl_appendf(cmd, "-Y %s", axy->ycol);
        //if (axy->extension)
        //    sl_appendf(cmd, "-e %i", axy->extension);
        append_escape(cmd, xylsfn);
        append_escape(cmd, unixylsfn);
        run(cmd, verbose);
        xylsfn = unixylsfn;
    }

    if (axy->cutobjs) {
        // cut the source lists to at most "cutobjs" objects.
        if (!cutxylsfn) {
            cutxylsfn = create_temp_file("cut", axy->tempdir);
            sl_append_nocopy(tempfiles, cutxylsfn);
        }

        if (cut_table(xylsfn, cutxylsfn, axy->cutobjs)) {
            ERROR("Failed to cut table %s to %i entries; output file %s", xylsfn, axy->cutobjs, cutxylsfn);
            return -1;
        }
        xylsfn = cutxylsfn;
    }

    if (axy->dont_augment)
        // done!
        goto cleanup;

    // start piling FITS headers in there.
    hdr = anqfits_get_header2(xylsfn, 0);
    if (!hdr) {
        ERROR("Failed to read FITS header from file %s", xylsfn);
        exit(-1);
    }

    // delete any existing processing directives
    delete_existing_an_headers(hdr);

    if (!(axy->W && axy->H)) {
        // Look for existing IMAGEW and IMAGEH in primary header.
        axy->W = qfits_header_getint(hdr, "IMAGEW", 0);
        axy->H = qfits_header_getint(hdr, "IMAGEH", 0);
        if (axy->W && axy->H) {
            addwh = FALSE;
        } else {
            // Look for IMAGEW and IMAGEH headers in first extension, else bail.
            qfits_header* hdr2 = anqfits_get_header2(xylsfn, 1);
            axy->W = qfits_header_getint(hdr2, "IMAGEW", 0);
            axy->H = qfits_header_getint(hdr2, "IMAGEH", 0);
            qfits_header_destroy(hdr2);
        }
        if (!(axy->W && axy->H)) {
            ERROR("Error: image width and height must be specified for XYLS inputs");
            exit(-1);
        }
    }

    // we may write long filenames.
    fits_header_add_longstring_boilerplate(hdr);

    if (addwh) {
        fits_header_add_int(hdr, "IMAGEW", axy->W, "image width");
        fits_header_add_int(hdr, "IMAGEH", axy->H, "image height");
    }
    qfits_header_add(hdr, "ANRUN", "T", "Solve this field!", NULL);

    if (axy->cpulimit > 0)
        fits_header_add_double(hdr, "ANCLIM", axy->cpulimit, "CPU time limit (seconds)");

    if (axy->xcol)
        qfits_header_add(hdr, "ANXCOL", axy->xcol, "Name of column containing X coords", NULL);
    if (axy->ycol)
        qfits_header_add(hdr, "ANYCOL", axy->ycol, "Name of column containing Y coords", NULL);

    if (axy->tagalong_all)
        qfits_header_add(hdr, "ANTAGALL", "T", "Tag-along all columns from index to RDLS", NULL);
    else
        for (i=0; i<sl_size(axy->tagalong); i++) {
            char key[64];
            sprintf(key, "ANTAG%i", i+1);
            qfits_header_add(hdr, key, sl_get(axy->tagalong, i), "Tag-along column from index to RDLS", NULL);
        }

    if (axy->sort_rdls)
        qfits_header_add(hdr, "ANRDSORT", axy->sort_rdls, "Sort RDLS file by this column", NULL);

    qfits_header_add(hdr, "ANVERUNI", axy->verify_uniformize ? "T":"F", "Uniformize field during verification", NULL);
    qfits_header_add(hdr, "ANVERDUP", axy->verify_dedup ? "T":"F", "Deduplicate field during verification", NULL);

    if (axy->odds_to_tune_up)
        fits_header_add_double(hdr, "ANODDSTU", axy->odds_to_tune_up, "Odds ratio to tune up a match");
    if (axy->odds_to_solve)
        fits_header_add_double(hdr, "ANODDSSL", axy->odds_to_solve, "Odds ratio to consider a field solved");
    if (axy->odds_to_bail)
        fits_header_add_double(hdr, "ANODDSBL", axy->odds_to_bail, "Odds ratio to consider a hypothesis rejected");
    if (axy->odds_to_stoplooking)
        fits_header_add_double(hdr, "ANODDSST", axy->odds_to_stoplooking, "Odds ratio to stop trying to improve the odds ratio");

    if ((axy->scalelo > 0.0) || (axy->scalehi > 0.0)) {
        double appu, appl;
        switch (axy->scaleunit) {
        case SCALE_UNITS_DEG_WIDTH:
            logverb("Scale range: %g to %g degrees wide\n", axy->scalelo, axy->scalehi);
            appl = deg2arcsec(axy->scalelo) / (double)axy->W;
            appu = deg2arcsec(axy->scalehi) / (double)axy->W;
            logverb("Image width %i pixels; arcsec per pixel range %g %g\n", axy->W, appl, appu);
            break;
        case SCALE_UNITS_ARCMIN_WIDTH:
            logverb("Scale range: %g to %g arcmin wide\n", axy->scalelo, axy->scalehi);
            appl = arcmin2arcsec(axy->scalelo) / (double)axy->W;
            appu = arcmin2arcsec(axy->scalehi) / (double)axy->W;
            logverb("Image width %i pixels; arcsec per pixel range %g %g\n", axy->W, appl, appu);
            break;
        case SCALE_UNITS_ARCSEC_PER_PIX:
            logverb("Scale range: %g to %g arcsec/pixel\n", axy->scalelo, axy->scalehi);
            appl = axy->scalelo;
            appu = axy->scalehi;
            break;
        case SCALE_UNITS_FOCAL_MM:
            logverb("Scale range: %g to %g mm focal length\n", axy->scalelo, axy->scalehi);
            // "35 mm" film is 36 mm wide.
            appu = rad2arcsec(atan(36. / (2. * axy->scalelo))) / (double)axy->W;
            appl = rad2arcsec(atan(36. / (2. * axy->scalehi))) / (double)axy->W;
            logverb("Image width %i pixels; arcsec per pixel range %g %g\n", axy->W, appl, appu);
            break;
        default:
            ERROR("Unknown scale unit code %i\n", axy->scaleunit);
            return -1;
        }
        dl_append(scales, appl);
        dl_append(scales, appu);
    }

    /* Hmm, do we want this??
     if ((dl_size(axy->scales) > 0) && guessed_scale)
     qfits_header_add(hdr, "ANAPPDEF", "T", "try the default scale range too.", NULL);
     */

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

    if (axy->quadsize_min > 0.0)
        fits_header_add_double(hdr, "ANQSFMIN", axy->quadsize_min, "minimum quad size: fraction");
    if (axy->quadsize_max > 0.0)
        fits_header_add_double(hdr, "ANQSFMAX", axy->quadsize_max, "maximum quad size: fraction");

    if (axy->set_crpix) {
        if (axy->set_crpix_center) {
            qfits_header_add(hdr, "ANCRPIXC", "T", "Set CRPIX to the image center.", NULL);
        } else {
            fits_header_add_double(hdr, "ANCRPIX1", axy->crpix[0], "Set CRPIX1 to this val.");
            fits_header_add_double(hdr, "ANCRPIX2", axy->crpix[1], "Set CRPIX2 to this val.");
        }
    }

    qfits_header_add(hdr, "ANTWEAK", (axy->tweak ? "T" : "F"), (axy->tweak ? "Tweak: yes please!" : "Tweak: no, thanks."), NULL);
    if (axy->tweak && axy->tweakorder)
        fits_header_add_int(hdr, "ANTWEAKO", axy->tweakorder, "Tweak order");

    if (axy->solvedfn)
        fits_header_addf_longstring(hdr, "ANSOLVED", "solved output file", "%s", axy->solvedfn);
    if (axy->solvedinfn)
        fits_header_addf_longstring(hdr, "ANSOLVIN", "solved input file", "%s", axy->solvedinfn);
    if (axy->cancelfn)
        fits_header_addf_longstring(hdr, "ANCANCEL", "cancel output file", "%s", axy->cancelfn);
    if (axy->matchfn)
        fits_header_addf_longstring(hdr, "ANMATCH", "match output file", "%s", axy->matchfn);
    if (axy->rdlsfn)
        fits_header_addf_longstring(hdr, "ANRDLS", "ra-dec output file", "%s", axy->rdlsfn);
    if (axy->scampfn)
        fits_header_addf_longstring(hdr, "ANSCAMP", "SCAMP reference catalog output file", "%s", axy->scampfn);
    if (axy->wcsfn)
        fits_header_addf_longstring(hdr, "ANWCS", "WCS header output filename", "%s", axy->wcsfn);
    if (axy->corrfn)
        fits_header_addf_longstring(hdr, "ANCORR", "Correspondences output filename", "%s", axy->corrfn);
    if (axy->codetol > 0.0)
        fits_header_add_double(hdr, "ANCTOL", axy->codetol, "code tolerance");
    if (axy->pixelerr > 0.0)
        fits_header_add_double(hdr, "ANPOSERR", axy->pixelerr, "star pos'n error (pixels)");

    if (axy->parity != PARITY_BOTH) {
        if (axy->parity == PARITY_NORMAL)
            qfits_header_add(hdr, "ANPARITY", "POS", "det(CD) > 0", NULL);
        else if (axy->parity == PARITY_FLIP)
            qfits_header_add(hdr, "ANPARITY", "NEG", "det(CD) < 0", NULL);
    }

    if ((axy->ra_center != LARGE_VAL) &&
        (axy->dec_center != LARGE_VAL) &&
        (axy->search_radius >= 0.0)) {
        fits_header_add_double(hdr, "ANERA", axy->ra_center, "RA center estimate (deg)");
        fits_header_add_double(hdr, "ANEDEC", axy->dec_center, "Dec center estimate (deg)");
        fits_header_add_double(hdr, "ANERAD", axy->search_radius, "Search radius from estimated posn (deg)");
    }

    for (i=0; i<il_size(axy->depths)/2; i++) {
        int depthlo, depthhi;
        char key[64];
        depthlo = il_get(axy->depths, 2*i);
        depthhi = il_get(axy->depths, 2*i + 1);
        sprintf(key, "ANDPL%i", (i+1));
        fits_header_addf(hdr, key, "", "%i", depthlo);
        sprintf(key, "ANDPU%i", (i+1));
        fits_header_addf(hdr, key, "", "%i", depthhi);
    }

    for (i=0; i<il_size(axy->fields)/2; i++) {
        int lo = il_get(axy->fields, 2*i);
        int hi = il_get(axy->fields, 2*i + 1);
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
    for (i=0; i<sl_size(axy->verifywcs); i++) {
        sip_t sip;
        const char* fn;
        int ext;

        fn = sl_get(axy->verifywcs, i);
        ext = il_get(axy->verifywcs_ext, i);
        if (!sip_read_header_file_ext(fn, ext, &sip)) {
            ERROR("Failed to parse WCS header from file \"%s\" ext %i", fn, ext);
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
            int j;
            for (j = 0; j < 8; j++) {
                sprintf(key, keys[j], I);
                fits_header_add_double(hdr, key, vals[j], "");
            }

            sprintf(key, "ANW%i", I);
            add_sip_coeffs(hdr, key, &sip);
        }
    }

    if (axy->predistort) {
        fits_header_add_double(hdr, "ANDPIX0", axy->predistort->wcstan.crpix[0], "Pre-distortion ref pix x");
        fits_header_add_double(hdr, "ANDPIX1", axy->predistort->wcstan.crpix[1], "Pre-distortion ref pix y");
        add_sip_coeffs(hdr, "AND", axy->predistort);
    }

    if (axy->pixel_xscale) {
        fits_header_add_double(hdr, "ANPXSCAL", axy->pixel_xscale, "x scaling to make square pixels");
    }

    fout = fopen(axy->axyfn, "wb");
    if (!fout) {
        SYSERROR("Failed to open output file %s", axy->axyfn);
        exit(-1);
    }

    logverb("Writing headers to file %s\n", axy->axyfn);

    if (qfits_header_dump(hdr, fout)) {
        ERROR("Failed to write FITS header");
        exit(-1);
    }
    qfits_header_destroy(hdr);

    // copy blocks from xyls to output.
    {
        FILE* fin;
        off_t offset;
        off_t nbytes;
        anqfits_t* anq;
        int ext = 1;

        anq = anqfits_open_hdu(xylsfn, ext);
        if (!anq) {
            ERROR("Failed to open xyls file %s up to extension %i", xylsfn, ext);
            exit(-1);
        }
        offset = anqfits_header_start(anq, ext);
        nbytes = anqfits_header_size(anq, ext) + anqfits_data_size(anq, ext);

        logverb("Copying data block of file %s to output %s.\n",
                xylsfn, axy->axyfn);
        anqfits_close(anq);

        fin = fopen(xylsfn, "rb");
        if (!fin) {
            SYSERROR("Failed to open xyls file \"%s\"", xylsfn);
            exit(-1);
        }

        if (pipe_file_offset(fin, offset, nbytes, fout)) {
            ERROR("Failed to copy the data segment of xylist file %s to %s",
                  xylsfn, axy->axyfn);
            exit(-1);
        }
        fclose(fin);
    }
    fclose(fout);

 cleanup:
    if (!axy->no_delete_temp) {
        for (i=0; i<sl_size(tempfiles); i++) {
            char* fn = sl_get(tempfiles, i);
            logverb("Deleting temp file %s\n", fn);
            if (unlink(fn)) {
                SYSERROR("Failed to delete temp file \"%s\"", fn);
            }
        }
    }

    dl_free(scales);
    sl_free2(cmd);
    sl_free2(tempfiles);
    return 0;
}

static void delete_existing_an_headers(qfits_header* hdr) {
    int i,j,k;
    char key[64];
    qfits_header_del(hdr, "ANRUN");
    qfits_header_del(hdr, "ANCLIM");
    qfits_header_del(hdr, "ANXCOL");
    qfits_header_del(hdr, "ANYCOL");
    qfits_header_del(hdr, "ANTAGALL");
    for (i=0; i<100; i++) {
        sprintf(key, "ANTAG%i", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANAPPL%i", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANAPPU%i", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANDPL%i", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANDPU%i", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANFD%i", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANFDL%i", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANFDU%i", i+1);
        qfits_header_del(hdr, key);
    }
    for (i=0; i<10; i++) {
        sprintf(key, "ANW%iPIX1", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iPIX2", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iVAL1", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iVAL2", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iCD11", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iCD12", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iCD21", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iCD22", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iSAO", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iSAPO", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iSBO", i+1);
        qfits_header_del(hdr, key);
        sprintf(key, "ANW%iSBPO", i+1);
        qfits_header_del(hdr, key);
        for (j=0; j<10; j++) {
            for (k=0; k<10; k++) {
                sprintf(key, "ANW%iA%i%i", i+1, j, k);
                qfits_header_del(hdr, key);
                sprintf(key, "ANW%iB%i%i", i+1, j, k);
                qfits_header_del(hdr, key);
                sprintf(key, "ANW%iAP%i%i", i+1, j, k);
                qfits_header_del(hdr, key);
                sprintf(key, "ANW%iBP%i%i", i+1, j, k);
                qfits_header_del(hdr, key);
            }
        }
    }
    qfits_header_del(hdr, "ANRDSORT");
    qfits_header_del(hdr, "ANVERUNI");
    qfits_header_del(hdr, "ANVERDUP");
    qfits_header_del(hdr, "ANODDSTU");
    qfits_header_del(hdr, "ANODDSSL");
    qfits_header_del(hdr, "ANODDSBL");
    qfits_header_del(hdr, "ANODDSST");
    qfits_header_del(hdr, "ANQSFMIN");
    qfits_header_del(hdr, "ANQSFMAX");
    qfits_header_del(hdr, "ANCRPIXC");
    qfits_header_del(hdr, "ANCRPIX1");
    qfits_header_del(hdr, "ANCRPIX2");
    qfits_header_del(hdr, "ANTWEAK");
    qfits_header_del(hdr, "ANTWEAKO");
    qfits_header_del(hdr, "ANSOLVED");
    qfits_header_del(hdr, "ANSOLVIN");
    qfits_header_del(hdr, "ANCANCEL");
    qfits_header_del(hdr, "ANMATCH");
    qfits_header_del(hdr, "ANRDLS");
    qfits_header_del(hdr, "ANSCAMP");
    qfits_header_del(hdr, "ANWCS");
    qfits_header_del(hdr, "ANCORR");
    qfits_header_del(hdr, "ANCTOL");
    qfits_header_del(hdr, "ANPOSERR");
    qfits_header_del(hdr, "ANPARITY");
    qfits_header_del(hdr, "ANERA");
    qfits_header_del(hdr, "ANEDEC");
    qfits_header_del(hdr, "ANERAD");
    qfits_header_del(hdr, "ANDPIX0");
    qfits_header_del(hdr, "ANDPIX1");
    qfits_header_del(hdr, "ANDSAO");
    qfits_header_del(hdr, "ANDSBO");
    qfits_header_del(hdr, "ANDSAPO");
    qfits_header_del(hdr, "ANDSBPO");
    for (j=0; j<10; j++) {
        for (k=0; k<10; k++) {
            sprintf(key, "ANDA%i%i", j, k);
            qfits_header_del(hdr, key);
            sprintf(key, "ANDB%i%i", j, k);
            qfits_header_del(hdr, key);
            sprintf(key, "ANDAP%i%i", j, k);
            qfits_header_del(hdr, key);
            sprintf(key, "ANDBP%i%i", j, k);
            qfits_header_del(hdr, key);
        }
    }
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
            ERROR("Failed to parse fragment: \"%s\"", str);
            return -1;
        }
        if (lo <= 0) {
            ERROR("Field number %i is invalid: must be >= 1.", lo);
            return -1;
        }
        if (lo > hi) {
            ERROR("Field range %i to %i is invalid: max must be >= min!", lo, hi);
            return -1;
        }
        il_append(fields, lo);
        il_append(fields, hi);
        str += nread;
        while ((*str == ',') || isspace((unsigned)(*str)))
            str++;
    }
    return 0;
}

