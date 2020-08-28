/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "build-index.h"
#include "wcs-xy2rd.h"
#include "rdlist.h"
#include "sip.h"
#include "sip_qfits.h"
#include "healpix.h"

const char* OPTIONS = "hvx:w:l:u:o:d:I:N:p:R:L:Mn:U:S:f:r:J:X:Y:";

static void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "      -x <input-xylist>    input: source positions; assumed sorted by brightness\n"
           "      -w <input-wcs>       input: WCS file for sources\n"
           "      -o <output-index>    output filename for index\n"
           "      [-X <x-column-name>]\n"
           "      [-Y <y-column-name>]\n"
           "      [-l <min-quad-size>]: minimum fraction of the image size (diagonal) to make quads (default 0.05)\n"
           "      [-u <max-quad-size>]: maximum fraction of the image size (diagonal) to make quads (default 1.0)\n"
           "      [-d <dimquads>] number of stars in a \"quad\" (default 4).\n"
           "      [-N <number of healpixels]: number of healpix grid cells to put in the image (default 10)\n"
           "      [-I <unique-id>] set the unique ID of this index\n"
           "      [-S]: sort column (default: assume already sorted)\n"
           "      [-f]: sort in descending order (eg, for FLUX); default ascending (eg, for MAG)\n"
           "      [-U]: healpix Nside for uniformization (default: same as -N)\n"
           "      [-n <sweeps>]    (ie, number of stars per fine healpix grid cell); default 10\n"
           "      [-r <dedup-radius>]: deduplication radius in arcseconds; default no deduplication\n"
           "      [-p <passes>]   number of rounds of quad-building (ie, # quads per healpix cell, default 16)\n"
           "      [-R <reuse-times>] number of times a star can be used (default 8).\n"
           "      [-L <max-reuses>] make extra passes through the healpixes, increasing the \"-r\" reuse (default 20)\n"
           "                     limit each time, up to \"max-reuses\".\n"
           "      [-M]: in-memory (don't use temp files)\n"
           "      [-J <jitter-in-pixels>]: set positional error of index stars, in pixels (default 1)\n"
           "\n"
           "      [-v]: add verbosity.\n"
           "\n", progname);
}


int main(int argc, char** argv) {
    int argchar;

    char* xylsfn = NULL;
    char* wcsfn = NULL;
    char* indexfn = NULL;

    double lowf = 0.1;
    double highf = 1.0;

    int loglvl = LOG_MSG;
    int i;
    int nhp = 10;
    int Nside;

    sl* tempfiles;
    char* tempdir = "/tmp";
    int wcsext = 0;
    char* rdlsfn;

    sip_t sip;
    double diagpix, diag, pscale;

    index_params_t myip;
    index_params_t* ip = &myip;
    double jitterpix = 1.0;
    char* xcol = NULL;
    char* ycol = NULL;

    build_index_defaults(ip);

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'X':
            xcol = optarg;
            break;
        case 'Y':
            ycol = optarg;
            break;
        case 'J':
            jitterpix = atof(optarg);
            break;
        case 'L':
            ip->Nloosen = atoi(optarg);
            break;
        case 'R':
            ip->Nreuse = atoi(optarg);
            break;
        case 'p':
            ip->passes = atoi(optarg);
            break;
        case 'M':
            ip->inmemory = TRUE;
            break;
        case 'U':
            ip->UNside = atoi(optarg);
            break;
        case 'S':
            ip->sortcol = optarg;
            break;
        case 'f':
            ip->sortasc = FALSE;
            break;
        case 'n':
            ip->sweeps = atoi(optarg);
            break;
        case 'r':
            ip->dedup = atof(optarg);
            break;
        case 'N':
            nhp = atoi(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case 'd':
            ip->dimquads = atoi(optarg);
            break;
        case 'I':
            ip->indexid = atoi(optarg);
            break;
        case 'h':
            print_help(argv[0]);
            exit(0);
        case 'x':
            xylsfn = optarg;
            break;
        case 'w':
            wcsfn = optarg;
            break;
        case 'o':
            indexfn = optarg;
            break;
        case 'u':
            highf = atof(optarg);
            break;
        case 'l':
            lowf = atof(optarg);
            break;
        default:
            return -1;
        }
	
    log_init(loglvl);

    if (!xylsfn || !wcsfn || !indexfn) {
        printf("Specify in & out filenames, bonehead!\n");
        print_help(argv[0]);
        exit( -1);
    }

    if (optind != argc) {
        print_help(argv[0]);
        printf("\nExtra command-line args were given: ");
        for (i=optind; i<argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        exit(-1);
    }

    tempfiles = sl_new(4);

    // wcs-xy2rd
    rdlsfn = create_temp_file("rdls", tempdir);
    sl_append_nocopy(tempfiles, rdlsfn);
    logmsg("Writing RA,Decs to %s\n", rdlsfn);
    if (wcs_xy2rd(wcsfn, wcsext, xylsfn, rdlsfn, xcol, ycol, FALSE, FALSE, NULL)) {
        ERROR("Failed to convert xylist to rdlist");
        exit(-1);
    }

    // compute quad size range.
    // read WCS.
    if (!sip_read_tan_or_sip_header_file_ext(wcsfn, wcsext, &sip, FALSE)) {
        ERROR("Failed to read WCS file %s", wcsfn);
        exit(-1);
    }
    // in pixels
    diagpix = hypot(sip.wcstan.imagew, sip.wcstan.imageh);
    // arcsec / pix
    pscale = sip_pixel_scale(&sip);
    // in arcsec
    diag = diagpix * pscale;
    // in arcmin
    ip->qlo = arcsec2arcmin(lowf * diag);
    ip->qhi = arcsec2arcmin(highf * diag);

    logmsg("Image is %i x %i pixels\n", (int)sip.wcstan.imagew, (int)sip.wcstan.imageh);
    logmsg("Setting quad scale range to [%g, %g] pixels, [%g, %g] arcsec ([%g, %g] arcmin)\n",
           diagpix * lowf, diagpix * highf, diag * lowf, diag * highf,
           ip->qlo, ip->qhi);

    ip->jitter = pscale * jitterpix;

    // number of healpixes:
    logmsg("Image area: %g arcsec^2\n", sip.wcstan.imagew * sip.wcstan.imageh * pscale*pscale);
    logmsg(" %g arcsec\n", sqrt(sip.wcstan.imagew * sip.wcstan.imageh * pscale*pscale));
    logmsg(" %g arcmin\n", arcsec2arcmin(sqrt(sip.wcstan.imagew * sip.wcstan.imageh * pscale*pscale)));

    logmsg("Desired hp area: %g arcsec^2\n", sip.wcstan.imagew * sip.wcstan.imageh * pscale*pscale / (double)nhp);
    logmsg(" %g arcsec\n", sqrt(sip.wcstan.imagew * sip.wcstan.imageh * pscale*pscale / (double)nhp));
    logmsg(" %g arcmin\n", arcsec2arcmin(sqrt(sip.wcstan.imagew * sip.wcstan.imageh * pscale*pscale / (double)nhp)));
    Nside = (int)ceil(healpix_nside_for_side_length_arcmin(arcsec2arcmin(sqrt(sip.wcstan.imagew * sip.wcstan.imageh * pscale*pscale / (double)nhp))));
    logverb("Chose healpix Nside=%i, side length %g arcmin\n", Nside, healpix_side_length_arcmin(Nside));
    ip->Nside = Nside;
    ip->scanoccupied = TRUE;
	
    if (build_index_files(rdlsfn, indexfn, ip)) {
        exit(-1);
    }
        
    // allquads
    /*
     {
     allquads_t* aq;
     qfits_header* hdr;

     aq = allquads_init();
     aq->dimquads = dimquads;
     aq->dimcodes = dimquad2dimcode(aq->dimquads);
     aq->id = id;
     aq->quadfn = quadfn;
     aq->codefn = codefn;
     aq->skdtfn = skdtfn;
     aq->quad_d2_lower = arcsec2distsq(diag * lowf);
     aq->quad_d2_upper = arcsec2distsq(diag * highf);
     aq->use_d2_lower = TRUE;
     aq->use_d2_upper = TRUE;

     if (allquads_open_outputs(aq)) {
     exit(-1);
     }
     hdr = codefile_get_header(aq->codes);
     qfits_header_add(hdr, "CIRCLE", "T", "Codes live in the circle", NULL);
     if (allquads_create_quads(aq) ||
     allquads_close(aq)) {
     exit(-1);
     }
     allquads_free(aq);
     }
     */

    printf("Done.\n");
    return 0;
}

