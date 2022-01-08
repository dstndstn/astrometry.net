/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>

#include "os-features.h"
#include "bl.h"
#include "fit-wcs.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "scamp.h"
#include "log.h"
#include "errors.h"
#include "matchfile.h"
#include "matchobj.h"
#include "boilerplate.h"
#include "xylist.h"
#include "rdlist.h"
#include "mathutil.h"
#include "verify.h"
#include "plotstuff.h"
#include "plotimage.h"
#include "cairoutils.h"
#include "fitsioutils.h"
#include "tweak2.h"
#include "mathutil.h"

static const char* OPTIONS = "hx:m:r:vj:p:i:J:o:W:w:s:X:Y:";

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "   -m <match input file>\n"
           "   -x <xyls input file>\n"
           "   -r <rdls input file>\n"
           "   [-W <initial WCS>] (default is to get result from match file)\n"
           "   [-w <wcs-output>]: write resulting SIP WCS to this file\n"
           "   [-s <scamp-output>]: write resulting SIP WCS to this file\n"
           "   [-o <order>]: SIP distortion order, default 2\n"
           "   [-p <plot output base filename>]\n"
           "   [-i <plot background image>]\n"
           "   [-v]: verbose\n"
           "   [-j <pixel-jitter>]: set image pixel jitter (default 1.0)\n"
           "   [-J <index-jitter>]: set index jitter (in arcsec, default 1.0)\n"
           "   [-X <CRPIX0>]: fix crpix\n"
           "   [-Y <CRPIX1>]: fix crpix\n"
           "\n", progname);
}

/*
 wget "http://antwrp.gsfc.nasa.gov/apod/image/0403/cmsky_cortner_full.jpg"
 #solve-field --backend-config backend.cfg -v --keep-xylist %s.xy --continue --scale-low 10 --scale-units degwidth cmsky_cortner_full.xy --no-tweak
 cp cmsky_cortner_full.xy 1.xy
 cp cmsky_cortner_full.rdls 1.rd
 cp cmsky_cortner_full.wcs 1.wcs
 cp cmsky_cortner_full.jpg 1.jpg
 wget "http://live.astrometry.net/status.php?job=alpha-201003-01883980&get=match.fits" -O 1.match

 X=http://live.astrometry.net/status.php?job=alpha-201003-36217312
 Y=2
 wget "${X}&get=field.xy.fits" -O ${Y}.xy
 wget "${X}&get=index.rd.fits" -O ${Y}.rd
 wget "${X}&get=wcs.fits" -O ${Y}.wcs
 wget "${X}&get=match.fits" -O ${Y}.match
 wget "http://antwrp.gsfc.nasa.gov/apod/image/1003/mb_2010-03-10_SeaGullThor900.jpg" -O ${Y}.jpg

 dstnthing -m 2.match -x 2.xy -r 2.rd -p 2 -i 2.jpg

 X=http://live.astrometry.net/status.php?job=alpha-201002-83316463
 Y=3
 wget "${X}&get=fullsize.png" -O - | pngtopnm | pnmtojpeg > ${Y}.jpg

 dstnthing -m 3.match -x 3.xy -r 3.rd -p 3 -i 3.jpg

 X=http://oven.cosmo.fas.nyu.edu/test/status.php?job=test-201003-60743215
 Y=4

 X=http://live.astrometry.net/status.php?job=alpha-201003-74071720
 Y=5

 wget "${X}&get=field.xy.fits" -O ${Y}.xy
 wget "${X}&get=index.rd.fits" -O ${Y}.rd
 wget "${X}&get=wcs.fits" -O ${Y}.wcs
 wget "${X}&get=match.fits" -O ${Y}.match
 wget "${X}&get=fullsize.png" -O - | pngtopnm | pnmtojpeg > ${Y}.jpg
 echo dstnthing -m ${Y}.match -x ${Y}.xy -r ${Y}.rd -p ${Y} -i ${Y}.jpg
 echo mencoder -o fit${Y}.avi -ovc lavc -lavcopts vcodec=mpeg4:keyint=1:autoaspect mf://${Y}-*c.png -mf fps=4:type=png

 X=http://live.astrometry.net/status.php?job=alpha-201003-75248251
 Y=6

 mencoder mf://${Y}-*c.png -mf fps=4:type=png -o /dev/null -ovc x264 \
 -x264encopts pass=1:turbo:bitrate=900:bframes=1:\
 me=umh:partitions=all:trellis=1:qp_step=4:qcomp=0.7:direct_pred=auto:keyint=300 \
 -vf harddup \
 -oac faac -faacopts br=192:mpeg=4:object=2 -channels 2 -srate 48000 \
 -ofps 4


 mencoder mf://${Y}-*c.png -mf fps=4:type=png -o /dev/null -ovc x264 -x264encopts pass=1:turbo:bitrate=900:bframes=1:me=umh:partitions=all:trellis=1:qp_step=4:qcomp=0.7:direct_pred=auto:keyint=300 -vf harddup -oac faac -faacopts br=192:mpeg=4:object=2 -channels 2 -srate 48000 -ofps 4
 mencoder mf://${Y}-*c.png -mf fps=4:type=png -o v${Y}.avi -ovc x264 -x264encopts pass=2:turbo:bitrate=900:bframes=1:me=umh:partitions=all:trellis=1:qp_step=4:qcomp=0.7:direct_pred=auto:keyint=300 -vf harddup -oac faac -faacopts br=192:mpeg=4:object=2 -channels 2 -srate 48000 -ofps 4

 ffmpeg -f image2 -i ${Y}-%02dc.png -r 12 -s 800x712 fit${Y}.mp4


 ### Works with quicktime and realplayer!
 mencoder "mf://${Y}-*c.png" -mf fps=10 -o fit${Y}.avi -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=800

	   


 */

void makeplot(char* plotfn, char* bgimgfn, int W, int H,
              int Nfield, double* fieldpix, double* fieldsigma2s,
              int Nindex, double* indexpix, int besti, int* theta,
              double* crpix) {
    int i;
    plot_args_t pargs;
    plotimage_t* img;
    cairo_t* cairo;
    logmsg("Creating plot %s\n", plotfn);
    plotstuff_init(&pargs);
    pargs.outformat = PLOTSTUFF_FORMAT_PNG;
    pargs.outfn = plotfn;
    if (bgimgfn) {
        img = plotstuff_get_config(&pargs, "image");
        img->format = PLOTSTUFF_FORMAT_JPG;
        plot_image_set_filename(img, bgimgfn);
        plot_image_setsize(&pargs, img);
        plotstuff_run_command(&pargs, "image");
    } else {
        float rgba[4] = {0, 0, 0.1, 1.0};
        plotstuff_set_size(&pargs, W, H);
        //plotstuff_init2(&pargs);
        plotstuff_set_rgba(&pargs, rgba);
        plotstuff_run_command(&pargs, "fill");
    }
    cairo = pargs.cairo;
    // red circles around every field star.
    cairo_set_color(cairo, "red");
    for (i=0; i<Nfield; i++) {
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE,
                               fieldpix[2*i+0], fieldpix[2*i+1],
                               2.0 * sqrt(fieldsigma2s[i]));
        cairo_stroke(cairo);
    }
    // green crosshairs at every index star.
    cairo_set_color(cairo, "green");
    for (i=0; i<Nindex; i++) {
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_XCROSSHAIR,
                               indexpix[2*i+0], indexpix[2*i+1], 3);
        cairo_stroke(cairo);
    }
    // thick white circles for corresponding field stars.
    cairo_set_line_width(cairo, 2);
    for (i=0; i<=besti; i++) {
        //printf("field %i -> index %i\n", i, theta[i]);
        if (theta[i] < 0)
            continue;
        cairo_set_color(cairo, "white");
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE,
                               fieldpix[2*i+0], fieldpix[2*i+1],
                               2.0 * sqrt(fieldsigma2s[i]));
        cairo_stroke(cairo);
        // thick cyan crosshairs for corresponding index stars.
        cairo_set_color(cairo, "cyan");
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_XCROSSHAIR,
                               indexpix[2*theta[i]+0],
                               indexpix[2*theta[i]+1],
                               3);
        cairo_stroke(cairo);
    }

    cairo_set_color(cairo, "yellow");
    cairo_set_line_width(cairo, 4);
    cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CROSSHAIR,
                           crpix[0], crpix[1], 10);
    cairo_stroke(cairo);

    plotstuff_output(&pargs);
}



int main(int argc, char** args) {
    int c;

    char* xylsfn = NULL;
    //char* wcsfn = NULL;
    char* matchfn = NULL;
    char* rdlsfn = NULL;
    char* plotfn = NULL;
    char* bgimgfn = NULL;

    char* wcsfn = NULL;

    double indexjitter = 1.0; // arcsec
    double pixeljitter = 1.0;
    int i;
    int W, H;
    xylist_t* xyls = NULL;
    rdlist_t* rdls = NULL;
    matchfile* mf;
    MatchObj* mo;
    sip_t sip;
    sip_t* sipout;

    double* fieldpix;
    int Nfield;
    starxy_t* xy;

    rd_t* rd;
    int Nindex;
    double* indexrd;
    double Q2;
    double qc[2];

    char* sipoutfn = NULL;
    char* scampout = NULL;

    int order = 2;

    int loglvl = LOG_MSG;
    double crpix[] = { LARGE_VAL, LARGE_VAL };
    anbool do_crpix = FALSE;

    //FILE* logstream = stderr;
    //fits_use_error_system();

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
			
        case 'o':
            order = atoi(optarg);
            break;
        case 'p':
            plotfn = optarg;
            break;
        case 'i':
            bgimgfn = optarg;
            break;
        case 'j':
            pixeljitter = atof(optarg);
            break;
        case 'J':
            indexjitter = atof(optarg);
            break;
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'r':
            rdlsfn = optarg;
            break;
        case 'x':
            xylsfn = optarg;
            break;
        case 'm':
            matchfn = optarg;
            break;
        case 'w':
            sipoutfn = optarg;
            break;
        case 's':
            scampout = optarg;
            break;
        case 'v':
            loglvl++;
            break;
        case 'W':
            wcsfn = optarg;
            break;
        case 'X':
            crpix[0] = atof(optarg);
            break;
        case 'Y':
            crpix[1] = atof(optarg);
            break;
        }
    }
    if (optind != argc) {
        print_help(args[0]);
        exit(-1);
    }
    if (!xylsfn || !matchfn || !rdlsfn) {
        print_help(args[0]);
        exit(-1);
    }

    do_crpix = (crpix[0] != LARGE_VAL) && (crpix[1] != LARGE_VAL);

    log_init(loglvl);

    // read XYLS.
    xyls = xylist_open(xylsfn);
    if (!xyls) {
        logmsg("Failed to read an xylist from file %s.\n", xylsfn);
        exit(-1);
    }
    xylist_set_include_flux(xyls, FALSE);
    xylist_set_include_background(xyls, FALSE);

    // read RDLS.
    rdls = rdlist_open(rdlsfn);
    if (!rdls) {
        logmsg("Failed to read an rdlist from file %s.\n", rdlsfn);
        exit(-1);
    }

    // image W, H
    W = xylist_get_imagew(xyls);
    H = xylist_get_imageh(xyls);
    if ((W == 0.0) || (H == 0.0)) {
        logmsg("XYLS file %s didn't contain IMAGEW and IMAGEH headers.\n", xylsfn);
        exit(-1);
    }
    logverb("Got image size %i x %i\n", W, H);

    // read match file.
    mf = matchfile_open(matchfn);
    if (!mf) {
        ERROR("Failed to read match file %s", matchfn);
        exit(-1);
    }
    mo = matchfile_read_match(mf);
    if (!mo) {
        ERROR("Failed to read match from file %s", matchfn);
        exit(-1);
    }

    // (x,y) positions of field stars.
    xy = xylist_read_field(xyls, NULL);
    if (!xy) {
        logmsg("Failed to read xyls entries.\n");
        exit(-1);
    }
    Nfield = starxy_n(xy);
    fieldpix = starxy_to_xy_array(xy, NULL);
    logmsg("Found %i field objects\n", Nfield);

    // (ra,dec) of index stars.
    rd = rdlist_read_field(rdls, NULL);
    if (!rd) {
        logmsg("Failed to read rdls entries.\n");
        exit(-1);
    }
    Nindex = rd_n(rd);
    logmsg("Found %i index objects\n", Nindex);

    if (wcsfn) {
        if (!sip_read_tan_or_sip_header_file_ext(wcsfn, 0, &sip, FALSE)) {
            ERROR("Failed to read initial SIP/TAN WCS from \"%s\"", wcsfn);
            return -1;
        }
    } else {
        sip_wrap_tan(&mo->wcstan, &sip);
    }

    // quad radius-squared = AB distance. (/4)
    Q2 = distsq(mo->quadpix, mo->quadpix + 2, 2);
    qc[0] = sip.wcstan.crpix[0];
    qc[1] = sip.wcstan.crpix[1];

    indexrd = malloc(2 * Nindex * sizeof(double));
    for (i=0; i<Nindex; i++)
        rd_getradec(rd, i, indexrd+2*i, indexrd+2*i+1);

    sipout = tweak2(fieldpix, Nfield, pixeljitter,
                    W, H,
                    indexrd, Nindex, indexjitter,
                    qc, Q2,
                    0.25, -100,
                    order, &sip, NULL, NULL, NULL,
                    do_crpix ? crpix : NULL);
    if (!sipout) {
        ERROR("tweak2() failed.\n");
        return -1;
    }

    free(indexrd);

    if (sipoutfn) {
        if (sip_write_to_file(sipout, sipoutfn)) {
            ERROR("Failed to write SIP result to file \"%s\"", sipoutfn);
            return -1;
        }
    }

    if (scampout) {
        qfits_header* hdr = NULL;
        hdr = xylist_get_primary_header(xyls);
        //qfits_header_read(axy->xylsfn);
        // Set NAXIS=2, NAXIS1=IMAGEW, NAXIS2=IMAGEH
        fits_header_mod_int(hdr, "NAXIS", 2, NULL);
        fits_header_add_int(hdr, "NAXIS1", W, NULL);
        fits_header_add_int(hdr, "NAXIS2", H, NULL);
        if (scamp_write_field(hdr, sipout, xy, scampout)) {
            ERROR("Failed to write SIP result to Scamp file \"%s\"", scampout);
            return -1;
        }
    }

    sip_free(sipout);
    xylist_close(xyls);
    matchfile_close(mf);
    rdlist_close(rdls);

    return 0;
}



