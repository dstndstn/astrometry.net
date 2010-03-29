
#include <stdio.h>

#include "bl.h"
#include "blind_wcs.h"
#include "sip.h"
#include "sip_qfits.h"
#include "log.h"
#include "errors.h"
#include "tweak.h"
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

static const char* OPTIONS = "hx:m:r:vj:p:i:";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   //"   -w <WCS input file>\n"
		   "   -m <match input file>\n"
		   "   -x <xyls input file>\n"
		   "   -r <rdls input file>\n"
		   "   [-p <plot output base filename>]\n"
		   "   [-i <plot background image>]\n"
           "   [-v]: verbose\n"
		   "   [-j <pixel-jitter>]: set pixel jitter (default 1.0)\n"
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
		plotstuff_set_size(&pargs, W, H);
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


extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;

	char* xylsfn = NULL;
	//char* wcsfn = NULL;
	char* matchfn = NULL;
	char* rdlsfn = NULL;
	char* plotfn = NULL;
	char* bgimgfn = NULL;

	double pixeljitter = 1.0;
	int i;
	int W, H;
	xylist_t* xyls = NULL;
	rdlist_t* rdls = NULL;
	matchfile* mf;
	MatchObj* mo;
	sip_t sip;

	double* fieldpix;
	int Nfield;
	starxy_t* xy;

	rd_t* rd;
	int Nindex;

	int step;

	double ra,dec;

	double* indexpix;
	double* fieldsigma2s;
	int besti;
	int* theta;
	double* odds;
	double logodds;
	double Q2, R2;
	double qc[2];
	double gamma;

	double* weights;
	double* matchxyz;
	double* matchxy;
	int Nmatch;
	tan_t newtan;

	int loglvl = LOG_MSG;
	//double crpix[] = { HUGE_VAL, HUGE_VAL };
	//FILE* logstream = stderr;
	//fits_use_error_system();

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case 'p':
			plotfn = optarg;
			break;
		case 'i':
			bgimgfn = optarg;
			break;
		case 'j':
			pixeljitter = atof(optarg);
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
			/*
			 case 'w':
			 wcsfn = optarg;
			 break;
			 */
		case 'm':
			matchfn = optarg;
			break;
        case 'v':
            loglvl++;
            break;
			/*
			 case 'X':
			 crpix[0] = atof(optarg);
			 break;
			 case 'Y':
			 crpix[1] = atof(optarg);
			 break;
			 */
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

	//log_to(logstream);
	log_init(loglvl);
	//errors_log_to(logstream);

	/*
	 if (W == 0 || H == 0) {
	 logerr("Need -W, -H\n");
	 exit(-1);
	 }
	 if (crpix[0] == HUGE_VAL)
	 crpix[0] = W/2.0;
	 if (crpix[1] == HUGE_VAL)
	 crpix[1] = H/2.0;
	 */


	// read XYLS.
	xyls = xylist_open(xylsfn);
	if (!xyls) {
		logmsg("Failed to read an xylist from file %s.\n", xylsfn);
		exit(-1);
	}

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

	sip_wrap_tan(&mo->wcstan, &sip);

	// quad radius-squared = AB distance.
	Q2 = distsq(mo->quadpix, mo->quadpix + 2, 2);
	qc[0] = sip.wcstan.crpix[0];
	qc[1] = sip.wcstan.crpix[1];

	indexpix = malloc(2 * Nindex * sizeof(double));
	fieldsigma2s = malloc(Nfield * sizeof(double));
	weights = malloc(Nfield * sizeof(double));
	matchxyz = malloc(Nfield * 3 * sizeof(double));
	matchxy = malloc(Nfield * 2 * sizeof(double));

	// variance growth rate wrt radius.
	gamma = 1.0;
	int STEPS = 20;

	for (step=0; step<STEPS; step++) {
		// Anneal
		gamma = pow(0.9, step);
		if (step == STEPS-1)
			gamma = 0.0;

		printf("Set gamma = %g\n", gamma);

		// Project RDLS into pixel space.
		for (i=0; i<Nindex; i++) {
			bool ok;
			rd_getradec(rd, i, &ra, &dec);
			ok = sip_radec2pixelxy(&sip, ra, dec, indexpix + i*2, indexpix + i*2 + 1);
			assert(ok);
		}
		logmsg("CRPIX is (%g,%g)\n", sip.wcstan.crpix[0], sip.wcstan.crpix[1]);

		for (i=0; i<Nfield; i++) {
			R2 = distsq(qc, fieldpix + 2*i, 2);
			fieldsigma2s[i] = square(pixeljitter) * (1.0 + gamma * R2/Q2);
		}

		logodds = verify_star_lists(indexpix, Nindex,
									fieldpix, fieldsigma2s, Nfield,
									W*H, 0.25,
									log(1e-100), HUGE_VAL,
									&besti, &odds, &theta, NULL);
		logmsg("Logodds: %g\n", logodds);
		logmsg("besti: %i\n", besti);

		if (plotfn) {
			char fn[256];
			sprintf(fn, "%s-%02i%c.png", plotfn, step, 'a');
			makeplot(fn, bgimgfn, W, H, Nfield, fieldpix, fieldsigma2s,
					 Nindex, indexpix, Nfield, theta, sip.wcstan.crpix);
		}

		Nmatch = 0;
		logmsg("Weights:");
		for (i=0; i<Nfield; i++) {
			double ra,dec;
			if (theta[i] < 0)
				continue;
			rd_getradec(rd, theta[i], &ra, &dec);
			radecdeg2xyzarr(ra, dec, matchxyz + Nmatch*3);
			memcpy(matchxy + Nmatch*2, fieldpix + i*2, 2*sizeof(double));
			weights[Nmatch] = verify_logodds_to_weight(odds[i]);
			logmsg(" %.2f", weights[Nmatch]);
			Nmatch++;
		}
		logmsg("\n");

		blind_wcs_compute_weighted(matchxyz, matchxy, weights, Nmatch, &newtan, NULL);

		logmsg("Original TAN WCS:\n");
		tan_print_to(&sip.wcstan, stdout);
		logmsg("Using %i (weighted) matches, new TAN WCS is:\n", Nmatch);
		tan_print_to(&newtan, stdout);

		if (plotfn) {
			char fn[256];

			for (i=0; i<Nindex; i++) {
				bool ok;
				rd_getradec(rd, i, &ra, &dec);
				ok = tan_radec2pixelxy(&newtan, ra, dec, indexpix + i*2, indexpix + i*2 + 1);
				assert(ok);
			}

			sprintf(fn, "%s-%02i%c.png", plotfn, step, 'b');
			makeplot(fn, bgimgfn, W, H, Nfield, fieldpix, fieldsigma2s,
					 Nindex, indexpix, Nfield, theta, newtan.crpix);
		}

		sip_t* newsip;
		tweak_t* t = tweak_new();
		starxy_t* sxy = starxy_new(Nmatch, FALSE, FALSE);
		il* imginds = il_new(256);
		il* refinds = il_new(256);
		dl* wts = dl_new(256);

		for (i=0; i<Nmatch; i++) {
			starxy_set_x(sxy, i, matchxy[2*i+0]);
			starxy_set_y(sxy, i, matchxy[2*i+1]);
		}
		tweak_init(t);
		tweak_push_ref_xyz(t, matchxyz, Nmatch);
		tweak_push_image_xy(t, sxy);
		for (i=0; i<Nmatch; i++) {
			il_append(imginds, i);
			il_append(refinds, i);
			dl_append(wts, weights[i]);
		}
		tweak_push_correspondence_indices(t, imginds, refinds, NULL, wts);
		tweak_push_wcs_tan(t, &newtan);
		t->sip->a_order = t->sip->b_order = t->sip->ap_order = t->sip->bp_order = 2;
		t->weighted_fit = TRUE;
		for (i=0; i<10; i++) {
			tweak_go_to(t, TWEAK_HAS_LINEAR_CD);
			//logmsg("\n");
			//sip_print_to(t->sip, stdout);
			t->state &= ~TWEAK_HAS_LINEAR_CD;
		}
		logmsg("Got SIP:\n");
		sip_print_to(t->sip, stdout);
		newsip = t->sip;

		if (plotfn) {
			char fn[256];

			for (i=0; i<Nindex; i++) {
				bool ok;
				rd_getradec(rd, i, &ra, &dec);
				ok = sip_radec2pixelxy(newsip, ra, dec, indexpix + i*2, indexpix + i*2 + 1);
				assert(ok);
			}

			sprintf(fn, "%s-%02i%c.png", plotfn, step, 'c');
			makeplot(fn, bgimgfn, W, H, Nfield, fieldpix, fieldsigma2s,
					 Nindex, indexpix, Nfield, theta, newsip->wcstan.crpix);
		}

		memcpy(&sip, newsip, sizeof(sip_t));

		starxy_free(sxy);
		tweak_free(t);
		free(theta);
		free(odds);
	}

	free(fieldsigma2s);
	free(fieldpix);
	free(indexpix);


	if (xylist_close(xyls)) {
		logmsg("Failed to close XYLS file.\n");
	}
	return 0;

}



