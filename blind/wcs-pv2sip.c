/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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
#include <assert.h>

#include "fitsioutils.h"
#include "ioutils.h"
#include "qfits.h"
#include "errors.h"
#include "log.h"
#include "an-bool.h"
#include "sip.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "starxy.h"
#include "tweak.h"

/*
 Scamp's copy of wcslib has  "raw_to_pv" in "proj.c".

 wcslib-4.4.4 has no such thing...


 tanrev()
 called by celrev(), by wcsrev()

 wcsrev()
 - calls linrev()          (pixcrd -> imgcrd)
 - calls celrev(, x=imgcrd,y=imgcrd)
 -   calls tanrev()        (x,y -> phi,theta)
 -     calls raw_to_pv()   (x,y -> xp, yp)
 -   calls sphrev()        (phi,theta -> lng,lat)

Got imcrd = (-0.0154408,-0.00816145)
= sip.c (x,y) before -deg2rd

 // lng = 0, lat = 1


CFHTLS via CVO headers:

PV1_0   =   6.888383659426E-03 / xi =   PV1_0                       
PV1_1   =    1.01653233841     /      + PV1_1 * x                   
PV1_2   =   7.672948165520E-03 /      + PV1_2 * y                   
PV1_3   =    0.00000000000     /      + PV1_3 * sqrt(x**2 + y**2)   
PV1_4   =  -1.613906528201E-03 /      + PV1_4 * x**2                
PV1_5   =  -1.170568723090E-03 /      + PV1_5 * x*y                 
PV1_6   =  -6.175903023930E-04 /      + PV1_6 * y**2                
PV1_7   =  -2.467136651059E-02 /      + PV1_7 * x**3                
PV1_8   =  -1.806292484275E-03 /      + PV1_8 * x**2 * y            
PV1_9   =  -2.439766180834E-02 /      + PV1_9 * x * y**2            
PV1_10  =  -4.872349869816E-04 /      + PV1_10* y**3                
PV2_0   =   5.963468826495E-03 / eta =  PV2_0                       
PV2_1   =    1.01450676752     /      + PV2_1 * y                   
PV2_2   =   7.677145522017E-03 /      + PV2_2 * x                   
PV2_3   =    0.00000000000     /      + PV2_3 * sqrt(x**2 + y**2)   
PV2_4   =  -1.353520666662E-03 /      + PV2_4 * y**2                
PV2_5   =  -1.247925715556E-03 /      + PV2_5 * y*x                 
PV2_6   =  -5.742047327244E-04 /      + PV2_6 * x**2                
PV2_7   =  -2.435753005264E-02 /      + PV2_7 * y**3                
PV2_8   =  -1.842813673530E-03 /      + PV2_8 * y**2 * x            
PV2_9   =  -2.444782516561E-02 /      + PV2_9 * y * x**2            
PV2_10  =  -4.717653697970E-04 /      + PV2_10* x**3                


   a = prj->p+100;
   b = prj->p;
   xp = *(a++);
   xp += *(a++)*x;


 */


int wcs_pv2sip(const char* wcsinfn, int ext,
			   const char* wcsoutfn,
			   anbool scamp_head_file,
			   double* xy, int Nxy,
			   int imageW, int imageH) {
	qfits_header* hdr = NULL;
	double* radec = NULL;
	int rtn = -1;
	tan_t tanwcs;
	double x,y, px,py;
	double xyz[3];

	double* xorig = NULL;
	double* yorig = NULL;
	double* rddist = NULL;
	int i, j;

	//           1  x  y  r x2 xy y2 x3 x2y xy2 y3 r3 x4 x3y x2y2 xy3 y4
	//          x5 x4y x3y2 x2y3 xy4 y5 r5 x6 x5y x4y2, x3y3 x2y4 xy5 y6
	//          x7 x6y x5y2 x4y3 x3y4 x2y5 xy6 y7 r7
	int xp[] = { 0, 1, 0, 0, 2, 1, 0, 3,  2,  1, 0, 0, 4,  3,   2,  1, 0,
				 5,  4,   3,   2,  1, 5, 0, 6,  5,   4,    3,   2,  1, 0,
				 7,  6,   5,   4,   3,   2,  1, 0, 0};
	int yp[] = { 0, 0, 1, 0, 0, 1, 2, 0,  1,  2, 3, 0, 0,  1,   2,  3, 4,
				 0,  1,   2,   3,  4, 0, 0, 0,  1,   2,    3,   4,  5, 6,
				 0,  1,   2,   3,   4,   5,  6, 7, 0};
	int rp[] = { 0, 0, 0, 1, 0, 0, 0, 0,  0,  0, 0, 3, 0,  0,   0,  0, 0,
				 0,  0,   0,   0,  0, 0, 5, 0,  0,   0,    0,   0,  0, 0,
				 0,  0,   0,   0,   0,   0,  0, 0, 7};
	double xpows[8];
	double ypows[8];
	double rpows[8];
	double pv1[40];
	double pv2[40];
	double r;

	if (scamp_head_file) {
		size_t sz = 0;
		char* txt;
		char* prefix;
		int np;
		int nt;
		unsigned char* txthdr;
		sl* lines;
		int i;
		txt = file_get_contents(wcsinfn, &sz, TRUE);
		if (!txt) {
			ERROR("Failed to read file %s", wcsinfn);
			goto bailout;
		}
		lines = sl_split(NULL, txt, "\n");
		prefix =
			"SIMPLE  =                    T / Standard FITS file                             "
			"BITPIX  =                    8 / ASCII or bytes array                           "
			"NAXIS   =                    0 / Minimal header                                 "
			"EXTEND  =                    T / There may be FITS ext                          "
			"WCSAXES =                    2 /                                                ";
		np = strlen(prefix);
		nt = np + FITS_LINESZ * sl_size(lines);
		txthdr = malloc(nt);
		memset(txthdr, ' ', np + FITS_LINESZ * sl_size(lines));
		memcpy(txthdr, prefix, np);
		for (i=0; i<sl_size(lines); i++)
			memcpy(txthdr + np + i*FITS_LINESZ, sl_get(lines, i), strlen(sl_get(lines, i)));
		sl_free2(lines);
		hdr = qfits_header_read_hdr_string(txthdr, nt);
		free(txthdr);
		free(txt);
	} else {
		hdr = qfits_header_readext(wcsinfn, ext);
	}
	if (!hdr) {
		ERROR("Failed to read header: file %s, ext %i\n", wcsinfn, ext);
		goto bailout;
	}
	
	tan_read_header(hdr, &tanwcs);

	for (i=0; i<sizeof(pv1)/sizeof(double); i++) {
		char key[10];
		sprintf(key, "PV1_%i", i);
		pv1[i] = qfits_header_getdouble(hdr, key, 0.0);
		sprintf(key, "PV2_%i", i);
		pv2[i] = qfits_header_getdouble(hdr, key, 0.0);
	}

	xorig = malloc(Nxy * sizeof(double));
	yorig = malloc(Nxy * sizeof(double));
	rddist = malloc(2 * Nxy * sizeof(double));

	for (j=0; j<Nxy; j++) {
		xorig[j] = xy[2*j+0];
		yorig[j] = xy[2*j+1];

		tan_pixelxy2iwc(&tanwcs, xorig[j], yorig[j], &x, &y);
		r = sqrt(x*x + y*y);
		xpows[0] = ypows[0] = rpows[0] = 1.0;
		for (i=1; i<sizeof(xpows)/sizeof(double); i++) {
			xpows[i] = xpows[i-1]*x;
			ypows[i] = ypows[i-1]*y;
			rpows[i] = rpows[i-1]*r;
		}
		px = py = 0;
		for (i=0; i<sizeof(xp)/sizeof(int); i++) {
			px += pv1[i] * xpows[xp[i]] * ypows[yp[i]] * rpows[rp[i]];
			py += pv2[i] * ypows[xp[i]] * xpows[yp[i]] * rpows[rp[i]];
		}
		tan_iwc2xyzarr(&tanwcs, px, py, xyz);
		xyzarr2radecdeg(xyz, rddist+2*j, rddist+2*j+1);
	}

	//
	{
		starxy_t sxy;
		tweak_t* t;
		il* imgi;
		il* refi;
		int sip_order = 5;
		int sip_inv_order = 5;

		sxy.N = Nxy;
		sxy.x = xorig;
		sxy.y = yorig;

		imgi = il_new(256);
		refi = il_new(256);
		for (i=0; i<Nxy; i++) {
			il_append(imgi, i);
			il_append(refi, i);
		}

		t = tweak_new();
		t->sip->a_order = t->sip->b_order = sip_order;
		t->sip->ap_order = t->sip->bp_order = sip_inv_order;
		tweak_push_wcs_tan(t, &tanwcs);
		tweak_push_ref_ad_array(t, rddist, Nxy);
		tweak_push_image_xy(t, &sxy);
		tweak_push_correspondence_indices(t, imgi, refi, NULL, NULL);
		tweak_go_to(t, TWEAK_HAS_LINEAR_CD);
		if (imageW)
			t->sip->wcstan.imagew = imageW;
		if (imageH)
			t->sip->wcstan.imageh = imageH;
		sip_write_to_file(t->sip, wcsoutfn);
		tweak_free(t);
	}
	rtn = 0;

 bailout:
	free(xorig);
	free(yorig);
	free(rddist);
	qfits_header_destroy(hdr);
	free(radec);
	return rtn;
}

		/*
		 xp = *(a++);     //a0
		 xp += *(a++)*x; //a1
		 xp += *(a++)*y; //a2
		 xp += *(a++)*r; //a3
		 xp += *(a++)*(x2=x*x); //a4
		 xp += *(a++)*(xy=x*y); //a5
		 xp += *(a++)*y2; //a6
		 xp += *(a++)*(x3=x*x2); //a7
		 xp += *(a++)*x2*y; //a8
		 xp += *(a++)*x*y2; //a9
		 xp += *(a++)*y3; //a10
		 xp += *(a++)*(r3=r*r*r); //a11
		 xp += *(a++)*(x4=x2*x2); //a12
		 xp += *(a++)*x3*y; //a13
		 xp += *(a++)*x2*y2; //a14
		 xp += *(a++)*x*y3; //a15
		 xp += *(a++)*y4; //a16
		 xp += *(a++)*(x5=x4*x); //a17
		 xp += *(a++)*x4*y; //a18
		 xp += *(a++)*x3*y2; //a19
		 xp += *(a++)*x2*y3; //a20
		 xp += *(a++)*x*y4; //a21
		 xp += *(a++)*y5; //a22
		 xp += *(a++)*(r5=r3*r*r); //a23
		 xp += *(a++)*(x6=x5*x); //a24
		 xp += *(a++)*x5*y; //a25
		 xp += *(a++)*x4*y2; //a26
		 xp += *(a++)*x3*y3; //a27
		 xp += *(a++)*x2*y4; //a28
		 xp += *(a++)*x*y5; //a29
		 xp += *(a++)*y6; //a30
		 xp += *(a++)*(x7=x6*x); //a31
		 xp += *(a++)*x6*y; //a32
		 xp += *(a++)*x5*y2; //a33
		 xp += *(a++)*x4*y3; //a34
		 xp += *(a++)*x3*y4; //a35
		 xp += *(a++)*x2*y5; //a36
		 xp += *(a++)*x*y6; //a37
		 xp += *(a++)*y7; //a38
		 xp += *a*(r7=r5*r*r); //a39
		 */


#include <stdlib.h>
#include <sys/param.h>
#include <math.h>

#include "boilerplate.h"
#include "bl.h"

const char* OPTIONS = "hve:sx:X:y:Y:a:b:W:H:";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] <input-wcs> <output-wcs>\n"
		   "   [-e <extension>] FITS HDU number to read WCS from (default 0 = primary)\n"
		   "   [-s]: treat input as Scamp .head file\n"
		   "   [-v]: +verboseness\n"
		   " Set the IMAGEW, IMAGEH in the output file:\n"
		   "   [-W <int>]\n"
		   "   [-H <int>]\n"
		   " Set the pixel values used to compute the distortion polynomials with:\n"
		   "   [-x <x-low>] (default: 1)\n"
		   "   [-y <y-low>] (default: 1)\n"
		   "   [-X <x-high>] (default: 1000)\n" // or value of IMAGEW header)\n"
		   "   [-Y <y-high>] (default: 1000)\n" // or value of IMAGEH header)\n"
		   "   [-a <x-step>] (default: closest to 100 yielding whole number of steps)\n"
		   "   [-b <y-step>] (default: closest to 100 yielding whole number of steps)\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int loglvl = LOG_MSG;
	char** myargs;
	int nargs;
	int c;

	char* wcsinfn = NULL;
	char* wcsoutfn = NULL;
	int ext = 0;
	anbool scamp = FALSE;
	double xlo = 1;
	double xhi = 1000;
	double xstep = 0;
	double ylo = 1;
	double yhi = 1000;
	double ystep = 0;

	dl* xylst;
	double x,y;
	double* xy;
	int Nxy;
	int W, H;

	W = H = 0;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case 'W':
			W = atoi(optarg);
			break;
		case 'H':
			H = atoi(optarg);
			break;
		case 'x':
			xlo = atof(optarg);
			break;
		case 'X':
			xhi = atof(optarg);
			break;
		case 'a':
			xstep = atof(optarg);
			break;
		case 'y':
			ylo = atof(optarg);
			break;
		case 'Y':
			yhi = atof(optarg);
			break;
		case 'b':
			ystep = atof(optarg);
			break;
		case 's':
			scamp = TRUE;
			break;
		case 'e':
			ext = atoi(optarg);
			break;
		case 'v':
			loglvl++;
			break;
		case '?':
		case 'h':
			print_help(args[0]);
			exit(0);
		}
	}
	nargs = argc - optind;
	myargs = args + optind;

	if (nargs != 2) {
		print_help(args[0]);
		exit(-1);
	}
	wcsinfn = myargs[0];
	wcsoutfn = myargs[1];

	log_init(loglvl);
	fits_use_error_system();

	logmsg("Reading WCS (with PV distortions) from %s, ext %i\n", wcsinfn, ext);
	logmsg("Writing WCS (with SIP distortions) to %s\n", wcsoutfn);

	assert(xhi >= xlo);
	assert(yhi >= ylo);
	if (xstep == 0) {
		int nsteps = MAX(1, round((xhi - xlo)/100.0));
		xstep = (xhi - xlo) / (double)nsteps;
	}
	if (ystep == 0) {
		int nsteps = MAX(1, round((yhi - ylo)/100.0));
		ystep = (yhi - ylo) / (double)nsteps;
	}
	logverb("Stepping from x = %g to %g, steps of %g\n", xlo, xhi, xstep);
	logverb("Stepping from y = %g to %g, steps of %g\n", ylo, yhi, ystep);

	xylst = dl_new(256);
	for (y=ylo; y<=(yhi+0.001); y+=ystep) {
		for (x=xlo; x<=(xhi+0.001); x+=xstep) {
			dl_append(xylst, x);
			dl_append(xylst, y);
		}
	}
	Nxy = dl_size(xylst)/2;
	xy = dl_to_array(xylst);
	dl_free(xylst);

	if (wcs_pv2sip(wcsinfn, ext, wcsoutfn, scamp, xy, Nxy, W, H)) {
		exit(-1);
	}

	free(xy);

	return 0;
}
