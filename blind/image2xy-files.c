/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Michael Blanton, Keir Mierle, David W. Hogg,
  Sam Roweis and Dustin Lang.

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

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/param.h>
#include <unistd.h>
#include <math.h>

#include "image2xy-files.h"
#include "image2xy.h"
#include "fitsio.h"
#include "ioutils.h"
#include "simplexy2.h"
#include "svn.h"
#include "dimage.h"
#include "errors.h"
#include "log.h"

static void cfitserr(int status) {
    sl* msgs = sl_new(4);
    char errmsg[FLEN_ERRMSG];
    int i;
    // pop the cfitsio error message stack...
    fits_get_errstatus(status, errmsg);
    sl_append(msgs, errmsg);
    while (fits_read_errmsg(errmsg))
        sl_append(msgs, errmsg);
    // ... and push it onto the astrometry.net error message stack... sigh.
    for (i=sl_size(msgs)-1; i>=0; i--)
        ERROR(sl_get(msgs, i));
    sl_free2(msgs);
}

#define FITS_CHECK(msg, ...) \
do { \
if (status) { \
cfitserr(status); \
ERROR(msg, ##__VA_ARGS__); \
goto bailout; \
} \
} while(0)

int image2xy_files(const char* infn, const char* outfn,
				   bool do_u8, int downsample, int downsample_as_required) {
	fitsfile *fptr = NULL;
	fitsfile *ofptr = NULL;
	int status = 0; // FIXME should have ostatus too
	int naxis;
	int maxnpeaks=0, npeaks;
	long naxisn[2];
	int kk;
	float sigma;
	int nhdus,maxper=0,maxsize=0,halfbox=0,hdutype,nimgs;
	float dpsf=0,plim=0,dlim=0,saddle=0;
    char* str;

	fits_open_file(&fptr, infn, READONLY, &status);
    FITS_CHECK("Failed to open FITS input file %s", infn);

	// Are there multiple HDU's?
	fits_get_num_hdus(fptr, &nhdus, &status);
    FITS_CHECK("Failed to read number of HDUs for input file %s", infn);
    logverb("nhdus=%d\n", nhdus);

	// Create output file
	fits_create_file(&ofptr, outfn, &status);
    FITS_CHECK("Failed to open FITS output file %s", outfn);

	fits_create_img(ofptr, 8, 0, NULL, &status);
    FITS_CHECK("Failed to create output image");

	fits_write_key(ofptr, TSTRING, "SRCFN", (char*)infn, "Source image", &status);
	/* Parameters for simplexy; save for debugging */
	fits_write_comment(ofptr, "Parameters used for source extraction", &status);

	fits_write_history(ofptr, "Created by Astrometry.net's image2xy program.", &status);
    FITS_CHECK("Failed to write HISTORY headers");

    asprintf_safe(&str, "SVN URL: %s", svn_url());
	fits_write_history(ofptr, str, &status);
    FITS_CHECK("Failed to write SVN HISTORY headers");
    free(str);
    asprintf_safe(&str, "SVN Rev: %i", svn_revision());
	fits_write_history(ofptr, str, &status);
    FITS_CHECK("Failed to write SVN HISTORY headers");
    free(str);
    asprintf_safe(&str, "SVN Date: %s", svn_date());
	fits_write_history(ofptr, str, &status);
    FITS_CHECK("Failed to write SVN HISTORY headers");
    free(str);
	fits_write_history(ofptr, "Visit us on the web at http://astrometry.net/", &status);
    FITS_CHECK("Failed to write SVN HISTORY headers");

	nimgs = 0;

	// Run simplexy on each HDU
	for (kk=1; kk <= nhdus; kk++) {
		char* ttype[] = {"X","Y","FLUX","BACKGROUND"};
		char* tform[] = {"E","E","E","E"};
		char* tunit[] = {"pix","pix","unknown","unknown"};
		long* fpixel;
		int a;
		int w, h;
        int bitpix;

        // FIXME - we leak this memory on error.
        float *thedata = NULL;
        unsigned char* theu8data = NULL;
        float *x;
        float *y;
        float *flux;
        float* background;

		fits_movabs_hdu(fptr, kk, &hdutype, &status);
		fits_get_hdu_type(fptr, &hdutype, &status);

		if (hdutype != IMAGE_HDU) 
			continue;

		fits_get_img_dim(fptr, &naxis, &status);
        FITS_CHECK("Failed to find image dimensions for HDU %i", kk);

		fits_get_img_size(fptr, 2, naxisn, &status);
        FITS_CHECK("Failed to find image dimensions for HDU %i", kk);

		nimgs++;

        logverb("Got naxis=%d, na1=%lu, na2=%lu\n", naxis, naxisn[0], naxisn[1]);

		if (naxis > 2)
            logmsg("This looks like a multi-color image: processing the first image plane only.  (NAXIS=%i)\n", naxis);

        fits_get_img_type(fptr, &bitpix, &status);
        FITS_CHECK("Failed to get FITS image type");

		fpixel = malloc(naxis * sizeof(long));
		for (a=0; a<naxis; a++)
			fpixel[a] = 1;

        if (bitpix == 8 && do_u8 && !downsample) {
            // u8 image.
            theu8data = malloc(naxisn[0] * naxisn[1]);
            if (!theu8data) {
                SYSERROR("Failed to allocate u8 image array");
                goto bailout;
            }
            fits_read_pix(fptr, TBYTE, fpixel, naxisn[0]*naxisn[1], NULL,
                          theu8data, NULL, &status);
        } else {
            thedata = malloc(naxisn[0] * naxisn[1] * sizeof(float));
            if (!thedata) {
                SYSERROR("Failed to allocate image array");
                goto bailout;
            }
            fits_read_pix(fptr, TFLOAT, fpixel, naxisn[0]*naxisn[1], NULL,
                          thedata, NULL, &status);
        }
		free(fpixel);
        FITS_CHECK("Failed to read image pixels");

		image2xy_image(theu8data, thedata, naxisn[0], naxisn[1],
					   downsample, downsample_as_required,
					   dpsf, plim, dlim, saddle, maxper, maxsize, halfbox,
					   maxnpeaks,
					   &x, &y, &flux, &background, &npeaks, &sigma);

		free(theu8data);
		free(thedata);

		fits_create_tbl(ofptr, BINARY_TBL, npeaks, 4, ttype, tform,
                        tunit, "SOURCES", &status);
        FITS_CHECK("Failed to create output table");

		fits_write_col(ofptr, TFLOAT, 1, 1, 1, npeaks, x, &status);
        FITS_CHECK("Failed to write X column");

		fits_write_col(ofptr, TFLOAT, 2, 1, 1, npeaks, y, &status);
        FITS_CHECK("Failed to write Y column");

		fits_write_col(ofptr, TFLOAT, 3, 1, 1, npeaks, flux, &status);
        FITS_CHECK("Failed to write FLUX column");

		fits_write_col(ofptr, TFLOAT, 4, 1, 1, npeaks, background, &status);
        FITS_CHECK("Failed to write BACKGROUND column");

		fits_modify_comment(ofptr, "TTYPE1", "X coordinate", &status);
        FITS_CHECK("Failed to set X TTYPE");

		fits_modify_comment(ofptr, "TTYPE2", "Y coordinate", &status);
        FITS_CHECK("Failed to set Y TTYPE");

		fits_modify_comment(ofptr, "TTYPE3", "Flux of source", &status);
        FITS_CHECK("Failed to set FLUX TTYPE");

		fits_modify_comment(ofptr, "TTYPE4", "Sky background of source", &status);
        FITS_CHECK("Failed to set BACKGROUND TTYPE");

		fits_write_key(ofptr, TINT, "SRCEXT", &kk,
                       "Extension number in src image", &status);
        FITS_CHECK("Failed to write SRCEXT");

		w = naxisn[0];
		h = naxisn[1];
		fits_write_key(ofptr, TINT, "IMAGEW", &w, "Input image width", &status);
        FITS_CHECK("Failed to write IMAGEW");

		fits_write_key(ofptr, TINT, "IMAGEH", &h, "Input image height", &status);
        FITS_CHECK("Failed to write IMAGEH");

		fits_write_key(ofptr, TFLOAT, "ESTSIGMA", &sigma,
				"Estimated source image variance", &status);
        FITS_CHECK("Failed to write ESTSIGMA");

        fits_write_key(ofptr, TFLOAT, "DPSF", &dpsf, "image2xy Assumed gaussian psf width", &status);
        fits_write_key(ofptr, TFLOAT, "PLIM", &plim, "image2xy Significance to keep", &status);
        fits_write_key(ofptr, TFLOAT, "DLIM", &dlim, "image2xy Closest two peaks can be", &status);
        fits_write_key(ofptr, TFLOAT, "SADDLE", &saddle, "image2xy Saddle difference (in sig)", &status);
        fits_write_key(ofptr, TINT, "MAXPER", &maxper, "image2xy Max num of peaks per object", &status);
        fits_write_key(ofptr, TINT, "MAXPEAKS", &maxnpeaks, "image2xy Max num of peaks total", &status);
        fits_write_key(ofptr, TINT, "MAXSIZE", &maxsize, "image2xy Max size for extended objects", &status);
        fits_write_key(ofptr, TINT, "HALFBOX", &halfbox, "image2xy Half-size for sliding sky window", &status);


		fits_write_comment(ofptr,
			"The X and Y points are specified assuming 1,1 is "
			"the center of the leftmost bottom pixel of the "
			"image in accordance with the FITS standard.", &status);
        FITS_CHECK("Failed to write comments");

		free(x);
		free(y);
        free(flux);
        free(background);
	}

	// Put in the optional NEXTEND keywoard
	fits_movabs_hdu(ofptr, 1, &hdutype, &status);
	assert(hdutype == IMAGE_HDU);
	fits_write_key(ofptr, TINT, "NEXTEND", &nimgs, "Number of extensions", &status);
	if (status == END_OF_FILE)
		status = 0; /* Reset after normal error */
    FITS_CHECK("Failed to write NEXTEND");

	fits_close_file(fptr, &status);
    FITS_CHECK("Failed to close FITS input file");
    fptr = NULL;

	fits_close_file(ofptr, &status);
    FITS_CHECK("Failed to close FITS output file");

    // for valgrind
    dselip_cleanup();

	return 0;

 bailout:
    if (fptr)
        fits_close_file(fptr, &status);
    if (ofptr)
        fits_close_file(ofptr, &status);
    return -1;
}
