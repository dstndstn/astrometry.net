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
#include "errors.h"
#include "log.h"
#include "cfitsutils.h"

int image2xy_files(const char* infn, const char* outfn,
				   bool do_u8, int downsample, int downsample_as_required,
                   int extension,
				   const char* bgimg, const char* bgsubimg, const char* maskimg,
				   double plim, int halfbox, int plane) {
	fitsfile *fptr = NULL;
	fitsfile *ofptr = NULL;
	int status = 0; // FIXME should have ostatus too
	int naxis;
	long naxisn[2];
	int kk;
	int nhdus,hdutype,nimgs;
	simplexy_t s;
    char* str;

    // QFITS to CFITSIO extension convention switch
    extension++;

	fits_open_file(&fptr, infn, READONLY, &status);
    CFITS_CHECK("Failed to open FITS input file %s", infn);

	// Are there multiple HDU's?
	fits_get_num_hdus(fptr, &nhdus, &status);
    CFITS_CHECK("Failed to read number of HDUs for input file %s", infn);
    logverb("nhdus=%d\n", nhdus);

    if (extension > nhdus) {
        logerr("Requested extension %i is greater than number of extensions (%i) in file %s\n",
               extension, nhdus, infn);
        return -1;
    }

	// Create output file
	fits_create_file(&ofptr, outfn, &status);
    CFITS_CHECK("Failed to open FITS output file %s", outfn);

	fits_create_img(ofptr, 8, 0, NULL, &status);
    CFITS_CHECK("Failed to create output image");

	fits_write_key(ofptr, TSTRING, "SRCFN", (char*)infn, "Source image", &status);
    if (extension)
        fits_write_key(ofptr, TINT, "SRCEXT", &extension, "Source image extension (1=primary)", &status);

	/* Parameters for simplexy; save for debugging */
	fits_write_comment(ofptr, "Parameters used for source extraction", &status);

	fits_write_history(ofptr, "Created by Astrometry.net's image2xy program.", &status);
    CFITS_CHECK("Failed to write HISTORY headers");

    asprintf_safe(&str, "SVN URL: %s", svn_url());
	fits_write_history(ofptr, str, &status);
    CFITS_CHECK("Failed to write SVN HISTORY headers");
    free(str);
    asprintf_safe(&str, "SVN Rev: %i", svn_revision());
	fits_write_history(ofptr, str, &status);
    CFITS_CHECK("Failed to write SVN HISTORY headers");
    free(str);
    asprintf_safe(&str, "SVN Date: %s", svn_date());
	fits_write_history(ofptr, str, &status);
    CFITS_CHECK("Failed to write SVN HISTORY headers");
    free(str);
	fits_write_history(ofptr, "Visit us on the web at http://astrometry.net/", &status);
    CFITS_CHECK("Failed to write SVN HISTORY headers");

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

        if (extension && kk != extension)
            continue;

		fits_movabs_hdu(fptr, kk, &hdutype, &status);
		fits_get_hdu_type(fptr, &hdutype, &status);

		if (hdutype != IMAGE_HDU) {
            if (extension)
                logerr("Requested extension %i in file %s is not an image.\n", extension, infn);
			continue;
        }

		fits_get_img_dim(fptr, &naxis, &status);
        CFITS_CHECK("Failed to find image dimensions for HDU %i", kk);

		fits_get_img_size(fptr, 2, naxisn, &status);
        CFITS_CHECK("Failed to find image dimensions for HDU %i", kk);

		nimgs++;

        logverb("Got naxis=%d, na1=%lu, na2=%lu\n", naxis, naxisn[0], naxisn[1]);

        fits_get_img_type(fptr, &bitpix, &status);
        CFITS_CHECK("Failed to get FITS image type");

		fpixel = malloc(naxis * sizeof(long));
		for (a=0; a<naxis; a++)
			fpixel[a] = 1;

		if (plane && naxis == 3) {
			if (plane <= naxisn[2]) {
				logmsg("Grabbing image plane %i\n", plane);
				fpixel[2] = plane;
			} else
				logerr("Requested plane %i but only %i are available.\n", plane, (int)naxisn[2]);
		} else if (plane)
			logmsg("Plane %i requested but this image has NAXIS = %i (not 3).\n", plane, naxis);
		else if (naxis > 2)
            logmsg("This looks like a multi-color image: processing the first image plane only.  (NAXIS=%i)\n", naxis);

        if (bitpix == 8 && do_u8 && !downsample) {
			simplexy2_set_u8_defaults(&s);

            // u8 image.
            s.image_u8 = malloc(naxisn[0] * naxisn[1]);
            if (!s.image_u8) {
                SYSERROR("Failed to allocate u8 image array");
                goto bailout;
            }
            fits_read_pix(fptr, TBYTE, fpixel, naxisn[0]*naxisn[1], NULL,
                          s.image_u8, NULL, &status);

        } else {
			simplexy2_set_defaults(&s);

            s.image = malloc(naxisn[0] * naxisn[1] * sizeof(float));
            if (!s.image) {
                SYSERROR("Failed to allocate image array");
                goto bailout;
            }
            fits_read_pix(fptr, TFLOAT, fpixel, naxisn[0]*naxisn[1], NULL,
                          s.image, NULL, &status);
        }
		free(fpixel);
        CFITS_CHECK("Failed to read image pixels");

		s.bgsubimgfn = bgsubimg;
		s.bgimgfn = bgimg;
		s.maskimgfn = maskimg;
		s.nx = naxisn[0];
		s.ny = naxisn[1];

		if (plim > 0)
			s.plim = plim;
		if (halfbox > 0)
			s.halfbox = halfbox;

		image2xy_image2(&s, downsample, downsample_as_required);

		fits_create_tbl(ofptr, BINARY_TBL, s.npeaks, 4, ttype, tform,
                        tunit, "SOURCES", &status);
        CFITS_CHECK("Failed to create output table");

		fits_write_col(ofptr, TFLOAT, 1, 1, 1, s.npeaks, s.x, &status);
        CFITS_CHECK("Failed to write X column");

		fits_write_col(ofptr, TFLOAT, 2, 1, 1, s.npeaks, s.y, &status);
        CFITS_CHECK("Failed to write Y column");

		fits_write_col(ofptr, TFLOAT, 3, 1, 1, s.npeaks, s.flux, &status);
        CFITS_CHECK("Failed to write FLUX column");

		fits_write_col(ofptr, TFLOAT, 4, 1, 1, s.npeaks, s.background, &status);
        CFITS_CHECK("Failed to write BACKGROUND column");

		fits_modify_comment(ofptr, "TTYPE1", "X coordinate", &status);
        CFITS_CHECK("Failed to set X TTYPE");

		fits_modify_comment(ofptr, "TTYPE2", "Y coordinate", &status);
        CFITS_CHECK("Failed to set Y TTYPE");

		fits_modify_comment(ofptr, "TTYPE3", "Flux of source", &status);
        CFITS_CHECK("Failed to set FLUX TTYPE");

		fits_modify_comment(ofptr, "TTYPE4", "Sky background of source", &status);
        CFITS_CHECK("Failed to set BACKGROUND TTYPE");

		fits_write_key(ofptr, TINT, "SRCEXT", &kk,
                       "Extension number in src image", &status);
        CFITS_CHECK("Failed to write SRCEXT");

		w = naxisn[0];
		h = naxisn[1];
		fits_write_key(ofptr, TINT, "IMAGEW", &w, "Input image width", &status);
        CFITS_CHECK("Failed to write IMAGEW");

		fits_write_key(ofptr, TINT, "IMAGEH", &h, "Input image height", &status);
        CFITS_CHECK("Failed to write IMAGEH");

		fits_write_key(ofptr, TFLOAT, "ESTSIGMA", &(s.sigma),
				"Estimated source image variance", &status);
        CFITS_CHECK("Failed to write ESTSIGMA");

        fits_write_key(ofptr, TFLOAT, "DPSF", &(s.dpsf), "image2xy Assumed gaussian psf width", &status);
        fits_write_key(ofptr, TFLOAT, "PLIM", &(s.plim), "image2xy Significance to keep", &status);
        fits_write_key(ofptr, TFLOAT, "DLIM", &(s.dlim), "image2xy Closest two peaks can be", &status);
        fits_write_key(ofptr, TFLOAT, "SADDLE", &(s.saddle), "image2xy Saddle difference (in sig)", &status);
        fits_write_key(ofptr, TINT, "MAXPER", &(s.maxper), "image2xy Max num of peaks per object", &status);
        fits_write_key(ofptr, TINT, "MAXPEAKS", &(s.maxnpeaks), "image2xy Max num of peaks total", &status);
        fits_write_key(ofptr, TINT, "MAXSIZE", &(s.maxsize), "image2xy Max size for extended objects", &status);
        fits_write_key(ofptr, TINT, "HALFBOX", &(s.halfbox), "image2xy Half-size for sliding sky window", &status);


		fits_write_comment(ofptr,
			"The X and Y points are specified assuming 1,1 is "
			"the center of the leftmost bottom pixel of the "
			"image in accordance with the FITS standard.", &status);
        CFITS_CHECK("Failed to write comments");

		simplexy2_free_contents(&s);
	}

	// Put in the optional NEXTEND keywoard
	fits_movabs_hdu(ofptr, 1, &hdutype, &status);
	assert(hdutype == IMAGE_HDU);
	fits_write_key(ofptr, TINT, "NEXTEND", &nimgs, "Number of extensions", &status);
	if (status == END_OF_FILE)
		status = 0; /* Reset after normal error */
    CFITS_CHECK("Failed to write NEXTEND");

	fits_close_file(fptr, &status);
    CFITS_CHECK("Failed to close FITS input file");
    fptr = NULL;

	fits_close_file(ofptr, &status);
    CFITS_CHECK("Failed to close FITS output file");

    // for valgrind
	simplexy2_clean_cache();

	return 0;

 bailout:
    if (fptr)
        fits_close_file(fptr, &status);
    if (ofptr)
        fits_close_file(ofptr, &status);
    return -1;
}
